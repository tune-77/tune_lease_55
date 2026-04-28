"""
gemini_ocr.py
────────────────────────────────────────────────────────────
Gemini Vision APIで書類写真を解析し、
バッチスコアリング用CSVに直接変換するスクリプト。

SDKを使わずHTTP直接呼び出しのため protobuf / tensorflow と競合しない。

使い方:
    python tools/gemini_ocr.py --images sheet1.jpg sheet2.jpg ... --output batch_ready.csv

必要なもの:
    pip install pillow requests   ← これだけ。競合なし。
    GEMINI_API_KEY 環境変数（または --api-key オプション）
────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 出力列定義（batch_scoring.py の _CSV_COLUMNS と完全一致）
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_COLUMNS = [
    "取引先ID", "審査日", "企業名", "業種大分類", "業種小分類",
    "取引区分", "営業担当部署", "紹介元", "検収時期(年)",
    "リース期間(月)", "取得価格(千円)", "物件ID（任意）", "物件名（任意）", "物件スコア（任意）",
    "競合状況", "競合提示金利(%)", "競合他社名", "契約種別",
    "売上高(千円)", "売上総利益(千円)", "営業利益(千円)", "経常利益(千円)", "当期純利益(千円)",
    "純資産(千円)", "総資産(千円)", "機械装置(千円)", "その他資産(千円)",
    "減価償却費(千円)", "減価償却累計(千円)", "支払リース料(千円)", "地代家賃(千円)",
    "銀行借入(千円)", "リース残高(千円)", "契約件数", "格付",
    "定性_設立経営年数", "定性_顧客安定性", "定性_返済履歴",
    "定性_事業将来性", "定性_設備目的", "定性_メイン取引銀行",
    "強みタグ", "担当者直感スコア(1-5)", "特記事項", "最終結果",
]

# ─────────────────────────────────────────────────────────────────────────────
# Geminiへ渡すプロンプト
# ─────────────────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = f"""
あなたはリース審査書類の解析専門家です。
この画像は日本のリース会社の審査記録表です。

以下の手順で表を解析してください:

【手順】
1. 画像内の表のヘッダー行を特定する
2. 各行のデータを正確に読み取る
3. 各列を以下の出力列にマッピングする（該当なしは空欄）
4. 結果をJSON配列で返す

【出力列】（この順番で必ず出力）:
{json.dumps(OUTPUT_COLUMNS, ensure_ascii=False)}

【列マッピングのヒント】
- 「取引先」「顧客コード」「管理番号」→ 取引先ID
- 「申請日」「受付日」「審査日」「契約日」→ 審査日（YYYY-MM-DD形式）
- 「企業名」「会社名」「商号」「借手」→ 企業名
- 「年商」「売上」「売上高」→ 売上高(千円)
- 「営業利益」→ 営業利益(千円)
- 「経常利益」→ 経常利益(千円)
- 「純利益」「当期純利益」→ 当期純利益(千円)
- 「純資産」「自己資本」→ 純資産(千円)
- 「総資産」「資産合計」→ 総資産(千円)
- 「銀行借入」「借入金」→ 銀行借入(千円)
- 「リース残高」「既存リース」→ リース残高(千円)
- 「取得価格」「物件価格」→ 取得価格(千円)
- 「リース期間」→ リース期間(月)（月数で）
- 「成約」「失注」「○」「×」→ 最終結果（"成約"か"失注"のみ）
- 金額が万円表記の場合は千円に変換（例: 1500万 → 15000）
- 金額が億円表記の場合は千円に変換（例: 1億 → 100000）

【注意】
- 数値はカンマ・単位（円、万、千円等）を除いた数字のみ
- 読み取れない・該当しない項目は空欄（""）
- 必ず全行を取得すること（ヘッダー行は除く）
- 最終結果は「成約」または「失注」のみ（不明は空欄）

【出力形式】（JSON配列のみ、他の説明文は不要）:
[
  {{"取引先ID": "...", "審査日": "YYYY-MM-DD", "企業名": "...", ...}},
  ...
]
"""


# ─────────────────────────────────────────────────────────────────────────────
# 画像をbase64エンコード
# ─────────────────────────────────────────────────────────────────────────────

def encode_image(image_path: str) -> tuple[str, str]:
    """画像をbase64エンコードしてMIMEタイプと一緒に返す。"""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".webp": "image/webp", ".heic": "image/heic", ".heif": "image/heif"}
    mime = mime_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, mime


# ─────────────────────────────────────────────────────────────────────────────
# Gemini APIで1枚を解析
# ─────────────────────────────────────────────────────────────────────────────

def analyze_image_with_gemini(image_path: str, api_key: str, model: str) -> list[dict]:
    """1枚の画像をGemini REST APIで解析してレコードリストを返す。SDKなし・競合なし。"""
    from PIL import Image as PILImage
    import io

    print(f"   🤖 Gemini解析中...")
    img = PILImage.open(image_path)

    # 大きすぎる場合はリサイズ（APIの4MB制限対策）
    max_size = 3000
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size)
        print(f"   📐 画像リサイズ: {img.size}")

    # JPEG形式でbase64エンコード
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    image_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    # Gemini REST APIリクエスト
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [
                {"text": EXTRACTION_PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,   # 低めで安定した抽出
            "maxOutputTokens": 8192,
        }
    }

    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        print(f"   ❌ APIエラー {resp.status_code}: {resp.text[:300]}")
        return []

    raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    # JSON部分を抽出
    import re
    json_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not json_match:
        print(f"   ⚠️  JSON形式で返ってきませんでした:")
        print(raw_text[:300])
        return []

    try:
        records = json.loads(json_match.group())
        print(f"   ✅ {len(records)} 件を抽出")
        return records
    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSONパースエラー: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gemini Visionで書類写真をバッチCSVに変換")
    parser.add_argument("--images", "-i", nargs="+", required=True, help="入力画像ファイル")
    parser.add_argument("--output", "-o", default="batch_ready.csv", help="出力CSVファイル")
    parser.add_argument("--api-key", "-k", help="Gemini APIキー（省略時はGEMINI_API_KEY環境変数）")
    parser.add_argument("--model", "-m", default="gemini-2.0-flash", help="使用モデル名")
    parser.add_argument("--delay", "-d", type=float, default=2.0, help="APIリクエスト間の待機秒数")
    args = parser.parse_args()

    # APIキー設定
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY が設定されていません。")
        print("   設定方法: export GEMINI_API_KEY='your_key_here'")
        print("   または: --api-key オプションで指定")
        sys.exit(1)

    # PIL確認（唯一の必須ライブラリ）
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("❌ Pillow が見つかりません: pip install pillow")
        sys.exit(1)

    print(f"✅ Gemini REST API ({args.model}) を使用します（SDK不要・依存競合なし）")

    # 画像ファイル展開
    import glob
    image_paths = []
    for pattern in args.images:
        expanded = glob.glob(pattern)
        image_paths.extend(sorted(expanded) if expanded else [pattern])

    print(f"\n🖼️  {len(image_paths)} 枚の画像を処理します\n")

    # 全画像を処理
    all_records: list[dict] = []
    failed: list[str] = []

    for i, img_path in enumerate(image_paths, 1):
        print(f"📷 [{i}/{len(image_paths)}] {Path(img_path).name}")
        if not Path(img_path).exists():
            print(f"   ❌ ファイルが見つかりません")
            failed.append(img_path)
            continue
        try:
            records = analyze_image_with_gemini(img_path, api_key, args.model)
            # 出典情報を付加
            for r in records:
                r["_source_file"] = Path(img_path).name
            all_records.extend(records)
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            failed.append(img_path)

        # レート制限対策
        if i < len(image_paths):
            time.sleep(args.delay)

    if not all_records:
        print("\n❌ データが取得できませんでした。")
        sys.exit(1)

    # DataFrame構築（OUTPUT_COLUMNSに合わせる）
    rows = []
    for rec in all_records:
        row = {}
        for col in OUTPUT_COLUMNS:
            val = rec.get(col, "")
            row[col] = "" if (val is None or str(val).strip() in ("nan", "None", "-", "—")) else str(val).strip()
        rows.append(row)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # 統計レポート
    print(f"\n{'━'*60}")
    print(f"📊 抽出結果サマリー: 合計 {len(df)} 件")
    result_vc = df["最終結果"].value_counts()
    for status in ["成約", "失注", ""]:
        cnt = int(result_vc.get(status, 0))
        label = status if status else "不明（未登録扱い）"
        if cnt > 0:
            print(f"   {label}: {cnt} 件")
    filled_nenshu = (df["売上高(千円)"] != "").sum()
    filled_date   = (df["審査日"] != "").sum()
    print(f"   売上高あり: {filled_nenshu} 件 / 審査日あり: {filled_date} 件")
    if failed:
        print(f"\n⚠️  処理失敗: {len(failed)} ファイル")
        for f in failed:
            print(f"   - {f}")

    # CSV出力
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n✅ 出力完了: {args.output}")
    print(f"{'━'*60}")
    print("次のステップ:")
    print(f"  1. {args.output} をExcelで開いて内容を確認・修正")
    print(f"  2. バッチ審査画面からアップロード → DB保存 → 係数自動再学習")
    print(f"{'━'*60}\n")


if __name__ == "__main__":
    main()
