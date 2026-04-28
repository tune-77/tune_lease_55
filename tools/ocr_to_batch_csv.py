"""
ocr_to_batch_csv.py
────────────────────────────────────────────────────────────────
OCR（Googleドライブ等）で取り込んだ生データを
バッチスコアリング用CSVに自動整形するスクリプト。

使い方:
    python tools/ocr_to_batch_csv.py --input raw_ocr.csv --output batch_ready.csv

入力: Google スプレッドシートからダウンロードした CSV（列名は日本語混在でOK）
出力: batch_scoring.py の _CSV_COLUMNS と完全一致した CSV

列マッピング設定 (_COLUMN_MAP) を実際の書類に合わせて調整してください。
────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 出力CSVの列定義（batch_scoring.py の _CSV_COLUMNS と完全一致）
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
# 列マッピング設定
# キー   : 入力CSV（OCR結果）の列名（部分一致・大文字小文字無視で検索）
# 値     : OUTPUT_COLUMNS のいずれか
# ★ 実際の書類の列名に合わせてここを編集してください ★
# ─────────────────────────────────────────────────────────────────────────────
_COLUMN_MAP: dict[str, str] = {
    # 基本情報
    "取引先": "取引先ID",
    "顧客コード": "取引先ID",
    "管理番号": "取引先ID",
    "no": "取引先ID",
    "no.": "取引先ID",
    "審査日": "審査日",
    "申請日": "審査日",
    "受付日": "審査日",
    "契約日": "審査日",
    "企業名": "企業名",
    "会社名": "企業名",
    "商号": "企業名",
    "借手": "企業名",
    "業種大": "業種大分類",
    "大分類": "業種大分類",
    "業種小": "業種小分類",
    "小分類": "業種小分類",
    "取引区分": "取引区分",
    "新規": "取引区分",
    "営業部": "営業担当部署",
    "担当部署": "営業担当部署",
    "部署": "営業担当部署",
    "紹介元": "紹介元",
    "ソース": "紹介元",
    "検収": "検収時期(年)",
    "検収年": "検収時期(年)",
    # 物件情報
    "リース期間": "リース期間(月)",
    "期間": "リース期間(月)",
    "取得価格": "取得価格(千円)",
    "物件価格": "取得価格(千円)",
    "取得原価": "取得価格(千円)",
    "物件名": "物件名（任意）",
    "物件": "物件名（任意）",
    # 競合
    "競合": "競合状況",
    "競合金利": "競合提示金利(%)",
    "他社金利": "競合提示金利(%)",
    "競合先": "競合他社名",
    "他社": "競合他社名",
    "契約種別": "契約種別",
    "種別": "契約種別",
    # 財務（千円単位）
    "売上高": "売上高(千円)",
    "売上": "売上高(千円)",
    "年商": "売上高(千円)",
    "総売上": "売上高(千円)",
    "売上総利益": "売上総利益(千円)",
    "粗利": "売上総利益(千円)",
    "営業利益": "営業利益(千円)",
    "経常利益": "経常利益(千円)",
    "当期純利益": "当期純利益(千円)",
    "純利益": "当期純利益(千円)",
    "純資産": "純資産(千円)",
    "自己資本": "純資産(千円)",
    "総資産": "総資産(千円)",
    "資産合計": "総資産(千円)",
    "機械装置": "機械装置(千円)",
    "機械": "機械装置(千円)",
    "その他資産": "その他資産(千円)",
    "減価償却費": "減価償却費(千円)",
    "減価償却": "減価償却費(千円)",
    "減価償却累計": "減価償却累計(千円)",
    "累計償却": "減価償却累計(千円)",
    "支払リース": "支払リース料(千円)",
    "リース料": "支払リース料(千円)",
    "地代家賃": "地代家賃(千円)",
    "賃借料": "地代家賃(千円)",
    "銀行借入": "銀行借入(千円)",
    "借入金": "銀行借入(千円)",
    "リース残高": "リース残高(千円)",
    "既存リース": "リース残高(千円)",
    "契約件数": "契約件数",
    "件数": "契約件数",
    "格付": "格付",
    "信用格付": "格付",
    # 結果
    "最終結果": "最終結果",
    "成約": "最終結果",
    "結果": "最終結果",
    "成否": "最終結果",
}

# ─────────────────────────────────────────────────────────────────────────────
# 値の正規化ルール
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_value(col: str, val) -> str:
    """列と値に応じて値を正規化する。"""
    if pd.isna(val) or str(val).strip() in ("", "nan", "NaN", "-", "—", "ー"):
        return ""
    v = str(val).strip()

    # 数値系列：カンマ・円・千円表記を除去
    numeric_cols = {
        "売上高(千円)", "売上総利益(千円)", "営業利益(千円)", "経常利益(千円)", "当期純利益(千円)",
        "純資産(千円)", "総資産(千円)", "機械装置(千円)", "その他資産(千円)",
        "減価償却費(千円)", "減価償却累計(千円)", "支払リース料(千円)", "地代家賃(千円)",
        "銀行借入(千円)", "リース残高(千円)", "取得価格(千円)",
        "リース期間(月)", "契約件数", "検収時期(年)", "競合提示金利(%)",
        "物件スコア（任意）", "担当者直感スコア(1-5)",
    }
    if col in numeric_cols:
        # 単位変換: 万円表記を千円に（例: "1,500万" → 15000）
        man_match = re.match(r"([0-9,，.]+)\s*万", v)
        if man_match:
            num_str = man_match.group(1).replace(",", "").replace("，", "")
            try:
                return str(int(float(num_str) * 10))
            except ValueError:
                pass
        # 億円表記
        oku_match = re.match(r"([0-9,，.]+)\s*億", v)
        if oku_match:
            num_str = oku_match.group(1).replace(",", "").replace("，", "")
            try:
                return str(int(float(num_str) * 100000))
            except ValueError:
                pass
        # 通常の数値：記号除去
        v_clean = re.sub(r"[^\d.\-]", "", v)
        try:
            return str(int(float(v_clean))) if v_clean else ""
        except ValueError:
            return ""

    # 取引区分の正規化
    if col == "取引区分":
        if any(k in v for k in ["新規", "新", "新しい"]):
            return "新規先"
        if any(k in v for k in ["既存", "既", "継続"]):
            return "既存先"
        return v

    # 競合状況の正規化
    if col == "競合状況":
        if any(k in v for k in ["あり", "有", "競合", "○", "◯", "1"]):
            return "競合あり"
        if any(k in v for k in ["なし", "無", "×", "0"]):
            return "競合なし"
        return v

    # 最終結果の正規化
    if col == "最終結果":
        if any(k in v for k in ["成約", "成立", "契約", "○", "◯", "1"]):
            return "成約"
        if any(k in v for k in ["失注", "失敗", "×", "NG", "0", "未成約"]):
            return "失注"
        return ""  # 不明は空欄にして「未登録」扱い

    # 格付の正規化
    if col == "格付":
        grade_map = {
            "1": "1-3", "2": "1-3", "3": "1-3",
            "4": "4-6", "5": "4-6", "6": "4-6",
            "7": "7-8", "8": "7-8",
            "9": "9(要注意)",
            "10": "10(破綻懸念)",
            "要注意": "9(要注意)",
            "破綻": "10(破綻懸念)",
            "無格付": "無格付", "なし": "無格付",
        }
        for k, mapped in grade_map.items():
            if v == k or v.startswith(k):
                return mapped
        return v

    # 審査日の正規化（YYYY-MM-DD形式に統一）
    if col == "審査日":
        v_date = re.sub(r"[/．。]", "-", v).replace("年", "-").replace("月", "-").replace("日", "")
        try:
            from datetime import datetime
            parsed = datetime.strptime(v_date.strip()[:10], "%Y-%m-%d")
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            pass
        # YYYYMMDD形式
        if re.match(r"^\d{8}$", v_date):
            return f"{v_date[:4]}-{v_date[4:6]}-{v_date[6:8]}"
        return v

    # 紹介元の正規化
    if col == "紹介元":
        if "銀行" in v:
            return "銀行紹介"
        if "メーカー" in v or "製造" in v:
            return "メーカー紹介"
        if "ディーラー" in v or "販売" in v:
            return "ディーラー紹介"
        return "その他"

    return v


# ─────────────────────────────────────────────────────────────────────────────
# 列名マッチング（部分一致・正規化）
# ─────────────────────────────────────────────────────────────────────────────

def _find_output_col(input_col: str) -> str | None:
    """入力列名から出力列名を返す。見つからなければ None。"""
    normalized = input_col.strip().lower().replace(" ", "").replace("　", "")
    for pattern, output in _COLUMN_MAP.items():
        if pattern.lower() in normalized:
            return output
    return None


# ─────────────────────────────────────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────────────────────────────────────

def convert(input_path: str, output_path: str, encoding: str = "utf-8-sig") -> None:
    print(f"📂 入力ファイル読み込み: {input_path}")
    try:
        df_in = pd.read_csv(input_path, encoding=encoding, dtype=str)
    except UnicodeDecodeError:
        df_in = pd.read_csv(input_path, encoding="shift_jis", dtype=str)

    print(f"   → {len(df_in)} 行、{len(df_in.columns)} 列を読み込みました")

    # 列マッピング
    col_mapping: dict[str, str] = {}
    unmapped: list[str] = []
    for col in df_in.columns:
        out_col = _find_output_col(col)
        if out_col:
            if out_col not in col_mapping.values():  # 重複防止
                col_mapping[col] = out_col
                print(f"   ✅ '{col}' → '{out_col}'")
            else:
                print(f"   ⚠️  '{col}' → '{out_col}'（重複スキップ）")
        else:
            unmapped.append(col)

    if unmapped:
        print(f"\n⚠️  マッピング未定義の列（手動確認してください）:")
        for c in unmapped:
            print(f"   - {c}")

    # 出力データフレーム構築
    rows = []
    for _, row in df_in.iterrows():
        out_row: dict[str, str] = {c: "" for c in OUTPUT_COLUMNS}
        for in_col, out_col in col_mapping.items():
            raw_val = row.get(in_col, "")
            out_row[out_col] = _normalize_value(out_col, raw_val)
        rows.append(out_row)

    df_out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # 統計レポート
    filled = df_out.apply(lambda c: (c != "").sum())
    print(f"\n📊 変換結果サマリー ({len(df_out)} 件):")
    print(f"   審査日あり:     {filled.get('審査日', 0)} 件")
    print(f"   売上高あり:     {filled.get('売上高(千円)', 0)} 件")
    print(f"   最終結果あり:   {filled.get('最終結果', 0)} 件")
    result_counts = df_out["最終結果"].value_counts()
    for status, cnt in result_counts.items():
        if status:
            print(f"   　{status}: {cnt} 件")

    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 出力完了: {output_path}")
    print("   バッチスコアリング画面からこのCSVをアップロードしてください。")


def main():
    parser = argparse.ArgumentParser(description="OCRデータをバッチCSV形式に変換")
    parser.add_argument("--input",  "-i", required=True, help="入力CSVファイルパス")
    parser.add_argument("--output", "-o", default="batch_ready.csv", help="出力CSVファイルパス")
    parser.add_argument("--encoding", "-e", default="utf-8-sig", help="入力エンコーディング")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"❌ ファイルが見つかりません: {args.input}", file=sys.stderr)
        sys.exit(1)

    convert(args.input, args.output, args.encoding)


if __name__ == "__main__":
    main()
