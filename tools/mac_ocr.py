"""
mac_ocr.py
────────────────────────────────────────────────────────────
macOSのVision framework（Live Text）を使って書類写真をOCRし、
バッチスコアリング用CSVに変換するスクリプト。

APIキー不要・レート制限なし・完全無料。
macOS 13 (Ventura) 以降で日本語精度が高い。

使い方:
    python tools/mac_ocr.py --images ~/Desktop/IMG_*.JPG --output ~/Desktop/batch_ready.csv

必要なもの:
    pip install ocrmac pillow pandas
────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from pathlib import Path

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
# 1枚をOCR → テキスト行リストを返す
# ─────────────────────────────────────────────────────────────────────────────

def ocr_image(image_path: str) -> list[str]:
    """ocrmacでOCRして行リストを返す。"""
    try:
        from ocrmac import ocrmac
        annotations = ocrmac.OCR(image_path, language_preference=["ja-JP", "en-US"]).recognize()
        # annotations は (text, confidence, bbox) のタプルリスト
        # Y座標でソートして読み順に並べる
        annotations_sorted = sorted(annotations, key=lambda a: (1 - a[2][1], a[2][0]))
        return [a[0] for a in annotations_sorted if a[0].strip()]
    except ImportError:
        print("❌ ocrmac が見つかりません: pip install ocrmac")
        sys.exit(1)
    except Exception as e:
        print(f"   ⚠️  OCRエラー: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# テキスト行から表の構造を解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_table_from_lines(lines: list[str]) -> list[list[str]]:
    """OCRテキスト行を表の行×列に変換する。"""
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # タブ区切り
        if "\t" in line:
            cells = line.split("\t")
        else:
            # 半角スペース2個以上 or 全角スペースで分割
            cells = re.split(r"[\s　]{2,}", line)
        cleaned = [c.strip() for c in cells if c.strip()]
        if cleaned:
            rows.append(cleaned)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 値の正規化
# ─────────────────────────────────────────────────────────────────────────────

def normalize_number(val: str) -> str:
    """数値文字列から単位・記号を除去して千円単位の整数に変換。"""
    v = str(val).strip()
    if not v or v in ("-", "—", "ー", "△", "▲"):
        return ""
    # マイナス符号を保持
    is_negative = v.startswith(("△", "▲", "-", "－"))
    v_clean = re.sub(r"[△▲\-－]", "", v)
    # 万円 → 千円
    man = re.match(r"([0-9,，]+)\s*万", v_clean)
    if man:
        try:
            return ("-" if is_negative else "") + str(int(float(man.group(1).replace(",", "").replace("，", "")) * 10))
        except ValueError:
            pass
    # 億円 → 千円
    oku = re.match(r"([0-9,，.]+)\s*億", v_clean)
    if oku:
        try:
            return ("-" if is_negative else "") + str(int(float(oku.group(1).replace(",", "").replace("，", "")) * 100000))
        except ValueError:
            pass
    # 通常数値（カンマ除去）
    num_str = re.sub(r"[^\d.]", "", v_clean)
    try:
        return ("-" if is_negative else "") + str(int(float(num_str))) if num_str else ""
    except ValueError:
        return ""


def normalize_value(col: str, val: str) -> str:
    """列に応じた値の正規化。"""
    v = str(val).strip()
    if not v or v in ("nan", "NaN", "None", "-", "—", "ー"):
        return ""

    numeric_cols = {
        "売上高(千円)", "売上総利益(千円)", "営業利益(千円)", "経常利益(千円)", "当期純利益(千円)",
        "純資産(千円)", "総資産(千円)", "機械装置(千円)", "その他資産(千円)",
        "減価償却費(千円)", "減価償却累計(千円)", "支払リース料(千円)", "地代家賃(千円)",
        "銀行借入(千円)", "リース残高(千円)", "取得価格(千円)",
        "リース期間(月)", "契約件数", "検収時期(年)", "競合提示金利(%)", "物件スコア（任意）",
    }
    if col in numeric_cols:
        return normalize_number(v)

    if col == "取引区分":
        if any(k in v for k in ["新規", "新"]): return "新規先"
        if any(k in v for k in ["既存", "継続"]): return "既存先"

    if col == "競合状況":
        if any(k in v for k in ["あり", "有", "○", "◯"]): return "競合あり"
        if any(k in v for k in ["なし", "無", "×"]): return "競合なし"

    if col == "最終結果":
        if any(k in v for k in ["成約", "成立", "○", "◯"]): return "成約"
        if any(k in v for k in ["失注", "×", "NG", "不成立"]): return "失注"
        return ""

    if col == "審査日":
        d = re.sub(r"[/．]", "-", v).replace("年", "-").replace("月", "-").replace("日", "")
        try:
            from datetime import datetime
            return datetime.strptime(d.strip()[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            if re.match(r"^\d{8}$", d):
                return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    if col == "紹介元":
        if "銀行" in v: return "銀行紹介"
        if "メーカー" in v: return "メーカー紹介"
        if "ディーラー" in v: return "ディーラー紹介"
        return "その他"

    if col == "格付":
        gmap = {"1":"1-3","2":"1-3","3":"1-3","4":"4-6","5":"4-6","6":"4-6",
                "7":"7-8","8":"7-8","要注意":"9(要注意)","無格付":"無格付"}
        return gmap.get(v, v)

    return v


# ─────────────────────────────────────────────────────────────────────────────
# ヘッダー行から列インデックスを推定
# ─────────────────────────────────────────────────────────────────────────────

HEADER_KEYWORDS = {
    "取引先ID": ["取引先", "顧客コード", "管理番号", "no", "no."],
    "審査日":   ["審査日", "申請日", "受付日", "契約日"],
    "企業名":   ["企業名", "会社名", "商号", "借手"],
    "業種大分類": ["業種大", "大分類"],
    "業種小分類": ["業種小", "小分類"],
    "取引区分":  ["取引区分", "新規", "既存"],
    "営業担当部署": ["営業部", "部署", "担当"],
    "紹介元":    ["紹介元", "ソース"],
    "検収時期(年)": ["検収"],
    "リース期間(月)": ["リース期間", "期間"],
    "取得価格(千円)": ["取得価格", "物件価格", "取得原価"],
    "物件名（任意）": ["物件名", "物件"],
    "競合状況":  ["競合"],
    "競合提示金利(%)": ["競合金利", "他社金利"],
    "売上高(千円)": ["売上高", "売上", "年商"],
    "売上総利益(千円)": ["売上総利益", "粗利"],
    "営業利益(千円)": ["営業利益"],
    "経常利益(千円)": ["経常利益"],
    "当期純利益(千円)": ["当期純利益", "純利益"],
    "純資産(千円)": ["純資産", "自己資本"],
    "総資産(千円)": ["総資産", "資産合計"],
    "機械装置(千円)": ["機械装置", "機械"],
    "その他資産(千円)": ["その他資産"],
    "減価償却費(千円)": ["減価償却"],
    "銀行借入(千円)": ["銀行借入", "借入金"],
    "リース残高(千円)": ["リース残高"],
    "契約件数": ["契約件数", "件数"],
    "格付": ["格付"],
    "最終結果": ["最終結果", "結果", "成否"],
}


def find_header_row(rows: list[list[str]]) -> tuple[int, dict[int, str]]:
    """ヘッダー行のインデックスと列マッピングを返す。"""
    best_idx, best_map, best_score = 0, {}, 0
    for i, row in enumerate(rows[:10]):
        col_map: dict[int, str] = {}
        for j, cell in enumerate(row):
            cell_lower = cell.lower().replace(" ", "")
            for out_col, keywords in HEADER_KEYWORDS.items():
                if any(kw in cell_lower for kw in keywords):
                    if out_col not in col_map.values():
                        col_map[j] = out_col
                        break
        if len(col_map) > best_score:
            best_score = len(col_map)
            best_idx = i
            best_map = col_map
    return best_idx, best_map


# ─────────────────────────────────────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────────────────────────────────────

def process_images(image_paths: list[str], output_path: str) -> None:
    all_records: list[dict] = []
    global_col_map: dict[int, str] = {}

    for i, img_path in enumerate(image_paths, 1):
        print(f"\n📷 [{i}/{len(image_paths)}] {Path(img_path).name}")
        if not Path(img_path).exists():
            print(f"   ❌ ファイルが見つかりません")
            continue

        print(f"   🔍 OCR実行中（macOS Vision）...")
        lines = ocr_image(img_path)
        if not lines:
            print(f"   ⚠️  テキストが取得できませんでした")
            continue
        print(f"   → {len(lines)} 行のテキストを検出")

        rows = parse_table_from_lines(lines)

        # ヘッダー行検索（最初のページで確定、以降は使い回す）
        if i == 1 or not global_col_map:
            header_idx, global_col_map = find_header_row(rows)
            print(f"   → ヘッダー行: {header_idx}行目 / {len(global_col_map)}列マッピング済み")
            data_rows = rows[header_idx + 1:]
        else:
            # 2枚目以降: 先頭がヘッダーっぽければスキップ
            if rows and not any(re.search(r"\d{4,}", c) for c in rows[0]):
                data_rows = rows[1:]
            else:
                data_rows = rows

        # 各行をレコード化
        page_count = 0
        for row in data_rows:
            if not row or len(row) < 2:
                continue
            record: dict = {c: "" for c in OUTPUT_COLUMNS}
            for col_idx, out_col in global_col_map.items():
                if col_idx < len(row):
                    record[out_col] = normalize_value(out_col, row[col_idx])
            # 企業名などキー項目が空の行はスキップ
            if not record.get("企業名") and not record.get("取引先ID"):
                continue
            all_records.append(record)
            page_count += 1

        print(f"   ✅ {page_count} 件を抽出")

    if not all_records:
        print("\n⚠️  データが取得できませんでした。")
        print("   ヒント: 写真の画質・角度を改善して再試行してください。")
        print("   または raw_ocr モードで確認: --raw オプション")
        return

    df = pd.DataFrame(all_records, columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # サマリー
    print(f"\n{'━'*60}")
    print(f"📊 完了: {len(df)} 件 → {output_path}")
    vc = df["最終結果"].value_counts()
    for s in ["成約", "失注"]:
        cnt = int(vc.get(s, 0))
        if cnt:
            print(f"   {s}: {cnt} 件")
    print(f"{'━'*60}")
    print("次のステップ:")
    print("  1. Excelで開いて内容確認・修正")
    print("  2. バッチ審査画面からアップロード")
    print(f"{'━'*60}\n")


def main():
    parser = argparse.ArgumentParser(description="macOS Vision OCRでバッチCSV生成")
    parser.add_argument("--images", "-i", nargs="+", required=True)
    parser.add_argument("--output", "-o", default="batch_ready.csv")
    parser.add_argument("--raw", action="store_true", help="OCR生テキストをファイルに出力して確認する")
    args = parser.parse_args()

    import glob
    image_paths = []
    for p in args.images:
        expanded = glob.glob(p)
        image_paths.extend(sorted(expanded) if expanded else [p])

    if args.raw:
        # 生テキストを確認するデバッグモード
        raw_out = args.output.replace(".csv", "_raw.txt")
        print(f"🔍 RAWモード: OCR結果を {raw_out} に出力します\n")
        with open(raw_out, "w", encoding="utf-8") as f:
            for img_path in image_paths[:3]:  # 最初の3枚だけ
                print(f"📷 {Path(img_path).name}")
                lines = ocr_image(img_path)
                f.write(f"\n{'='*60}\n{img_path}\n{'='*60}\n")
                for i, line in enumerate(lines):
                    f.write(f"{i:03d}: {line}\n")
                    if i < 30:
                        print(f"  {i:03d}: {line}")
        print(f"\n✅ 生テキスト保存: {raw_out}")
        print("このファイルを確認して列名を教えてください。")
        return

    print(f"🖼️  {len(image_paths)} 枚の画像を処理します（APIキー不要）\n")
    process_images(image_paths, args.output)


if __name__ == "__main__":
    main()
