"""
Claude Vision API を使ってスキャンPDFの表を読み取りCSV化するスクリプト。

使い方:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 pdf_vision_ocr.py /path/to/scan.pdf --out /path/to/output.csv

オプション:
    --pages  読むページ範囲 (例: 1-3 or 1,3,5)。省略時は全ページ
    --model  使用モデル (デフォルト: claude-haiku-4-5-20251001)
    --dpi    画像解像度 (デフォルト: 200)
"""

import argparse
import base64
import csv
import io
import os
import sys
import time
from pathlib import Path

import anthropic
import fitz  # PyMuPDF


HEADERS = [
    "ユーザーコード", "発生年月日", "部署", "商談区分", "物件名",
    "業種", "銀行貸残", "リース残", "売上高", "売上高総利益",
    "営業利益", "経常利益", "当期利益", "減価償却費", "減価償却費(経費)",
    "機械・装置", "その他有形固定資産", "賃借料", "賃借料(経費)",
    "格付", "取引状態区分", "契約種類", "結果コード", "結果",
    "理由", "商談ソースコード", "商談ソース", "取得価額", "期間", "利回り",
]

SYSTEM_PROMPT = """あなたはリース会社の審査データ読み取りの専門家です。
スキャンされた表の画像から正確にデータを抽出します。
数字は記載通りに読み取ります。空欄は空文字で出力します。"""

USER_PROMPT = """この画像はリース審査データの一覧表（スキャンPDF）です。

【手順】
1. まず画像の一番上の行（ヘッダー行）にある列名を左から順に全て読んでください
2. その列順のまま、全データ行をCSVで出力してください

【重要ルール】
- ヘッダーは画像から読んだ実際の列名を使う（固定のヘッダーを使わない）
- 全行出力する（途中で省略しない）
- 空欄セルは空文字（カンマのみ）
- 金額は数字のみ（カンマ・円マーク不要）
- 格付欄：0、4、5、6 などそのまま読む（空欄は空文字）
- 期間欄：60、48 などの数字のみ
- 利回り欄：3.08% のように%付きで読む
- CSV以外の説明文は不要

出力形式（例）:
```csv
ユーザーコード,発生年月日,部署,商談区分,物件名,...
316961,2025-12-05,埼玉営業部,1,トラック,...
```
"""


def page_to_base64(doc: fitz.Document, page_idx: int, dpi: int = 200) -> str:
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    return base64.standard_b64encode(img_bytes).decode("utf-8")


def extract_csv_block(text: str) -> str:
    """```csv ... ``` ブロックを抽出。なければそのまま返す。"""
    if "```csv" in text:
        start = text.index("```csv") + 6
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()
    return text.strip()


def read_page(client: anthropic.Anthropic, img_b64: str, model: str, page_num: int) -> list[dict]:
    print(f"  API送信中...", end="", flush=True)
    t0 = time.time()

    message = client.messages.create(
        model=model,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            }
        ],
    )

    elapsed = time.time() - t0
    print(f" {elapsed:.1f}秒")

    raw_text = message.content[0].text
    csv_text = extract_csv_block(raw_text)

    rows = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        # ヘッダーが一致しない列は無視・不足列は空で補完
        normalized = {h: row.get(h, "") for h in HEADERS}
        rows.append(normalized)

    print(f"  → {len(rows)}行読み取り完了")
    return rows


def parse_pages(pages_arg: str, total: int) -> list[int]:
    if not pages_arg:
        return list(range(total))
    result = []
    for part in pages_arg.split(","):
        if "-" in part:
            a, b = part.split("-")
            result.extend(range(int(a) - 1, int(b)))
        else:
            result.append(int(part) - 1)
    return [p for p in result if 0 <= p < total]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="入力PDFパス")
    parser.add_argument("--out", help="出力CSVパス")
    parser.add_argument("--pages", help="ページ範囲 (例: 1-3 or 1,3,5)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("エラー: ANTHROPIC_API_KEY 環境変数が未設定です")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    pdf_path = Path(args.pdf)
    out_path = Path(args.out) if args.out else pdf_path.with_suffix("").parent / (pdf_path.stem + "-vision.csv")

    client = anthropic.Anthropic(api_key=api_key)
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    page_indices = parse_pages(args.pages, total_pages)
    print(f"PDF: {pdf_path.name} ({total_pages}ページ)")
    print(f"読み取りページ: {[p+1 for p in page_indices]}")
    print(f"モデル: {args.model}, DPI: {args.dpi}")
    print()

    all_rows = []
    for i, page_idx in enumerate(page_indices):
        print(f"[{i+1}/{len(page_indices)}] ページ {page_idx+1} 処理中...")
        img_b64 = page_to_base64(doc, page_idx, args.dpi)
        rows = read_page(client, img_b64, args.model, page_idx + 1)
        all_rows.extend(rows)
        if i < len(page_indices) - 1:
            time.sleep(0.5)  # レート制限対策

    print(f"\n合計 {len(all_rows)} 行")

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"出力: {out_path}")


if __name__ == "__main__":
    main()
