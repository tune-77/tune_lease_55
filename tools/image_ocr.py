"""
image_ocr.py
────────────────────────────────────────────────────────────
書類写真（JPG/PNG）を日本語OCRしてCSVに変換するスクリプト。
Mac標準のVision frameworkを優先使用。なければpytesseractを使用。

使い方:
    python tools/image_ocr.py --images *.jpg --output raw_ocr.csv

必要なもの（どちらか）:
  [推奨] macOS 13以降: 追加インストール不要
  [代替] pip install pytesseract pillow
         brew install tesseract tesseract-lang
────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import argparse
import csv
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# OCRエンジン: macOS Vision framework（最高精度・追加インストール不要）
# ─────────────────────────────────────────────────────────────────────────────

VISION_SCRIPT = """
import sys
import Quartz
from Vision import VNRecognizeTextRequest, VNImageRequestHandler
from Foundation import NSURL

image_path = sys.argv[1]
url = NSURL.fileURLWithPath_(image_path)
handler = VNImageRequestHandler.alloc().initWithURL_options_(url, {})
request = VNRecognizeTextRequest.alloc().init()
request.setRecognitionLanguages_(["ja-JP", "en-US"])
request.setUsesLanguageCorrection_(True)
request.setRecognitionLevel_(1)  # accurate
handler.performRequests_error_([request], None)
observations = request.results()
lines = []
for obs in observations:
    candidate = obs.topCandidates_(1)[0]
    lines.append(candidate.string())
print("\\n".join(lines))
"""


def ocr_with_vision(image_path: str) -> str:
    """macOS Vision framework でOCRを実行して文字列を返す。"""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
        f.write(VISION_SCRIPT)
        script_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, script_path, image_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return ""
    finally:
        os.unlink(script_path)


def is_vision_available() -> bool:
    """macOS Vision frameworkが使えるか確認。"""
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from Vision import VNRecognizeTextRequest"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# OCRエンジン: pytesseract（フォールバック）
# ─────────────────────────────────────────────────────────────────────────────

def ocr_with_tesseract(image_path: str) -> str:
    """pytesseractでOCRを実行して文字列を返す。"""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        # 前処理：コントラスト強化
        try:
            from PIL import ImageEnhance, ImageFilter
            img = img.convert("L")  # グレースケール
            img = ImageEnhance.Contrast(img).enhance(2.0)
            img = img.filter(ImageFilter.SHARPEN)
        except Exception:
            pass
        text = pytesseract.image_to_string(img, lang="jpn", config="--psm 6")
        return text.strip()
    except ImportError:
        print("❌ pytesseract が見つかりません。")
        print("   インストール: pip install pytesseract pillow")
        print("   日本語パック: brew install tesseract-lang")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# テキスト→行列変換（タブ・スペース・罫線文字で分割）
# ─────────────────────────────────────────────────────────────────────────────

def text_to_rows(text: str) -> list[list[str]]:
    """OCRテキストを行×列のリストに変換する。"""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # タブ区切りまたはスペース2個以上で列分割
        if "\t" in line:
            cells = line.split("\t")
        else:
            import re
            cells = re.split(r"\s{2,}", line)
        rows.append([c.strip() for c in cells if c.strip()])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

def process_images(image_paths: list[str], output_path: str) -> None:
    # OCRエンジン選択
    if is_vision_available():
        print("✅ macOS Vision framework を使用します（高精度）")
        ocr_func = ocr_with_vision
    else:
        print("⚠️  pytesseract を使用します（Vision framework 未検出）")
        ocr_func = ocr_with_tesseract

    all_rows: list[list[str]] = []
    header_found = False

    for i, img_path in enumerate(image_paths, 1):
        print(f"\n📷 [{i}/{len(image_paths)}] {Path(img_path).name} を処理中...")
        if not Path(img_path).exists():
            print(f"   ❌ ファイルが見つかりません: {img_path}")
            continue

        text = ocr_func(img_path)
        if not text:
            print(f"   ⚠️  テキストが取得できませんでした")
            continue

        rows = text_to_rows(text)
        print(f"   → {len(rows)} 行を検出")

        # 最初のページのヘッダー行を保持
        if not header_found and rows:
            # 最も列数が多い行をヘッダー候補として使用
            max_cols = max(len(r) for r in rows[:5]) if rows else 0
            if max_cols >= 3:
                header_found = True
                all_rows.extend(rows)
            else:
                all_rows.extend(rows)
        else:
            # 2枚目以降: 最初の行がヘッダーと似ていれば除外
            if rows and len(all_rows) > 0:
                # 1行目が数字を含まない場合はヘッダーとみなしてスキップ
                import re
                first_row = rows[0]
                has_numbers = any(re.search(r"\d{3,}", cell) for cell in first_row)
                if not has_numbers and len(first_row) >= 3:
                    rows = rows[1:]  # ヘッダーをスキップ
            all_rows.extend(rows)

    if not all_rows:
        print("\n❌ OCRでデータが取得できませんでした。")
        print("   写真の画質・角度を改善してから再試行してください。")
        return

    # CSV出力
    print(f"\n📝 CSV書き出し: {output_path}")
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    print(f"✅ 完了: {len(all_rows)} 行を書き出しました → {output_path}")
    print()
    print("━" * 60)
    print("次のステップ:")
    print(f"  1. {output_path} をExcel/Numbers で開いて列を確認・修正")
    print(f"  2. 列名を整えたら以下を実行:")
    print(f"     python tools/ocr_to_batch_csv.py --input {output_path} --output batch_ready.csv")
    print(f"  3. batch_ready.csv をバッチ審査画面にアップロード")
    print("━" * 60)


def main():
    parser = argparse.ArgumentParser(description="書類写真を日本語OCRしてCSV出力")
    parser.add_argument(
        "--images", "-i", nargs="+", required=True,
        help="入力画像ファイル（例: *.jpg または 1.jpg 2.jpg ...）"
    )
    parser.add_argument(
        "--output", "-o", default="raw_ocr.csv",
        help="出力CSVファイルパス（デフォルト: raw_ocr.csv）"
    )
    args = parser.parse_args()

    # glob展開（シェルが展開していない場合の対応）
    import glob
    image_paths = []
    for pattern in args.images:
        expanded = glob.glob(pattern)
        image_paths.extend(sorted(expanded) if expanded else [pattern])

    if not image_paths:
        print("❌ 画像ファイルが見つかりません", file=sys.stderr)
        sys.exit(1)

    print(f"🖼️  {len(image_paths)} 枚の画像を処理します")
    process_images(image_paths, args.output)


if __name__ == "__main__":
    main()
