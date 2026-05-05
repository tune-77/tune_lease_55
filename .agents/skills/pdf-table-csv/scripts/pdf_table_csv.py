#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

from industry_normalizer import normalize_industry_major, normalize_industry_sub


HEADERS = [
    "ユーザーコード",
    "発生年月日",
    "部署",
    "商談区分",
    "物件名",
    "業種",
    "銀行貸残",
    "リース残",
    "売上高",
    "売上高総利益",
    "営業利益",
    "経常利益",
    "当期利益",
    "減価償却費",
    "減価償却費(経費)",
    "機械・装置",
    "その他有形固定資産",
    "賃借料",
    "賃借料(経費)",
    "格付",
    "取引状態区分",
    "契約種類",
    "結果コード",
    "結果",
    "理由",
    "商談ソースコード",
    "商談ソース",
    "取得価額",
    "期間",
    "利回り",
]

AMOUNT_COLS = {
    "銀行貸残",
    "リース残",
    "売上高",
    "売上高総利益",
    "営業利益",
    "経常利益",
    "当期利益",
    "減価償却費",
    "減価償却費(経費)",
    "機械・装置",
    "その他有形固定資産",
    "賃借料",
    "賃借料(経費)",
    "取得価額",
}

BATCH_TEMPLATE_MAP = {
    "取引先ID": "ユーザーコード",
    "業種大分類": "業種",
    "取引区分": "取引状態区分",
    "営業担当部署": "部署",
    "リース期間(月)": "期間",
    "取得価格(百万円)": "取得価額",
    "物件名（任意）": "物件名",
    "売上高(百万円)": "売上高",
    "売上総利益(百万円)": "売上高総利益",
    "営業利益(百万円)": "営業利益",
    "経常利益(百万円)": "経常利益",
    "当期純利益(百万円)": "当期利益",
    "機械装置(百万円)": "機械・装置",
    "その他資産(百万円)": "その他有形固定資産",
    "減価償却費(百万円)": "減価償却費",
    "減価償却累計(百万円)": "減価償却費(経費)",
    "支払リース料(百万円)": "賃借料",
    "地代家賃(百万円)": "賃借料(経費)",
    "銀行借入(百万円)": "銀行貸残",
    "リース残高(百万円)": "リース残",
    "格付": "格付",
}


def split_pdf(src: Path, out_dir: Path | None = None) -> list[Path]:
    from pypdf import PdfReader, PdfWriter

    out_dir = out_dir or src.parent
    reader = PdfReader(str(src))
    outputs: list[Path] = []
    for i, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)
        out = out_dir / f"{src.stem}-page-{i:03d}.pdf"
        with out.open("wb") as f:
            writer.write(f)
        outputs.append(out)
    return outputs


def render_page_pdf(pdf: Path, out_png: Path) -> Path:
    import fitz
    from PIL import Image

    doc = fitz.open(str(pdf))
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    raw = out_png.with_name(out_png.stem + "-raw.png")
    pix.save(str(raw))
    im = Image.open(raw)
    # The source scan tables are landscape inside portrait PDF pages.
    oriented = out_png.with_name(out_png.stem + "-oriented.png")
    im.rotate(-90, expand=True).save(oriented)
    return preprocess_table_image(oriented, out_png)


def preprocess_table_image(image_path: Path, out_png: Path) -> Path:
    """Deskew and crop the table before grid detection.

    Scanned pages often have a small rotation. If we cut cells from the skewed
    image directly, the narrow left-side columns lose the first digit/date
    character. This step estimates the dominant horizontal rule angle, rotates
    the page, then crops to the table's rule bounding box with a small margin.
    """
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

    # Use long horizontal rules, not text strokes, to estimate page skew.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    horizontal = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel)
    lines = cv2.HoughLinesP(
        horizontal,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=max(120, image.shape[1] // 8),
        maxLineGap=20,
    )
    angles: list[float] = []
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(float, line)
            if x2 == x1:
                continue
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(angle) <= 5:
                angles.append(angle)
    skew = float(np.median(angles)) if angles else 0.0

    if abs(skew) >= 0.05:
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, skew, 1.0)
        image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

    # Crop to the table rules. This removes handwritten page marks and margins.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel)
    rule_mask = cv2.bitwise_or(h_lines, v_lines)
    contours, _ = cv2.findContours(rule_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > image.shape[1] * 0.15 or h > image.shape[0] * 0.15:
            boxes.append((x, y, x + w, y + h))
    if boxes:
        x1 = max(0, min(b[0] for b in boxes) - 20)
        y1 = max(0, min(b[1] for b in boxes) - 20)
        x2 = min(image.shape[1], max(b[2] for b in boxes) + 20)
        y2 = min(image.shape[0], max(b[3] for b in boxes) + 20)
        image = image[y1:y2, x1:x2]

    cv2.imwrite(str(out_png), image)
    return out_png


def detect_grid(image_path: Path) -> tuple[list[int], list[int]]:
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cannot read image: {image_path}")
    _, th = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    h = cv2.morphologyEx(
        th,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)),
    )
    v = cv2.morphologyEx(
        th,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20)),
    )

    def clusters(proj, thresh: int) -> list[int]:
        idx = np.where(proj > thresh)[0]
        if len(idx) == 0:
            return []
        res = []
        start = prev = int(idx[0])
        for val in idx[1:]:
            val = int(val)
            if val - prev > 3:
                res.append((start + prev) // 2)
                start = val
            prev = val
        res.append((start + prev) // 2)
        return res

    ys = clusters(h.sum(axis=1), 255 * 120)
    xs = clusters(v.sum(axis=0), 255 * 20)
    return ys, xs


def ocr_page_to_csv(pdf: Path, out_csv: Path) -> int:
    from PIL import Image, ImageEnhance, ImageOps
    from ocrmac import ocrmac

    work = Path(tempfile.mkdtemp(prefix="pdf_table_csv_"))
    image_path = render_page_pdf(pdf, work / f"{pdf.stem}.png")
    ys, xs = detect_grid(image_path)
    if len(xs) - 1 != len(HEADERS):
        raise RuntimeError(f"expected {len(HEADERS)} columns, detected {len(xs) - 1}")

    im = Image.open(image_path).convert("L")
    rows = []
    cell_dir = work / "cells"
    cell_dir.mkdir(exist_ok=True)
    for r in range(len(ys) - 1):
        row = []
        for c in range(len(xs) - 1):
            x1, x2 = xs[c] + 2, xs[c + 1] - 2
            y1, y2 = ys[r] + 2, ys[r + 1] - 2
            crop = im.crop((x1, y1, x2, y2))
            crop = ImageOps.autocontrast(crop)
            crop = ImageEnhance.Contrast(crop).enhance(1.8)
            crop = crop.resize((max(1, crop.width * 5), max(1, crop.height * 5)))
            cell_path = cell_dir / f"cell_{r:02d}_{c:02d}.png"
            crop.save(cell_path)
            try:
                ann = ocrmac.OCR(
                    str(cell_path),
                    language_preference=["ja-JP", "en-US"],
                ).recognize()
                text = " ".join(a[0] for a in ann).strip()
            except Exception:
                text = ""
            row.append(text)
        rows.append(row)
        print(f"row {r + 1}/{len(ys) - 1}", flush=True)

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(rows)
    return len(rows)


def backup_csv(path: Path, suffix: str) -> Path:
    backup = path.with_name(f"{path.stem}_before_{suffix}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def clean_num(value: str) -> str:
    s = (value or "").strip()
    s = s.translate(
        str.maketrans(
            {
                "Ｏ": "0",
                "O": "0",
                "o": "0",
                "D": "0",
                "〇": "0",
                "ｏ": "0",
                "ー": "-",
                "−": "-",
                "，": ",",
                "．": ".",
            }
        )
    )
    return re.sub(r"[^0-9.%-]", "", s.replace(",", "").replace(" ", ""))


def scale_value(value: str) -> str:
    s = clean_num(value).replace("%", "")
    if not s:
        return ""
    try:
        scaled = Decimal(s) / Decimal("1000")
    except InvalidOperation:
        return value
    out = format(scaled.normalize(), "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    return out or "0"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def scale_million(path: Path) -> None:
    backup_csv(path, "million_unit")
    headers, rows = read_csv(path)
    for row in rows:
        for col in AMOUNT_COLS:
            if col in row:
                row[col] = scale_value(row[col])
    write_csv(path, headers, rows)


def convert_rating(value: str) -> str:
    s = (value or "").strip().translate(
        str.maketrans(
            {
                "１": "1",
                "２": "2",
                "３": "3",
                "４": "4",
                "５": "5",
                "６": "6",
                "７": "7",
                "８": "8",
                "０": "0",
                "Ｄ": "D",
                "Ｕ": "U",
                "七": "7",
                "ｌ": "l",
                "Ｉ": "I",
                "＿": "_",
                "－": "-",
                "ー": "-",
                "−": "-",
            }
        )
    )
    compact = re.sub(r"\s+", "", s)
    if compact == "" or compact in {"0", "D", "O", "U"}:
        return "無格付"
    if compact in {"l", "I", "|"}:
        return "2"
    if compact == "7":
        return "要注意先"
    if compact == "10_1":
        return "要注意先"
    if compact in {"8", "82", "8_2", "8-2", "8..", "8.2"} or compact.startswith("8"):
        return "要注意先"
    if compact in {"2", "3", "4", "5", "6"}:
        return compact
    if compact == "1":
        return "2"
    match = re.search(r"[23456]", compact)
    if match:
        return match.group(0)
    return value


def normalize_department(value: str) -> str:
    """Normalize OCR text to one of the supported sales departments.

    Unknown OCR noise is returned as blank so the template-fill step writes 0.
    """
    s = (value or "").strip()
    if not s:
        return ""
    compact = re.sub(r"\s+", "", s)
    compact = compact.translate(
        str.maketrans(
            {
                "營": "営",
                "業": "業",
                "部": "部",
                "官": "宮",
                "宮": "宮",
                "玉": "玉",
                "王": "玉",
            }
        )
    )
    known = {
        "宇都宮営業部": ("宇都宮", "宇都官", "都宮"),
        "小山営業部": ("小山",),
        "足利営業部": ("足利",),
        "埼玉営業部": ("埼玉", "埼王"),
    }
    for dept, aliases in known.items():
        if dept in compact or any(alias in compact for alias in aliases):
            return dept
    return ""


def normalize_rating(path: Path) -> None:
    backup_csv(path, "rating_normalize")
    headers, rows = read_csv(path)
    for row in rows:
        row["格付"] = convert_rating(row.get("格付", ""))
    write_csv(path, headers, rows)


def normalize_common_ocr(path: Path) -> None:
    headers, rows = read_csv(path)
    for row in rows:
        row["部署"] = normalize_department(row.get("部署", ""))
        deal = (row.get("商談区分") or "") + " " + (row.get("契約種類") or "")
        if "自動車" in deal:
            row["商談区分"] = "自動車"
        elif "割" in deal:
            row["商談区分"] = "割賦"
        else:
            row["商談区分"] = "一般"
        state = row.get("取引状態区分") or ""
        if state:
            row["取引状態区分"] = "新規先" if "新" in state else "既存先"
        contract = row.get("契約種類") or ""
        if "自動車" in contract:
            row["契約種類"] = "自動車リース"
        elif "割" in contract:
            row["契約種類"] = "割賦"
        elif contract:
            row["契約種類"] = "一般リース"
        result = row.get("結果") or ""
        if "失" in result:
            row["結果"] = "失注"
        elif result:
            row["結果"] = "当社受注"
        source = row.get("商談ソース") or ""
        if "銀" in source:
            row["商談ソース"] = "銀行"
        elif source:
            row["商談ソース"] = "ユーザー"
        if clean_num(row.get("期間", "")) == "09":
            row["期間"] = "60"
        elif row.get("期間"):
            row["期間"] = clean_num(row["期間"])
        if row.get("利回り"):
            row["利回り"] = clean_num(row["利回り"]).rstrip("%") + "%"
    write_csv(path, headers, rows)


def normalize_date(value: str) -> str:
    from datetime import datetime

    s = (value or "").strip()
    if not s:
        return "0"
    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    match = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})$", s)
    if match:
        year, month, day = match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return s


def year_from_date(value: str) -> str:
    date = normalize_date(value)
    return date[:4] if re.match(r"^\d{4}-", date) else "0"


def nonblank_or_zero(row: dict[str, str], key: str) -> str:
    value = (row.get(key) or "").strip()
    return value if value else "0"


def map_source(value: str) -> str:
    if "銀行" in (value or ""):
        return "銀行紹介"
    return "その他" if value else "0"


def map_contract(value: str) -> str:
    if "自動車" in (value or ""):
        return "自動車"
    if "割" in (value or ""):
        return "割賦"
    return "一般" if value else "0"


def map_result(value: str) -> str:
    if "失" in (value or ""):
        return "失注"
    if "受注" in (value or "") or "成" in (value or ""):
        return "成約"
    return value or "0"


def map_rate(value: str) -> str:
    rate = (value or "").strip().replace("%", "")
    return rate if rate else "0"


def fill_batch_template(template_csv: Path, source_csv: Path, out_csv: Path | None = None) -> None:
    """Fill the current batch screening template from a 30-column OCR CSV.

    The template header is preserved. Columns not available in the OCR CSV are
    filled with "0", matching the user's operational convention for blanks.
    """
    out_csv = out_csv or template_csv
    backup_csv(out_csv, "template_fill") if out_csv.exists() else None

    with template_csv.open(encoding="utf-8-sig", newline="") as f:
        template_headers = next(csv.reader(f))
    _, source_rows = read_csv(source_csv)

    output_rows = []
    for src in source_rows:
        row = {header: "0" for header in template_headers}
        for template_col, source_col in BATCH_TEMPLATE_MAP.items():
            if template_col in row:
                row[template_col] = nonblank_or_zero(src, source_col)
        if "業種大分類" in row:
            row["業種大分類"] = normalize_industry_major(src.get("業種大分類")) or nonblank_or_zero(src, "業種大分類")
        if "業種小分類" in row:
            row["業種小分類"] = normalize_industry_sub(src.get("業種小分類"), src.get("業種大分類")) or "0"
        if "審査日" in row:
            row["審査日"] = normalize_date(src.get("発生年月日", ""))
        if "検収時期(年)" in row:
            row["検収時期(年)"] = year_from_date(src.get("発生年月日", ""))
        if "紹介元" in row:
            row["紹介元"] = map_source(src.get("商談ソース", ""))
        if "契約種別" in row:
            row["契約種別"] = map_contract(src.get("契約種類", ""))
        if "最終結果" in row:
            row["最終結果"] = map_result(src.get("結果", ""))
        if "獲得レート(%)" in row:
            row["獲得レート(%)"] = map_rate(src.get("利回り", ""))
        if "失注理由" in row:
            row["失注理由"] = nonblank_or_zero(src, "理由")
        if "結果登録メモ" in row:
            row["結果登録メモ"] = nonblank_or_zero(src, "理由")
        output_rows.append(row)

    write_csv(out_csv, template_headers, output_rows)


def command_all(src: Path, page: int, out: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="pdf_table_csv_all_") as tmp:
        pages = split_pdf(src, Path(tmp))
        if page < 1 or page > len(pages):
            raise RuntimeError(f"page must be 1..{len(pages)}")
        ocr_page_to_csv(pages[page - 1], out)
    normalize_common_ocr(out)
    scale_million(out)
    normalize_rating(out)


def command_pipeline(
    src_pdf: Path,
    template_csv: Path,
    page: int,
    csv_out: Path | None = None,
    out_template: Path | None = None,
) -> tuple[Path, Path]:
    """Run the full PDF -> normalized OCR CSV -> batch template workflow."""
    csv_out = csv_out or template_csv.with_name(f"{src_pdf.stem}-page-{page:03d}.csv")
    out_template = out_template or template_csv
    command_all(src_pdf, page, csv_out)
    fill_batch_template(template_csv, csv_out, out_template)
    return csv_out, out_template


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_split = sub.add_parser("split")
    p_split.add_argument("pdf", type=Path)
    p_split.add_argument("--out-dir", type=Path)

    p_ocr = sub.add_parser("ocr-page")
    p_ocr.add_argument("pdf", type=Path)
    p_ocr.add_argument("--out", type=Path, required=True)

    p_scale = sub.add_parser("scale-million")
    p_scale.add_argument("csv", type=Path)

    p_rating = sub.add_parser("normalize-rating")
    p_rating.add_argument("csv", type=Path)

    p_fill = sub.add_parser("fill-template")
    p_fill.add_argument("template_csv", type=Path)
    p_fill.add_argument("source_csv", type=Path)
    p_fill.add_argument("--out", type=Path)

    p_all = sub.add_parser("all")
    p_all.add_argument("pdf", type=Path)
    p_all.add_argument("--page", type=int, default=1)
    p_all.add_argument("--out", type=Path, required=True)

    p_pipeline = sub.add_parser("pipeline")
    p_pipeline.add_argument("pdf", type=Path)
    p_pipeline.add_argument("template_csv", type=Path)
    p_pipeline.add_argument("--page", type=int, default=1)
    p_pipeline.add_argument("--csv-out", type=Path)
    p_pipeline.add_argument("--out-template", type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "split":
        for path in split_pdf(args.pdf, args.out_dir):
            print(path)
    elif args.cmd == "ocr-page":
        rows = ocr_page_to_csv(args.pdf, args.out)
        print(f"{args.out} rows={rows}")
    elif args.cmd == "scale-million":
        scale_million(args.csv)
        print(args.csv)
    elif args.cmd == "normalize-rating":
        normalize_rating(args.csv)
        print(args.csv)
    elif args.cmd == "fill-template":
        fill_batch_template(args.template_csv, args.source_csv, args.out)
        print(args.out or args.template_csv)
    elif args.cmd == "all":
        command_all(args.pdf, args.page, args.out)
        print(args.out)
    elif args.cmd == "pipeline":
        csv_out, template_out = command_pipeline(
            args.pdf,
            args.template_csv,
            args.page,
            args.csv_out,
            args.out_template,
        )
        print(f"ocr_csv={csv_out}")
        print(f"batch_template={template_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
