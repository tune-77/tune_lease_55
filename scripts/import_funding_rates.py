#!/usr/bin/env python3
"""
自社調達金利をCSVから funding_rates テーブルへ取り込む。

使い方:
  python scripts/import_funding_rates.py data/funding_rates_input.csv
  python scripts/import_funding_rates.py --manual 2026-05,5,1.30

CSVフォーマット（ヘッダー行必須）:
  year_month, term_years, rate_pct, note
"""

import argparse
import csv
import logging
import re
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "lease_data.db"
VALID_TERMS = {1, 3, 5, 7, 10}
RATE_MIN, RATE_MAX = -1.0, 15.0


def _validate(year_month: str, term_years: int, rate_pct: float) -> None:
    if not re.fullmatch(r"\d{4}-\d{2}", year_month):
        raise ValueError(f"year_month の形式が不正です（YYYY-MM 必須）: {year_month!r}")
    if term_years not in VALID_TERMS:
        raise ValueError(f"term_years は {sorted(VALID_TERMS)} のいずれかにしてください: {term_years}")
    if not (RATE_MIN <= rate_pct <= RATE_MAX):
        raise ValueError(f"rate_pct が範囲外 ({RATE_MIN}〜{RATE_MAX}%): {rate_pct}")


def import_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    inserted = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            try:
                ym = row["year_month"].strip()
                term = int(row["term_years"].strip())
                rate = float(row["rate_pct"].strip())
                note = row.get("note", "").strip() or None
                _validate(ym, term, rate)
            except (KeyError, ValueError) as e:
                log.warning("行 %d をスキップ: %s", i, e)
                continue

            conn.execute(
                """
                INSERT OR REPLACE INTO funding_rates
                  (year_month, term_years, rate_pct, source, note)
                VALUES (?, ?, ?, 'csv', ?)
                """,
                (ym, term, rate, note),
            )
            inserted += 1

    conn.commit()
    return inserted


def import_manual(conn: sqlite3.Connection, manual: str) -> None:
    parts = manual.strip().split(",")
    if len(parts) < 3:
        raise SystemExit("--manual の形式: YYYY-MM,TERM,RATE  例: 2026-05,5,1.30")
    ym, term_s, rate_s = parts[0].strip(), parts[1].strip(), parts[2].strip()
    term, rate = int(term_s), float(rate_s)
    _validate(ym, term, rate)
    conn.execute(
        """
        INSERT OR REPLACE INTO funding_rates
          (year_month, term_years, rate_pct, source)
        VALUES (?, ?, ?, 'manual')
        """,
        (ym, term, rate),
    )
    conn.commit()
    log.info("手動登録: %s  %dY  %.2f%%", ym, term, rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="自社調達金利を取り込む")
    parser.add_argument("csv_file", nargs="?", help="取り込むCSVファイルのパス")
    parser.add_argument("--manual", help="手動入力: YYYY-MM,TERM,RATE")
    parser.add_argument("--db", default=str(DB_PATH), help="DBファイルのパス")
    args = parser.parse_args()

    if not args.csv_file and not args.manual:
        parser.error("csv_file または --manual のどちらかを指定してください")

    conn = sqlite3.connect(args.db)
    try:
        if args.csv_file:
            p = Path(args.csv_file)
            if not p.exists():
                raise SystemExit(f"ファイルが見つかりません: {p}")
            n = import_csv(conn, p)
            log.info("CSV取込完了: %d 件登録しました", n)

        if args.manual:
            import_manual(conn, args.manual)

        total = conn.execute("SELECT COUNT(*) FROM funding_rates").fetchone()[0]
        log.info("funding_rates テーブル合計: %d 件", total)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
