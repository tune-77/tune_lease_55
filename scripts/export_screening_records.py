#!/usr/bin/env python3
"""screening_records テーブルの内容を CSV に書き出す CLI スクリプト。

使い方:
    python scripts/export_screening_records.py
    python scripts/export_screening_records.py --db data/lease_data.db --out out.csv
    python scripts/export_screening_records.py --outcome contracted
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path


def export(db_path: str, out_path: str, outcome: str | None = None) -> int:
    """DB から screening_records をエクスポートし、行数を返す。"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    where = ""
    params: list = []
    if outcome:
        where = "WHERE outcome = ?"
        params = [outcome]

    rows = conn.execute(
        f"SELECT * FROM screening_records {where} ORDER BY id ASC", params
    ).fetchall()
    conn.close()

    if not rows:
        print("レコードが見つかりませんでした。")
        return 0

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))

    print(f"{len(rows)} 件を {out_path} に出力しました。")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="screening_records を CSV へ出力する")
    parser.add_argument(
        "--db",
        default="data/lease_data.db",
        help="SQLite DBファイルパス (default: data/lease_data.db)",
    )
    parser.add_argument(
        "--out",
        default="screening_records_export.csv",
        help="出力 CSV ファイルパス (default: screening_records_export.csv)",
    )
    parser.add_argument(
        "--outcome",
        choices=["contracted", "lost", "delinquent", "completed"],
        default=None,
        help="outcome でフィルタリング (default: 全件)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"エラー: DB ファイルが見つかりません: {db_path}", file=sys.stderr)
        sys.exit(1)

    export(str(db_path), args.out, args.outcome)


if __name__ == "__main__":
    main()
