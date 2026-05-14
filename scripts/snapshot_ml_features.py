"""
P0-003: ml_features 週次スナップショット
lease_data.db の ml_features テーブルを CSV にエクスポートし、
90日超過ファイルを自動削除する。
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _setup_logging() -> None:
    logging.basicConfig(
        format="[%(asctime)s JST] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )


def validate_db(db_path: Path) -> None:
    if not db_path.exists():
        print(f"db not found: {db_path}", file=sys.stderr)
        sys.exit(2)


def validate_table(conn: sqlite3.Connection, table: str) -> None:
    if not _TABLE_RE.match(table):
        print(f"invalid table name: {table}", file=sys.stderr)
        sys.exit(3)
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    if cur.fetchone() is None:
        print(f"table not found: {table}", file=sys.stderr)
        sys.exit(3)


def export_to_csv(
    conn: sqlite3.Connection,
    table: str,
    out_path: Path,
    chunk_size: int = 1000,
    dry_run: bool = False,
) -> int:
    prefix = "[dry-run] " if dry_run else ""
    cur = conn.execute(f"SELECT * FROM {table}")  # noqa: S608  # table validated
    columns = [d[0] for d in cur.description]
    total = 0
    if dry_run:
        logging.info("%swould write to %s", prefix, out_path)
        while True:
            rows = cur.fetchmany(chunk_size)
            if not rows:
                break
            total += len(rows)
        logging.info("%swould export %d rows", prefix, total)
        return total
    try:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            writer.writerow(columns)
            while True:
                rows = cur.fetchmany(chunk_size)
                if not rows:
                    break
                writer.writerows(rows)
                total += len(rows)
    except PermissionError:
        print(f"permission denied: {out_path}", file=sys.stderr)
        sys.exit(4)
    size_kb = out_path.stat().st_size / 1024
    logging.info("export done (%s rows, %.1f KB)", total, size_kb)
    return total


def enforce_retention(
    out_dir: Path,
    days: int,
    dry_run: bool = False,
) -> int:
    prefix = "[dry-run] " if dry_run else ""
    now = datetime.now(tz=JST).timestamp()
    cutoff = now - days * 86400
    deleted = 0
    for p in out_dir.glob("ml_features_*.csv"):
        try:
            if p.stat().st_mtime < cutoff:
                if dry_run:
                    logging.info("%swould delete %s", prefix, p.name)
                else:
                    p.unlink()
                    logging.info("deleted %s", p.name)
                deleted += 1
        except Exception as e:
            logging.warning("could not delete %s: %s", p.name, e)
    return deleted


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(description="Export ml_features to CSV snapshot")
    parser.add_argument("--db", default="lease_data.db", help="SQLite DB path")
    parser.add_argument("--table", default="ml_features", help="Table to export")
    parser.add_argument("--out-dir", default="data/snapshots", help="Output directory")
    parser.add_argument("--retention-days", type=int, default=90, help="Days to keep")
    parser.add_argument("--dry-run", action="store_true", help="Log only, no file I/O")
    args = parser.parse_args(argv)

    prefix = "[dry-run] " if args.dry_run else ""
    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    today = datetime.now(tz=JST).strftime("%Y%m%d")
    out_path = out_dir / f"ml_features_{today}.csv"

    logging.info("%ssnapshot start: db=%s table=%s", prefix, db_path, args.table)

    validate_db(db_path)

    if not out_dir.exists():
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            logging.info("created directory %s", out_dir)
        else:
            logging.info("%swould create directory %s", prefix, out_dir)

    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f"cannot open db: {e}", file=sys.stderr)
        return 1

    try:
        validate_table(conn, args.table)
        logging.info("%sexporting to %s", prefix, out_path)
        n = export_to_csv(conn, args.table, out_path, dry_run=args.dry_run)
        logging.info("%sexported %d rows", prefix, n)
    finally:
        conn.close()

    deleted = enforce_retention(out_dir, args.retention_days, dry_run=args.dry_run)
    if deleted:
        logging.info("%sdeleted %d old file(s)", prefix, deleted)

    logging.info("%ssnapshot complete", prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
