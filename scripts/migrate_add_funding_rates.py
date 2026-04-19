#!/usr/bin/env python3
"""
data/lease_data.db に funding_rates テーブルを追加するマイグレーション。
CREATE TABLE IF NOT EXISTS を使うため冪等（何度実行してもデータが壊れない）。
"""

import logging
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "lease_logic_sumaho12" / "data" / "lease_data.db"

DDL = """
CREATE TABLE IF NOT EXISTS funding_rates (
    year_month  TEXT    NOT NULL,
    term_years  INTEGER NOT NULL,
    rate_pct    REAL    NOT NULL,
    source      TEXT    DEFAULT 'manual',
    note        TEXT,
    created_at  TEXT    DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (year_month, term_years)
);
CREATE INDEX IF NOT EXISTS idx_funding_rates_ym ON funding_rates(year_month);
"""


def migrate(conn: sqlite3.Connection) -> None:
    for stmt in DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
    log.info("funding_rates テーブルを確認/作成しました: %s", DB_PATH)


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        migrate(conn)
        row = conn.execute("SELECT COUNT(*) FROM funding_rates").fetchone()
        log.info("現在の登録件数: %d 件", row[0])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
