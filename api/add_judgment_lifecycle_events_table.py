"""
judgment_lifecycle_events テーブルの追加マイグレーション。

案件ごとの判断イベントを時系列で追跡するための器だけを作る。
既存の結果登録・RAG・スコアリング・判断資産昇格には接続しない。

Usage:
    python api/add_judgment_lifecycle_events_table.py
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing


DB_PATH = os.getenv("LEASE_DB_PATH", "data/lease_data.db")


def _open_db(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_judgment_lifecycle_events_table(path: str = DB_PATH) -> None:
    """judgment_lifecycle_events テーブルとインデックスを冪等に作成する。"""

    with closing(_open_db(path)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS judgment_lifecycle_events (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id             TEXT    NOT NULL,
                event_at            TEXT    NOT NULL DEFAULT (datetime('now')),
                event_type          TEXT    NOT NULL,
                event_stage         TEXT    NOT NULL DEFAULT '',
                actor               TEXT    NOT NULL DEFAULT '',
                source              TEXT    NOT NULL DEFAULT '',
                deal_outcome        TEXT    DEFAULT '',
                review_outcome      TEXT    DEFAULT '',
                post_outcome        TEXT    DEFAULT '',
                judgment_asset_id   TEXT    DEFAULT '',
                decision_before     TEXT    DEFAULT '',
                decision_after      TEXT    DEFAULT '',
                human_signal        TEXT    DEFAULT '',
                ai_signal           TEXT    DEFAULT '',
                note                TEXT    DEFAULT '',
                payload             TEXT    DEFAULT '',
                created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_jle_case
                ON judgment_lifecycle_events(case_id, event_at);
            CREATE INDEX IF NOT EXISTS idx_jle_type
                ON judgment_lifecycle_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_jle_stage
                ON judgment_lifecycle_events(event_stage);
            CREATE INDEX IF NOT EXISTS idx_jle_asset
                ON judgment_lifecycle_events(judgment_asset_id);
            """
        )
        conn.commit()
    print(f"[judgment-lifecycle] judgment_lifecycle_events table ready: {path}")


if __name__ == "__main__":
    init_judgment_lifecycle_events_table()
