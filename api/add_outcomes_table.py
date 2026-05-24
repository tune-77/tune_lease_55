"""
screening_outcomes テーブルの追加マイグレーション。

既存の lease_data.db に対して IF NOT EXISTS で安全に実行できる。
既存テーブル・カラムには一切変更を加えない。

スキーマは migrate_outcomes.py の DDL_SCREENING_OUTCOMES に準拠。

使い方:
    python api/add_outcomes_table.py
"""
from __future__ import annotations

import os
import pathlib
import sqlite3
from contextlib import closing

_DATA_DIR = os.environ.get(
    "LEASE_DATA_DIR",
    str(pathlib.Path(__file__).parent.parent / "data"),
)
DB_PATH = os.path.join(_DATA_DIR, "lease_data.db")


def _open_db(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_screening_outcomes_table(path: str = DB_PATH) -> None:
    """screening_outcomes テーブルとインデックスを冪等に作成する。

    スキーマは migrate_outcomes.py の DDL_SCREENING_OUTCOMES と同一。
    """
    with closing(_open_db(path)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS screening_outcomes (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id             TEXT    NOT NULL,
                screening_id        INTEGER,
                contract_date       TEXT,
                scheduled_end_date  TEXT,
                actual_status       TEXT    NOT NULL DEFAULT 'unknown',
                delinquent          INTEGER NOT NULL DEFAULT 0,
                loss_given_default  REAL,
                checked_at          TEXT    NOT NULL DEFAULT (datetime('now')),
                notes               TEXT,
                created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at          TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_so_case_id
                ON screening_outcomes(case_id);
            CREATE INDEX IF NOT EXISTS idx_so_status
                ON screening_outcomes(actual_status);
            CREATE INDEX IF NOT EXISTS idx_so_screening
                ON screening_outcomes(screening_id);
        """)
        conn.commit()
    print(f"[outcomes] screening_outcomes table ready: {path}")


# ── CRUD ──────────────────────────────────────────────────────────────────────

def insert_outcome(
    case_id: str,
    actual_status: str = "unknown",
    screening_id: int | None = None,
    contract_date: str | None = None,
    scheduled_end_date: str | None = None,
    delinquent: int = 0,
    loss_given_default: float | None = None,
    notes: str | None = None,
    path: str = DB_PATH,
) -> int:
    """レコードを追加して新規 id を返す。"""
    init_screening_outcomes_table(path)
    with closing(_open_db(path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO screening_outcomes
                (case_id, actual_status, screening_id, contract_date,
                 scheduled_end_date, delinquent, loss_given_default, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (case_id, actual_status, screening_id, contract_date,
             scheduled_end_date, delinquent, loss_given_default, notes),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]


def list_outcomes(
    screening_id: int | None = None,
    case_id: str | None = None,
    actual_status: str | None = None,
    limit: int = 100,
    path: str = DB_PATH,
) -> list[dict]:
    """一覧取得。screening_id / case_id / actual_status で絞り込み可能。"""
    init_screening_outcomes_table(path)
    clauses: list[str] = []
    params: list = []
    if screening_id is not None:
        clauses.append("screening_id = ?")
        params.append(screening_id)
    if case_id:
        clauses.append("case_id = ?")
        params.append(case_id)
    if actual_status:
        clauses.append("actual_status = ?")
        params.append(actual_status)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with closing(_open_db(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT id, case_id, screening_id, contract_date, scheduled_end_date,
                   actual_status, delinquent, loss_given_default, checked_at,
                   notes, created_at, updated_at
            FROM screening_outcomes
            {where}
            ORDER BY checked_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_outcome(outcome_id: int, path: str = DB_PATH) -> dict | None:
    """id 指定で1件取得。存在しなければ None。"""
    init_screening_outcomes_table(path)
    with closing(_open_db(path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM screening_outcomes WHERE id = ?", (outcome_id,)
        ).fetchone()
    return dict(row) if row else None


if __name__ == "__main__":
    init_screening_outcomes_table()
