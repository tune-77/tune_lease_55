"""
screening_outcomes テーブルの追加マイグレーション。

既存の lease_data.db に対して IF NOT EXISTS で安全に実行できる。
既存テーブル・カラムには一切変更を加えない。

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
    """screening_outcomes テーブルとインデックスを冪等に作成する。"""
    with closing(_open_db(path)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS screening_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                screening_id INTEGER NOT NULL,
                company_name TEXT,
                outcome TEXT,
                delinquent INTEGER DEFAULT 0,
                months_since_contract INTEGER,
                notes TEXT,
                recorded_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_outcomes_screening_id"
            " ON screening_outcomes(screening_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_outcomes_company"
            " ON screening_outcomes(company_name)"
        )
        conn.commit()
    print(f"[outcomes] screening_outcomes table ready: {path}")


# ── CRUD ──────────────────────────────────────────────────────────────────────

def insert_outcome(
    screening_id: int,
    company_name: str | None,
    outcome: str | None,
    delinquent: int = 0,
    months_since_contract: int | None = None,
    notes: str | None = None,
    path: str = DB_PATH,
) -> int:
    """レコードを追加して新規 id を返す。"""
    init_screening_outcomes_table(path)
    with closing(_open_db(path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO screening_outcomes
                (screening_id, company_name, outcome, delinquent, months_since_contract, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (screening_id, company_name, outcome, delinquent, months_since_contract, notes),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]


def list_outcomes(
    screening_id: int | None = None,
    company_name: str | None = None,
    limit: int = 100,
    path: str = DB_PATH,
) -> list[dict]:
    """一覧取得。screening_id / company_name で絞り込み可能。"""
    init_screening_outcomes_table(path)
    clauses: list[str] = []
    params: list = []
    if screening_id is not None:
        clauses.append("screening_id = ?")
        params.append(screening_id)
    if company_name:
        clauses.append("company_name = ?")
        params.append(company_name)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with closing(_open_db(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT id, screening_id, company_name, outcome, delinquent,
                   months_since_contract, notes, recorded_at
            FROM screening_outcomes
            {where}
            ORDER BY recorded_at DESC
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
