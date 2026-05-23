"""
会話履歴テーブル (conversation_history) の初期化・CRUD。
DB は data_cases.py と同じ lease_data.db を共有する。
"""
from __future__ import annotations

import os
import pathlib
import sqlite3
from contextlib import closing
from typing import Optional

# __file__ 基準でポータブルに解決（絶対パスのハードコードを排除）
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


def init_conversation_history_table() -> None:
    """conversation_history テーブルとインデックスを冪等に作成する。"""
    with closing(_open_db()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                company_name TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_company ON conversation_history(company_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_history(session_id)"
        )
        conn.commit()


def save_conversation_messages(session_id: str, company_name: str, messages: list[dict]) -> None:
    """
    複数のメッセージを一括で保存する。
    messages は [{"role": str, "content": str}, ...] の形式。
    """
    if not messages:
        return
    init_conversation_history_table()
    rows = [
        (session_id, company_name or "", msg["role"], msg["content"])
        for msg in messages
        if msg.get("role") and msg.get("content")
    ]
    if not rows:
        return
    with closing(_open_db()) as conn:
        conn.executemany(
            "INSERT INTO conversation_history (session_id, company_name, role, content) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def get_conversation_history(company_name: str, limit: int = 5) -> list[dict]:
    """
    企業名で過去の会話履歴を取得する。
    直近 limit セッション分（session_id でグループ化）を返す。
    """
    init_conversation_history_table()
    with closing(_open_db()) as conn:
        conn.row_factory = sqlite3.Row
        # 直近 N セッションの session_id を取得
        sessions = conn.execute(
            """
            SELECT DISTINCT session_id, MAX(created_at) AS latest
            FROM conversation_history
            WHERE company_name = ?
            GROUP BY session_id
            ORDER BY latest DESC
            LIMIT ?
            """,
            (company_name, limit),
        ).fetchall()

        if not sessions:
            return []

        session_ids = [r["session_id"] for r in sessions]
        placeholders = ",".join("?" * len(session_ids))
        rows = conn.execute(
            f"""
            SELECT id, session_id, company_name, role, content, created_at
            FROM conversation_history
            WHERE session_id IN ({placeholders})
            ORDER BY created_at ASC
            """,
            session_ids,
        ).fetchall()

    result: dict[str, dict] = {}
    for row in rows:
        sid = row["session_id"]
        if sid not in result:
            result[sid] = {
                "session_id": sid,
                "company_name": row["company_name"],
                "created_at": row["created_at"],
                "messages": [],
            }
        result[sid]["messages"].append({
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        })

    # latest順に並べて返す
    ordered = sorted(result.values(), key=lambda x: x["created_at"], reverse=True)
    return ordered


def delete_conversation_session(session_id: str) -> int:
    """session_id に紐づく全レコードを削除し、削除件数を返す。"""
    init_conversation_history_table()
    with closing(_open_db()) as conn:
        cur = conn.execute(
            "DELETE FROM conversation_history WHERE session_id = ?", (session_id,)
        )
        conn.commit()
        return cur.rowcount


def get_past_arbiter_summaries(company_name: str, limit: int = 3) -> list[dict]:
    """
    同一企業の過去討論における軍師の最終判断を取得する（記憶注入用）。
    Returns list of {"session_id", "content", "created_at"}.
    """
    init_conversation_history_table()
    with closing(_open_db()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT session_id, content, created_at
            FROM conversation_history
            WHERE company_name = ? AND role = 'agent_gunshi'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (company_name, limit),
        ).fetchall()
    return [dict(r) for r in rows]
