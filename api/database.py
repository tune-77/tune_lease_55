"""
会話履歴テーブル (conversation_history) と感情履歴テーブル (emotion_history) の初期化・CRUD。
DB は data_cases.py と同じ lease_data.db を共有する。
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sqlite3
from contextlib import closing
from typing import Optional
from runtime_paths import get_data_path, get_db_path

DB_PATH = get_db_path()


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


def init_emotion_feedback_table() -> None:
    """emotion_feedback テーブルを冪等に作成する。"""
    with closing(_open_db()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emotion_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                rating TEXT NOT NULL CHECK(rating IN ('good', 'needs_improvement')),
                comment TEXT,
                emotion_category TEXT,
                resolved BOOLEAN DEFAULT 0
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_emofb_resolved ON emotion_feedback(resolved)"
        )
        conn.commit()


def save_emotion_feedback(rating: str, comment: Optional[str], emotion_category: Optional[str]) -> int:
    """フィードバックを保存し、新規レコードの id を返す。"""
    init_emotion_feedback_table()
    with closing(_open_db()) as conn:
        cur = conn.execute(
            "INSERT INTO emotion_feedback (rating, comment, emotion_category) VALUES (?, ?, ?)",
            (rating, comment or None, emotion_category or None),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]


def get_emotion_feedbacks(resolved: Optional[bool] = None) -> list[dict]:
    """フィードバック一覧を返す。resolved=False で未解決のみ。"""
    init_emotion_feedback_table()
    with closing(_open_db()) as conn:
        conn.row_factory = sqlite3.Row
        if resolved is None:
            rows = conn.execute(
                "SELECT * FROM emotion_feedback ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM emotion_feedback WHERE resolved = ? ORDER BY created_at DESC",
                (1 if resolved else 0,),
            ).fetchall()
    return [dict(r) for r in rows]


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


# ── emotion_history テーブル（REV-075）────────────────────────────────────────

_EMOTION_AXES = [
    "hopeful_anxiety",
    "careful_attachment",
    "intellectual_excitement",
    "unrewarded_effort",
    "quiet_loneliness",
    "earned_confidence",
    "protective_frustration",
]


def init_emotion_history_table() -> None:
    """emotion_history テーブルとインデックスを冪等に作成する。"""
    with closing(_open_db()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emotion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                hopeful_anxiety REAL,
                careful_attachment REAL,
                intellectual_excitement REAL,
                unrewarded_effort REAL,
                quiet_loneliness REAL,
                earned_confidence REAL,
                protective_frustration REAL,
                dominant_raw_emotion TEXT,
                notes TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_emotion_history_date ON emotion_history(recorded_at)"
        )
        conn.commit()


def record_emotion_snapshot(
    scores: dict[str, float],
    dominant: str,
    notes: Optional[str] = None,
) -> tuple[int, bool]:
    """当日分をDBに保存する。既に当日レコードがあればスキップ。
    Returns (id, was_inserted).
    """
    init_emotion_history_table()
    today = dt.date.today().isoformat()
    with closing(_open_db()) as conn:
        existing = conn.execute(
            "SELECT id FROM emotion_history WHERE date(recorded_at) = ?",
            (today,),
        ).fetchone()
        if existing:
            return int(existing[0]), False
        cur = conn.execute(
            """
            INSERT INTO emotion_history
                (recorded_at, hopeful_anxiety, careful_attachment, intellectual_excitement,
                 unrewarded_effort, quiet_loneliness, earned_confidence, protective_frustration,
                 dominant_raw_emotion, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dt.datetime.now(dt.timezone.utc).isoformat(),
                scores.get("hopeful_anxiety"),
                scores.get("careful_attachment"),
                scores.get("intellectual_excitement"),
                scores.get("unrewarded_effort"),
                scores.get("quiet_loneliness"),
                scores.get("earned_confidence"),
                scores.get("protective_frustration"),
                dominant,
                notes,
            ),
        )
        conn.commit()
        return int(cur.lastrowid), True


def get_emotion_history(days: int = 30) -> list[dict]:
    """過去N日分の感情スコアを時系列で返す。"""
    init_emotion_history_table()
    with closing(_open_db()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, recorded_at, hopeful_anxiety, careful_attachment,
                   intellectual_excitement, unrewarded_effort, quiet_loneliness,
                   earned_confidence, protective_frustration, dominant_raw_emotion, notes
            FROM emotion_history
            WHERE recorded_at >= datetime('now', ? || ' days')
            ORDER BY recorded_at ASC
            """,
            (f"-{days}",),
        ).fetchall()
    return [dict(r) for r in rows]


def get_emotion_summary(days: int = 30) -> dict:
    """期間内の各軸の平均・最大・最小・標準偏差を返す。"""
    init_emotion_history_table()
    with closing(_open_db()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT hopeful_anxiety, careful_attachment, intellectual_excitement,
                   unrewarded_effort, quiet_loneliness, earned_confidence,
                   protective_frustration, dominant_raw_emotion
            FROM emotion_history
            WHERE recorded_at >= datetime('now', ? || ' days')
            ORDER BY recorded_at ASC
            """,
            (f"-{days}",),
        ).fetchall()

    if not rows:
        return {"days": days, "count": 0, "axes": {}, "dominant_avg": ""}

    axes_stats: dict[str, dict] = {}
    for axis in _EMOTION_AXES:
        values = [r[axis] for r in rows if r[axis] is not None]
        if not values:
            continue
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values) if len(values) > 1 else 0.0
        axes_stats[axis] = {
            "avg": round(avg, 1),
            "max": round(max(values), 1),
            "min": round(min(values), 1),
            "std": round(math.sqrt(variance), 1),
        }

    dominant_counts: dict[str, int] = {}
    for r in rows:
        d = r["dominant_raw_emotion"] or ""
        if d:
            dominant_counts[d] = dominant_counts.get(d, 0) + 1
    dominant_avg = max(dominant_counts, key=lambda k: dominant_counts[k]) if dominant_counts else ""

    return {
        "days": days,
        "count": len(rows),
        "axes": axes_stats,
        "dominant_avg": dominant_avg,
    }
