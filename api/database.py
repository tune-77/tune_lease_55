"""
会話履歴テーブル (conversation_history) と感情履歴テーブル (emotion_history) の初期化・CRUD。
DB は data_cases.py と同じ lease_data.db を共有する。
"""
from __future__ import annotations

import datetime as dt
import math
import os
from typing import Optional

from api.db_connection import get_connection, placeholder, ensure_schema, _is_postgres


def _insert_get_id(conn, sql: str, params: tuple) -> int:
    """INSERT して新規レコードのIDを返す（SQLite/PostgreSQL 両対応）。"""
    cur = conn.cursor()
    if _is_postgres():
        cur.execute(sql + " RETURNING id", params)
        row = cur.fetchone()
        return row[0] if row else 0
    else:
        cur.execute(sql, params)
        return cur.lastrowid or 0


def init_conversation_history_table() -> None:
    """conversation_history テーブルとインデックスを冪等に作成する。"""
    ensure_schema()


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
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.executemany(
            f"INSERT INTO conversation_history (session_id, company_name, role, content) VALUES ({ph}, {ph}, {ph}, {ph})",
            rows,
        )


def get_conversation_history(company_name: str, limit: int = 5) -> list[dict]:
    """
    企業名で過去の会話履歴を取得する。
    直近 limit セッション分（session_id でグループ化）を返す。
    """
    init_conversation_history_table()
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        # 直近 N セッションの session_id を取得
        cur.execute(
            f"""
            SELECT DISTINCT session_id, MAX(created_at) AS latest
            FROM conversation_history
            WHERE company_name = {ph}
            GROUP BY session_id
            ORDER BY latest DESC
            LIMIT {ph}
            """,
            (company_name, limit),
        )
        sessions = cur.fetchall()

        if not sessions:
            return []

        session_ids = [r["session_id"] for r in sessions]
        in_placeholders = ",".join([ph] * len(session_ids))
        cur.execute(
            f"""
            SELECT id, session_id, company_name, role, content, created_at
            FROM conversation_history
            WHERE session_id IN ({in_placeholders})
            ORDER BY created_at ASC
            """,
            session_ids,
        )
        rows = cur.fetchall()

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
    ensure_schema()


def save_emotion_feedback(rating: str, comment: Optional[str], emotion_category: Optional[str]) -> int:
    """フィードバックを保存し、新規レコードの id を返す。"""
    init_emotion_feedback_table()
    ph = placeholder()
    with get_connection() as conn:
        return _insert_get_id(
            conn,
            f"INSERT INTO emotion_feedback (rating, comment, emotion_category) VALUES ({ph}, {ph}, {ph})",
            (rating, comment or None, emotion_category or None),
        )


def get_emotion_feedbacks(resolved: Optional[bool] = None) -> list[dict]:
    """フィードバック一覧を返す。resolved=False で未解決のみ。"""
    init_emotion_feedback_table()
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        if resolved is None:
            cur.execute(
                "SELECT * FROM emotion_feedback ORDER BY created_at DESC"
            )
        else:
            cur.execute(
                f"SELECT * FROM emotion_feedback WHERE resolved = {ph} ORDER BY created_at DESC",
                (1 if resolved else 0,),
            )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def delete_conversation_session(session_id: str) -> int:
    """session_id に紐づく全レコードを削除し、削除件数を返す。"""
    init_conversation_history_table()
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"DELETE FROM conversation_history WHERE session_id = {ph}", (session_id,)
        )
        return cur.rowcount


def get_past_arbiter_summaries(company_name: str, limit: int = 3) -> list[dict]:
    """
    同一企業の過去討論における軍師の最終判断を取得する（記憶注入用）。
    Returns list of {"session_id", "content", "created_at"}.
    """
    init_conversation_history_table()
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT session_id, content, created_at
            FROM conversation_history
            WHERE company_name = {ph} AND role = 'agent_gunshi'
            ORDER BY created_at DESC
            LIMIT {ph}
            """,
            (company_name, limit),
        )
        rows = cur.fetchall()
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
    ensure_schema()


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
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT id FROM emotion_history WHERE date(recorded_at) = {ph}",
            (today,),
        )
        existing = cur.fetchone()
        if existing:
            return int(existing[0]), False
        new_id = _insert_get_id(
            conn,
            f"""INSERT INTO emotion_history
                (recorded_at, hopeful_anxiety, careful_attachment, intellectual_excitement,
                 unrewarded_effort, quiet_loneliness, earned_confidence, protective_frustration,
                 dominant_raw_emotion, notes)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})""",
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
        return int(new_id), True


def get_emotion_history(days: int = 30) -> list[dict]:
    """過去N日分の感情スコアを時系列で返す。"""
    init_emotion_history_table()
    threshold = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id, recorded_at, hopeful_anxiety, careful_attachment,
                   intellectual_excitement, unrewarded_effort, quiet_loneliness,
                   earned_confidence, protective_frustration, dominant_raw_emotion, notes
            FROM emotion_history
            WHERE recorded_at >= {ph}
            ORDER BY recorded_at ASC
            """,
            (threshold,),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_emotion_summary(days: int = 30) -> dict:
    """期間内の各軸の平均・最大・最小・標準偏差を返す。"""
    init_emotion_history_table()
    threshold = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT hopeful_anxiety, careful_attachment, intellectual_excitement,
                   unrewarded_effort, quiet_loneliness, earned_confidence,
                   protective_frustration, dominant_raw_emotion
            FROM emotion_history
            WHERE recorded_at >= {ph}
            ORDER BY recorded_at ASC
            """,
            (threshold,),
        )
        rows = cur.fetchall()

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


# ── デモ向け 30日トレンド補完（Cloud Run 対策）──────────────────────────────────
#
# Cloud Run（CLOUDRUN_DATA_MODE=demo）はコンテナが揮発性で、cold start のたびに
# bundle の demo.db を再seedし、その後「当日分」のスナップショットしか記録しない。
# そのため emotion_history が実質1件になり、フロントの EmotionTrendChart が
# 「データが少なすぎます（1件）」で 30日トレンドを描画できない。
# 実データが無い（疎な）ときだけ、現在の mood を基点に過去日を決定論的に補完する。
# 実データの日付は絶対に上書きせず、notes="seed:demo-backfill" で識別可能にする。

_BACKFILL_NOTE = "seed:demo-backfill"


def _backfill_variation(date_iso: str, axis: str) -> float:
    """日付×軸から決定論的な微変動（約 ±7、ゆるやかなドリフト付き）を返す。

    プロセスやDBをまたいでも同じ日付・軸なら同じ値になるよう hashlib を使う。
    """
    import hashlib

    digest = hashlib.sha256(f"{date_iso}:{axis}".encode("utf-8")).digest()
    # 0..1 の擬似乱数
    jitter = (int.from_bytes(digest[:4], "big") / 0xFFFFFFFF) * 2.0 - 1.0  # -1..1
    # 日付起点のゆるやかな波（軸ごとに位相をずらす）
    day_ord = dt.date.fromisoformat(date_iso).toordinal()
    phase = (int.from_bytes(digest[4:6], "big") % 360) * math.pi / 180.0
    drift = math.sin(day_ord / 5.0 + phase) * 4.0
    return jitter * 7.0 + drift


def backfill_emotion_history(
    base_scores: dict[str, float],
    dominant: str,
    days: int = 30,
    min_rows: int = 2,
) -> int:
    """emotion_history が疎なとき、過去 days 日分の欠損日を補完する（デモ用）。

    - 既存レコードが min_rows 件以上あれば何もしない（実データを尊重）。
    - 既に行が存在する日付は絶対に上書きしない。
    - 当日分は record_emotion_snapshot 側が扱うため補完対象から除外する。
    Returns 補完した行数。
    """
    if not base_scores:
        return 0

    init_emotion_history_table()
    today = dt.date.today()
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        threshold = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
        ).isoformat()
        cur.execute(
            f"SELECT COUNT(*) FROM emotion_history WHERE recorded_at >= {ph}",
            (threshold,),
        )
        existing_count = int(cur.fetchone()[0])
        if existing_count >= min_rows:
            return 0

        cur.execute(
            f"SELECT date(recorded_at) FROM emotion_history WHERE recorded_at >= {ph}",
            (threshold,),
        )
        existing_dates = {row[0] for row in cur.fetchall()}

        inserted = 0
        # 古い日→新しい日の順で挿入し、当日は除外する。
        for offset in range(days, 0, -1):
            day = today - dt.timedelta(days=offset)
            day_iso = day.isoformat()
            if day_iso in existing_dates:
                continue
            recorded_at = dt.datetime(
                day.year, day.month, day.day, 12, 0, 0, tzinfo=dt.timezone.utc
            ).isoformat()
            values = [recorded_at]
            for axis in _EMOTION_AXES:
                base = float(base_scores.get(axis) or 0.0)
                val = base + _backfill_variation(day_iso, axis)
                values.append(round(max(0.0, min(100.0, val)), 1))
            values.append(dominant)
            values.append(_BACKFILL_NOTE)
            cur.execute(
                f"""INSERT INTO emotion_history
                    (recorded_at, hopeful_anxiety, careful_attachment,
                     intellectual_excitement, unrewarded_effort, quiet_loneliness,
                     earned_confidence, protective_frustration,
                     dominant_raw_emotion, notes)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})""",
                tuple(values),
            )
            inserted += 1
        return inserted
