"""Human judgment corrections used as reviewed model-improvement candidates."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

DEFAULT_DB_PATH = "data/lease_data.db"
VALID_DECISIONS = {"承認", "条件付", "否決"}
VALID_REVIEW_STATUSES = {"candidate", "approved", "rejected"}
DECISION_LABELS = {"否決": 0, "条件付": 1, "承認": 2}
PII_KEYS = {
    "name", "company_name", "address", "phone", "email", "representative",
    "hojin_name",
}

_DDL = """
CREATE TABLE IF NOT EXISTS judgment_feedback (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id            TEXT NOT NULL,
    recorded_at        TEXT NOT NULL,
    source             TEXT NOT NULL,
    model_decision     TEXT NOT NULL,
    human_decision     TEXT NOT NULL,
    changed            INTEGER NOT NULL,
    reason             TEXT NOT NULL,
    score              REAL,
    input_snapshot     TEXT,
    evidence_snapshot  TEXT,
    review_status      TEXT NOT NULL DEFAULT 'candidate',
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_judgment_feedback_case_id
    ON judgment_feedback(case_id);
CREATE INDEX IF NOT EXISTS idx_judgment_feedback_review
    ON judgment_feedback(review_status, changed);
"""


def normalize_decision(value: str) -> str:
    text = str(value or "").strip().replace("条件付き", "条件付")
    if "否決" in text or "否認" in text:
        return "否決"
    if "条件付" in text or "要審議" in text or "ボーダー" in text:
        return "条件付"
    if "承認" in text or "良決" in text:
        return "承認"
    return ""


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if str(key).lower() in PII_KEYS else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _json_or_none(value: dict[str, Any] | None) -> str | None:
    if not value:
        return None
    return json.dumps(_redact(value), ensure_ascii=False, sort_keys=True)


def record_judgment_feedback(
    *,
    case_id: str,
    model_decision: str,
    human_decision: str,
    reason: str,
    source: str,
    score: float | None = None,
    input_snapshot: dict[str, Any] | None = None,
    evidence_snapshot: dict[str, Any] | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    model = normalize_decision(model_decision)
    human = normalize_decision(human_decision)
    clean_reason = str(reason or "").strip()
    if not str(case_id or "").strip():
        return {"success": False, "record_id": -1, "error": "case_id is required"}
    if model not in VALID_DECISIONS or human not in VALID_DECISIONS:
        return {"success": False, "record_id": -1, "error": "invalid decision"}
    if model == human:
        return {"success": False, "record_id": -1, "error": "decision was not changed"}
    if len(clean_reason) < 5:
        return {"success": False, "record_id": -1, "error": "reason must be at least 5 characters"}

    try:
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        recorded_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(db_path) as conn:
            conn.executescript(_DDL)
            cur = conn.execute(
                """
                INSERT INTO judgment_feedback
                    (case_id, recorded_at, source, model_decision, human_decision,
                     changed, reason, score, input_snapshot, evidence_snapshot)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                """,
                (
                    str(case_id).strip(),
                    recorded_at,
                    str(source or "unknown").strip(),
                    model,
                    human,
                    clean_reason,
                    float(score) if score is not None else None,
                    _json_or_none(input_snapshot),
                    _json_or_none(evidence_snapshot),
                ),
            )
            record_id = int(cur.lastrowid)
        return {
            "success": True,
            "record_id": record_id,
            "review_status": "candidate",
            "model_label": DECISION_LABELS[model],
            "human_label": DECISION_LABELS[human],
            "error": None,
        }
    except Exception as exc:
        return {"success": False, "record_id": -1, "error": str(exc)}


def get_judgment_feedback_summary(db_path: str = DEFAULT_DB_PATH) -> dict[str, int]:
    try:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(_DDL)
            total, candidates, approved = conn.execute(
                """
                SELECT COUNT(*),
                       SUM(CASE WHEN review_status='candidate' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN review_status='approved' THEN 1 ELSE 0 END)
                FROM judgment_feedback
                """
            ).fetchone()
        return {
            "total": int(total or 0),
            "candidates": int(candidates or 0),
            "approved": int(approved or 0),
        }
    except Exception:
        return {"total": 0, "candidates": 0, "approved": 0}


def count_unprocessed_feedback(db_path: str = DEFAULT_DB_PATH) -> int:
    """review_status='candidate' のフィードバック件数を返す（PDCA実行判定用）。"""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(_DDL)
            (n,) = conn.execute(
                "SELECT COUNT(*) FROM judgment_feedback WHERE review_status='candidate'"
            ).fetchone()
        return int(n or 0)
    except Exception:
        return 0


def review_judgment_feedback(
    record_id: int,
    review_status: str,
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    status = str(review_status or "").strip()
    if status not in VALID_REVIEW_STATUSES:
        return {"success": False, "error": "invalid review_status"}
    try:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(_DDL)
            cur = conn.execute(
                "UPDATE judgment_feedback SET review_status=? WHERE id=?",
                (status, int(record_id)),
            )
        if cur.rowcount != 1:
            return {"success": False, "error": "feedback not found"}
        return {"success": True, "record_id": int(record_id), "review_status": status}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def load_judgment_training_candidates(
    *,
    approved_only: bool = True,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return structured rows for a future approval-judgment model.

    These rows must not be mixed into the delinquency/default model.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(_DDL)
            where = "WHERE review_status='approved'" if approved_only else "WHERE changed=1"
            rows = conn.execute(
                f"""
                SELECT id, case_id, recorded_at, source, model_decision,
                       human_decision, reason, score, input_snapshot,
                       evidence_snapshot, review_status
                FROM judgment_feedback
                {where}
                ORDER BY id
                """
            ).fetchall()
        result = []
        for row in rows:
            result.append({
                "id": row[0],
                "case_id": row[1],
                "recorded_at": row[2],
                "source": row[3],
                "model_decision": row[4],
                "human_decision": row[5],
                "target_label": DECISION_LABELS[row[5]],
                "reason": row[6],
                "score": row[7],
                "input_snapshot": json.loads(row[8]) if row[8] else {},
                "evidence_snapshot": json.loads(row[9]) if row[9] else {},
                "review_status": row[10],
            })
        return result
    except Exception:
        return []


def build_judgment_learning_prompt_block(
    *,
    limit: int = 8,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Build a compact prompt block from human-approved judgment corrections."""
    rows = load_judgment_training_candidates(
        approved_only=True,
        db_path=db_path,
    )
    if not rows:
        return ""

    lines = [
        "【レビュー済み実案件の判断差分】",
        "以下は人間が理由付きで承認した補助事例です。類似案件の確認観点として使い、",
        "個別案件を自動承認・自動否決する根拠にはしないでください。",
    ]
    for item in rows[-max(1, limit):]:
        score = (
            f"、スコア{float(item['score']):.1f}"
            if item.get("score") is not None
            else ""
        )
        lines.append(
            f"- {item['model_decision']}→{item['human_decision']}{score}: "
            f"{item['reason']}"
        )
    return "\n".join(lines)
