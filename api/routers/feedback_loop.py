from __future__ import annotations

import contextlib
import json
import os
import re
import datetime as dt
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from api.db_connection import current_backend, get_connection, placeholder
from api.cloudrun_writeback import record_cloudrun_input_event
from judgment_asset_bandit import (
    append_feedback_event,
    build_bandit_signals,
    read_feedback_rows,
    signal_for_candidate,
)

try:
    from filelock import FileLock, Timeout as _FileLockTimeout
    _FILELOCK_AVAILABLE = True
except ImportError:
    _FILELOCK_AVAILABLE = False
    _FileLockTimeout = Exception  # type: ignore[assignment,misc]

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
_REPO_ROOT = str(Path(_SCRIPT_DIR).parent.parent)

_HUMAN_RESPONSE_FEEDBACK_LOG = Path(_REPO_ROOT) / "data" / "human_response_feedback.jsonl"
_SCREENING_LOOP_FEEDBACK_LOG = Path(_REPO_ROOT) / "data" / "screening_loop_feedback.jsonl"
_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATES_JSONL = Path(_REPO_ROOT) / "data" / "autoresearch_judgment_asset_candidates.jsonl"
_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON = Path(_REPO_ROOT) / "data" / "autoresearch_judgment_asset_candidate_state.json"
_NEWS_JUDGMENT_SIGNALS_JSONL = Path(_REPO_ROOT) / "data" / "news_judgment_signals.jsonl"
_CANONICAL_JUDGMENT_RULES_JSON = Path(_REPO_ROOT) / "data" / "canonical_judgment_rules.json"
_JUDGMENT_ASSET_USAGE_FEEDBACK_LOG = Path(_REPO_ROOT) / "data" / "judgment_asset_usage_feedback.jsonl"
_JUDGMENT_ASSET_FEEDBACK_DROPS_LOG = Path(_REPO_ROOT) / "data" / "judgment_asset_feedback_drops.jsonl"
_LANGUAGE_JUDGMENT_MATERIALS_JSONL = Path(_REPO_ROOT) / "data" / "language_judgment_materials.jsonl"
_RESPONSE_IMPACT_PREDICTIONS_JSONL = Path(_REPO_ROOT) / "data" / "response_impact_predictions.jsonl"
_SCREENING_INPUT_ASSIST_EVENTS_JSONL = Path(_REPO_ROOT) / "data" / "screening_input_assist_events.jsonl"
_JUDGMENT_ASSET_PROMOTION_LOCK_PATH = Path(_REPO_ROOT) / "data" / ".judgment_asset_promotion.lock"
_HUMAN_RESPONSE_POSITIVE_RATINGS = {"shion_like", "good"}
_HUMAN_RESPONSE_NEGATIVE_RATINGS = {"thin", "generic", "not_shion", "bad"}
_CLOUDRUN_RETURN_DB = Path(_REPO_ROOT) / "data" / "cloudrun_experience_return.db"
_CLOUDRUN_RETURN_REVIEW_TABLES = {
    "score_input": "cloudrun_score_inputs",
    "ocr_result": "cloudrun_ocr_results",
    "shion_review": "shion_screening_reviews",
    "judgment_asset": "cloudrun_judgment_asset_candidates",
}
_CLOUDRUN_RETURN_STATUSES = {"candidate", "approved", "held", "rejected"}

router = APIRouter(tags=["feedback-loop"])

_SELF_PROPOSAL_LAYER_LABELS = {
    "usage_based": "利用ログ由来",
    "feedback_based": "人間反応・判断ログ由来",
    "system_audit_based": "システム監査由来",
}


def _with_self_proposal_review_meta(
    proposals: list[dict[str, Any]],
    *,
    evidence_layer: str,
) -> list[dict[str, Any]]:
    """Add lightweight review metadata for older persisted self proposals."""
    enriched: list[dict[str, Any]] = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        item = dict(proposal)
        item.setdefault("evidence_layer", evidence_layer)
        item.setdefault("evidence_layer_label", _SELF_PROPOSAL_LAYER_LABELS.get(evidence_layer, evidence_layer))
        item.setdefault("review_policy", "human_decision_required")
        item.setdefault("effect_tracking", "採用/保留/却下と、success_metric の事後変化で確認する")
        if not isinstance(item.get("genetic_profile"), dict) or not item.get("genetic_profile"):
            item["genetic_profile"] = {
                "variant_type": "self_proposal",
                "selection_pressure": item.get("evidence_layer_label") or evidence_layer,
                "validation_status": item.get("human_decision_status") or item.get("status") or "needs_human_review",
                "success_metric": item.get("success_metric") or "",
                "risk": item.get("risk") or "",
                "proposal_style": item.get("proposal_style") or "standard",
            }
        if not item.get("review_status_label"):
            item["review_status_label"] = "レビュー候補" if item.get("success_metric") else "観測継続"
        enriched.append(item)
    return enriched


class HumanResponseFeedbackRequest(BaseModel):
    message: str = ""
    response: str = ""
    rating: Literal["shion_like", "good", "thin", "generic", "not_shion", "bad"]
    comment: str = ""
    route: str = ""
    continuity_hook: str = ""
    user_id: str = "default"


class ScreeningLoopFeedbackRequest(BaseModel):
    surface: str = "screening"
    target: Literal["issue", "ringi_policy"]
    rating: str
    issue_text: str = ""
    ringi_policy_text: str = ""
    comment: str = ""
    score: Optional[float] = None
    hantei: str = ""
    context: dict = Field(default_factory=dict)
    user_id: str = "default"


class ShionScreeningReviewSaveRequest(BaseModel):
    case_id: str = ""
    company_name: str = ""
    industry_major: str = ""
    industry_sub: str = ""
    sales_dept: str = ""
    score: Optional[float] = None
    hantei: str = ""
    q_risk: Optional[float] = None
    umap_anomaly_score: Optional[float] = None
    memory_refs: int = 0
    knowledge_refs: int = 0
    identity_used: bool = False
    review_text: str
    prompt_text: str = ""
    form_snapshot: dict = Field(default_factory=dict)
    result_snapshot: dict = Field(default_factory=dict)
    user_feedback: str = ""


class ShionScreeningReviewFeedbackRequest(BaseModel):
    user_feedback: Literal[
        "useful",
        "needs_fix",
        "wrong",
        "specific",
        "thin",
        "discomfort_hit",
        "over_inferred",
    ]


class JudgmentAssetCandidateFeedbackRequest(BaseModel):
    feedback: Literal["useful", "neutral", "rejected"]
    case_id: str = ""
    review_id: Optional[int] = None
    comment: str = ""
    edited_claim: str = ""


class JudgmentAssetCandidateManualRequest(BaseModel):
    claim: str
    candidate_type: Literal["confirmation_question", "condition_signal", "application_rule", "caution"] = "confirmation_question"
    research_topic: str = "manual"
    case_id: str = ""
    review_id: Optional[int] = None


class JudgmentAssetPromotionReviewRequest(BaseModel):
    action: Literal["hold", "reject"] = "hold"
    comment: str = ""


class ScreeningExperienceCaseRequest(BaseModel):
    demo_case_id: str = ""
    source_case_id: str = ""
    company_name: str
    period: str = ""
    industry_major: str = ""
    industry_sub: str = ""
    sales_dept: str = ""
    score: Optional[float] = None
    decision: str = ""
    outcome: str = ""
    similarity: str = ""
    action_taken: str = ""
    lesson: str = ""
    difference: str = ""
    source: str = "manual"
    form_snapshot: dict = Field(default_factory=dict)
    result_snapshot: dict = Field(default_factory=dict)


class ScreeningInputAssistEventRequest(BaseModel):
    action: Literal[
        "input_started",
        "search_click",
        "search_result",
        "search_failed",
        "candidate_selected",
        "copied",
        "score_submitted",
    ]
    surface: str = "screening"
    session_id: str = ""
    company_no: str = ""
    company_name: str = ""
    industry_major: str = ""
    industry_sub: str = ""
    asset_name: str = ""
    source_case_id: str = ""
    source_company_name: str = ""
    candidate_count: int = 0
    diff_count: int = 0
    confirm_count: int = 0
    copied_field_count: int = 0
    changed_after_copy_count: int = 0
    elapsed_ms: Optional[int] = None
    copied_fields: list[str] = Field(default_factory=list)
    confirm_fields: list[str] = Field(default_factory=list)
    note: str = ""
    metadata: dict = Field(default_factory=dict)


class CloudRunReturnReviewRequest(BaseModel):
    review_status: Literal["approved", "held", "rejected"]
    note: str = ""


class CloudRunReturnPromotionRequest(BaseModel):
    kind: Literal["judgment_asset"] = "judgment_asset"


def _append_human_response_feedback(req: HumanResponseFeedbackRequest) -> dict:
    import datetime as _dt
    import hashlib as _hashlib
    import json as _json

    message = str(req.message or "").strip()
    response = str(req.response or "").strip()
    route = str(req.route or _relationship_signal_route(message)).strip()
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "id": _hashlib.sha256(f"{message}\n{response}\n{req.rating}\n{req.comment}".encode("utf-8")).hexdigest()[:16],
        "user_id": req.user_id,
        "rating": req.rating,
        "route": route,
        "message_preview": message[:240],
        "response_start": response[:500],
        "continuity_hook": str(req.continuity_hook or "").strip()[:300],
        "comment": str(req.comment or "").strip()[:500],
    }
    _HUMAN_RESPONSE_FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _HUMAN_RESPONSE_FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _append_screening_loop_feedback(req: ScreeningLoopFeedbackRequest) -> dict:
    import datetime as _dt
    import hashlib as _hashlib
    import json as _json

    target = str(req.target or "").strip()
    rating = str(req.rating or "").strip()
    issue_text = str(req.issue_text or "").strip()[:500]
    ringi_policy_text = str(req.ringi_policy_text or "").strip()[:500]
    comment = str(req.comment or "").strip()[:500]
    safe_context = {
        "customer_type": str((req.context or {}).get("customer_type") or "")[:40],
        "main_bank": str((req.context or {}).get("main_bank") or "")[:40],
        "competitor": str((req.context or {}).get("competitor") or "")[:40],
        "has_lease_history": bool((req.context or {}).get("has_lease_history")),
        "has_asset_context": bool((req.context or {}).get("has_asset_context")),
    }
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "id": _hashlib.sha256(
            f"{target}\n{rating}\n{issue_text}\n{ringi_policy_text}\n{comment}".encode("utf-8")
        ).hexdigest()[:16],
        "surface": str(req.surface or "screening").strip()[:80],
        "target": target,
        "rating": rating,
        "issue_text": issue_text,
        "ringi_policy_text": ringi_policy_text,
        "comment": comment,
        "score": req.score,
        "hantei": str(req.hantei or "").strip()[:80],
        "context": safe_context,
        "user_id": str(req.user_id or "default").strip()[:80],
        "loop": "screening_judgment_ux",
    }
    _SCREENING_LOOP_FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _SCREENING_LOOP_FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _append_screening_input_assist_event(req: ScreeningInputAssistEventRequest) -> dict:
    import datetime as _dt
    import hashlib as _hashlib
    import json as _json

    def clean_text(value: Any, limit: int = 120) -> str:
        return str(value or "").strip()[:limit]

    copied_fields = [clean_text(item, 80) for item in (req.copied_fields or []) if clean_text(item, 80)][:80]
    confirm_fields = [clean_text(item, 80) for item in (req.confirm_fields or []) if clean_text(item, 80)][:80]
    metadata = req.metadata if isinstance(req.metadata, dict) else {}
    safe_metadata = {
        clean_text(key, 80): clean_text(value, 240)
        for key, value in list(metadata.items())[:20]
    }
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    base = "\n".join([
        now,
        req.action,
        req.session_id,
        req.source_case_id,
        clean_text(req.company_no, 40),
        clean_text(req.company_name, 120),
    ])
    entry = {
        "ts": now,
        "id": _hashlib.sha256(base.encode("utf-8")).hexdigest()[:16],
        "action": req.action,
        "surface": clean_text(req.surface or "screening", 80),
        "session_id": clean_text(req.session_id, 120),
        "company_no": clean_text(req.company_no, 40),
        "company_name": clean_text(req.company_name, 120),
        "industry_major": clean_text(req.industry_major, 80),
        "industry_sub": clean_text(req.industry_sub, 80),
        "asset_name": clean_text(req.asset_name, 120),
        "source_case_id": clean_text(req.source_case_id, 80),
        "source_company_name": clean_text(req.source_company_name, 120),
        "candidate_count": max(0, int(req.candidate_count or 0)),
        "diff_count": max(0, int(req.diff_count or 0)),
        "confirm_count": max(0, int(req.confirm_count or 0)),
        "copied_field_count": max(0, int(req.copied_field_count or 0)),
        "changed_after_copy_count": max(0, int(req.changed_after_copy_count or 0)),
        "elapsed_ms": req.elapsed_ms if req.elapsed_ms is not None and req.elapsed_ms >= 0 else None,
        "copied_fields": copied_fields,
        "confirm_fields": confirm_fields,
        "note": clean_text(req.note, 300),
        "metadata": safe_metadata,
    }
    _SCREENING_INPUT_ASSIST_EVENTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with _SCREENING_INPUT_ASSIST_EVENTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _read_screening_input_assist_events(limit: int = 500) -> list[dict[str, Any]]:
    if not _SCREENING_INPUT_ASSIST_EVENTS_JSONL.exists():
        return []
    rows: list[dict[str, Any]] = []
    with _SCREENING_INPUT_ASSIST_EVENTS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-max(1, min(int(limit or 500), 5000)):]


def _summarize_screening_input_assist_events(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_action: dict[str, int] = {}
    sessions: set[str] = set()
    copied_sessions: set[str] = set()
    submitted_after_copy_sessions: set[str] = set()
    elapsed_after_copy: list[int] = []
    copied_field_total = 0
    confirm_field_total = 0
    changed_after_copy_values: list[int] = []

    copied_at_by_session: dict[str, str] = {}
    for row in rows:
        action = str(row.get("action") or "")
        by_action[action] = by_action.get(action, 0) + 1
        session_id = str(row.get("session_id") or "")
        if session_id:
            sessions.add(session_id)
        if action == "copied":
            if session_id:
                copied_sessions.add(session_id)
                copied_at_by_session[session_id] = str(row.get("ts") or "")
            copied_field_total += int(row.get("copied_field_count") or 0)
            confirm_field_total += int(row.get("confirm_count") or 0)
        if action == "score_submitted":
            if session_id and session_id in copied_at_by_session:
                submitted_after_copy_sessions.add(session_id)
                changed_after_copy_values.append(int(row.get("changed_after_copy_count") or 0))
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                elapsed_raw = metadata.get("elapsed_from_copy_ms") if isinstance(metadata, dict) else None
                try:
                    elapsed = int(elapsed_raw)
                except (TypeError, ValueError):
                    elapsed = row.get("elapsed_ms")
                if isinstance(elapsed, int) and elapsed >= 0:
                    elapsed_after_copy.append(elapsed)

    copy_count = by_action.get("copied", 0)
    copied_session_count = len(copied_sessions)
    search_count = by_action.get("search_click", 0)
    score_submit_count = by_action.get("score_submitted", 0)
    return {
        "event_count": len(rows),
        "session_count": len(sessions),
        "by_action": by_action,
        "search_count": search_count,
        "copy_count": copy_count,
        "copy_rate": round(copy_count / search_count, 3) if search_count else None,
        "score_submit_count": score_submit_count,
        "submitted_after_copy_count": len(submitted_after_copy_sessions),
        "submitted_after_copy_rate": round(len(submitted_after_copy_sessions) / copied_session_count, 3) if copied_session_count else None,
        "avg_copied_fields": round(copied_field_total / copy_count, 1) if copy_count else None,
        "avg_confirm_fields": round(confirm_field_total / copy_count, 1) if copy_count else None,
        "avg_changed_after_copy": round(sum(changed_after_copy_values) / len(changed_after_copy_values), 1) if changed_after_copy_values else None,
        "avg_elapsed_after_copy_ms": round(sum(elapsed_after_copy) / len(elapsed_after_copy)) if elapsed_after_copy else None,
    }


def _ensure_shion_screening_reviews_table() -> None:
    from api.db_connection import ensure_schema

    ensure_schema()


def _save_shion_screening_review(req: ShionScreeningReviewSaveRequest) -> dict:
    import json as _json

    _ensure_shion_screening_reviews_table()
    ph = placeholder()
    review_text = str(req.review_text or "").strip()
    if not review_text:
        raise HTTPException(status_code=422, detail="review_text is required")

    form_snapshot = _json.dumps(req.form_snapshot or {}, ensure_ascii=False)
    result_snapshot = _json.dumps(req.result_snapshot or {}, ensure_ascii=False)
    values = (
        str(req.case_id or "")[:120],
        str(req.company_name or "")[:160],
        str(req.industry_major or "")[:160],
        str(req.industry_sub or "")[:160],
        str(req.sales_dept or "")[:120],
        req.score,
        str(req.hantei or "")[:120],
        req.q_risk,
        req.umap_anomaly_score,
        int(req.memory_refs or 0),
        int(req.knowledge_refs or 0),
        bool(req.identity_used),
        review_text[:8000],
        str(req.prompt_text or "")[:8000],
        form_snapshot[:20000],
        result_snapshot[:20000],
        str(req.user_feedback or "")[:80],
    )
    columns = (
        "case_id, company_name, industry_major, industry_sub, sales_dept, score, hantei, "
        "q_risk, umap_anomaly_score, memory_refs, knowledge_refs, identity_used, review_text, "
        "prompt_text, form_snapshot, result_snapshot, user_feedback"
    )
    placeholders = ", ".join([ph] * len(values))
    with get_connection() as conn:
        cur = conn.cursor()
        if current_backend() == "postgresql":
            cur.execute(
                f"INSERT INTO shion_screening_reviews ({columns}) VALUES ({placeholders}) RETURNING id, created_at",
                values,
            )
            row = cur.fetchone()
            new_id = int(row[0])
            created_at = str(row[1])
        else:
            cur.execute(
                f"INSERT INTO shion_screening_reviews ({columns}) VALUES ({placeholders})",
                values,
            )
            new_id = int(cur.lastrowid)
            created_at = ""
    return {
        "id": new_id,
        "case_id": values[0],
        "company_name": values[1],
        "score": values[5],
        "hantei": values[6],
        "created_at": created_at,
    }


def _list_shion_screening_reviews(
    industry_sub: str = "",
    company_name: str = "",
    case_id: str = "",
    limit: int = 5,
) -> list[dict]:
    _ensure_shion_screening_reviews_table()
    ph = placeholder()
    where: list[str] = []
    params: list[Any] = []
    if case_id.strip():
        where.append(f"case_id = {ph}")
        params.append(case_id.strip())
    if industry_sub.strip():
        where.append(f"industry_sub = {ph}")
        params.append(industry_sub.strip())
    if company_name.strip():
        where.append(f"company_name LIKE {ph}")
        params.append(f"%{company_name.strip()}%")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    capped_limit = max(1, min(int(limit or 5), 20))
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id, case_id, company_name, industry_major, industry_sub, sales_dept,
                   score, hantei, q_risk, umap_anomaly_score, review_text,
                   memory_refs, knowledge_refs, identity_used, user_feedback, created_at
              FROM shion_screening_reviews
              {where_sql}
             ORDER BY created_at DESC, id DESC
             LIMIT {capped_limit}
            """,
            tuple(params),
        )
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def _update_shion_screening_review_feedback(review_id: int, user_feedback: str) -> dict:
    _ensure_shion_screening_reviews_table()
    ph = placeholder()
    normalized = str(user_feedback or "").strip()
    allowed = {"useful", "needs_fix", "wrong", "specific", "thin", "discomfort_hit", "over_inferred"}
    if normalized not in allowed:
        raise HTTPException(status_code=422, detail=f"user_feedback must be one of: {', '.join(sorted(allowed))}")
    with get_connection() as conn:
        cur = conn.cursor()
        if current_backend() == "postgresql":
            cur.execute(
                f"""
                UPDATE shion_screening_reviews
                   SET user_feedback = {ph}
                 WHERE id = {ph}
                 RETURNING id, user_feedback
                """,
                (normalized, review_id),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="review not found")
            return {"id": int(row[0]), "user_feedback": str(row[1])}
        cur.execute(
            f"UPDATE shion_screening_reviews SET user_feedback = {ph} WHERE id = {ph}",
            (normalized, review_id),
        )
        if cur.rowcount <= 0:
            raise HTTPException(status_code=404, detail="review not found")
    return {"id": review_id, "user_feedback": normalized}


def _log_judgment_asset_feedback_drop(
    *,
    review_id: int,
    user_feedback: str,
    reason: str,
    case_id: str = "",
) -> None:
    """判断資産フィードバックが記録されず握りつぶされた事実を可視化用に残す.

    data/judgment_asset_feedback_drops.jsonl に追記するだけ。
    判断資産の集計・昇格・スコアリングには一切使わない（原因調査専用）。

    Cloud Run のディスクはスケールゼロ・再起動で失われるため、ローカル追記に加えて
    既存の cloudrun_input_writeback 経路（GCS）にも同じ理由を送る。日次同期
    （scripts/sync_cloudrun_inputs_from_gcs.py）がこのイベントを読み、正本の
    data/judgment_asset_feedback_drops.jsonl へ反映する。
    """
    import json as _json
    from datetime import datetime as _dt

    entry = {
        "schema_version": "1",
        "review_id": review_id,
        "user_feedback": str(user_feedback or ""),
        "reason": reason,
        "case_id": case_id,
        "dropped_at": _dt.now().isoformat(timespec="seconds"),
    }
    try:
        _JUDGMENT_ASSET_FEEDBACK_DROPS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _JUDGMENT_ASSET_FEEDBACK_DROPS_LOG.open("a", encoding="utf-8") as _f:
            _f.write(_json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass

    try:
        record_cloudrun_input_event(
            event_type="judgment_asset_feedback_drop",
            surface="screening",
            payload=entry,
        )
    except Exception:
        pass


def _record_judgment_asset_feedback_from_review(review_id: int, user_feedback: str) -> None:
    """紫苑レビューへのユーザーフィードバックを判断資産使用ログに記録する（バックグラウンドタスク用）.

    data/judgment_asset_usage_feedback.jsonl に追記するだけ。
    プロンプト・スコアリング・DB スキーマには一切触れない。
    記録されずに握りつぶされた場合は data/judgment_asset_feedback_drops.jsonl に理由を残す。
    """
    import json as _json
    from datetime import datetime as _dt

    _FEEDBACK_JSONL = _JUDGMENT_ASSET_USAGE_FEEDBACK_LOG
    _OUTCOME_MAP = {
        "useful": "helped",
        "specific": "helped",
        "discomfort_hit": "helped",
        "needs_fix": "challenged",
        "thin": "challenged",
        "over_inferred": "challenged",
        "wrong": "rejected",
    }
    outcome = _OUTCOME_MAP.get(str(user_feedback or "").strip())
    if not outcome:
        _log_judgment_asset_feedback_drop(
            review_id=review_id, user_feedback=user_feedback, reason="unknown_outcome",
        )
        return

    try:
        _ensure_shion_screening_reviews_table()
        ph = placeholder()
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT case_id, result_snapshot FROM shion_screening_reviews WHERE id = {ph}",
                (review_id,),
            )
            row = cur.fetchone()
            if not row:
                _log_judgment_asset_feedback_drop(
                    review_id=review_id, user_feedback=user_feedback, reason="review_not_found",
                )
                return
            case_id = str(row[0] or "")
            result_snapshot_str = str(row[1] or "{}")
    except Exception:
        _log_judgment_asset_feedback_drop(
            review_id=review_id, user_feedback=user_feedback, reason="review_lookup_error",
        )
        return

    try:
        snapshot = _json.loads(result_snapshot_str)
        refs: list[str] = []
        if isinstance(snapshot.get("knowledge_refs"), list):
            refs = [str(r) for r in snapshot["knowledge_refs"] if r]
        if not refs and isinstance(snapshot.get("memory_debug"), dict):
            md = snapshot["memory_debug"]
            if isinstance(md.get("knowledge_refs"), list):
                refs = [str(r) for r in md["knowledge_refs"] if r]
    except Exception:
        refs = []

    # 実ルール参照が無いフィードバックは判断資産に紐付けようがないため、
    # judgment_asset_growth の unknown_rule 減点対象になる偽rule_idは書き込まない。
    if not refs:
        _log_judgment_asset_feedback_drop(
            review_id=review_id, user_feedback=user_feedback, reason="no_matching_refs", case_id=case_id,
        )
        return

    now = _dt.now().isoformat(timespec="seconds")
    _FEEDBACK_JSONL.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _FEEDBACK_JSONL.open("a", encoding="utf-8") as _f:
            for rule_id in refs[:12]:
                entry = {
                    "schema_version": "1",
                    "rule_id": rule_id.strip(),
                    "outcome": outcome,
                    "case_id": case_id,
                    "note": "",
                    "source": "shion_screening_review",
                    "used_at": now,
                }
                _f.write(_json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        _log_judgment_asset_feedback_drop(
            review_id=review_id, user_feedback=user_feedback, reason="usage_log_write_error", case_id=case_id,
        )


def _screening_candidate_terms(*values: str) -> set[str]:
    text = " ".join(str(value or "") for value in values)
    terms = set(re.findall(r"[A-Za-z0-9_]{3,}|[一-龥ぁ-んァ-ンー]{2,}", text))
    weak = {"リース", "審査", "確認", "判断", "対象", "企業", "物件", "場合", "する", "ます"}
    expanded = {term.lower() for term in terms if term not in weak}
    if any(key in text for key in ("トラック", "車両", "運送", "物流", "配送")):
        expanded.update({"車両", "中古", "残価", "流動性", "稼働", "燃料", "人件費"})
    if any(key in text for key in ("工作機械", "機械", "製造", "工場", "加工")):
        expanded.update({"機械", "設備", "保守", "稼働", "更新", "残価", "流動性"})
    if any(key in text for key in ("厨房", "飲食", "店舗", "新店舗")):
        expanded.update({"設備", "保守", "稼働", "残価", "撤退", "中古"})
    return expanded



def _load_autoresearch_judgment_asset_candidates(limit: int = 500) -> list[dict[str, Any]]:
    state: dict[str, Any] = {}
    if _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON.exists():
        try:
            raw_state = json.loads(_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(raw_state, dict):
                state = raw_state
        except (json.JSONDecodeError, OSError):
            state = {}
    rows: list[dict[str, Any]] = []
    if _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATES_JSONL.exists():
        try:
            for line in _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATES_JSONL.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    candidate_state = state.get(str(item.get("id") or ""))
                    if isinstance(candidate_state, dict):
                        item = {**item, **candidate_state}
                    rows.append(item)
                    if len(rows) >= limit:
                        break
        except OSError:
            rows = []
    rows = [row for row in rows if str(row.get("id") or "") != "demo-renewal-asset-candidate"]
    return rows[:limit]



def _load_news_judgment_signals(limit: int = 100) -> list[dict[str, Any]]:
    state: dict[str, Any] = {}
    if _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON.exists():
        try:
            raw_state = json.loads(_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(raw_state, dict):
                state = raw_state
        except (json.JSONDecodeError, OSError):
            state = {}
    rows: list[dict[str, Any]] = []
    today = dt.date.today()
    if _NEWS_JUDGMENT_SIGNALS_JSONL.exists():
        try:
            for line in _NEWS_JUDGMENT_SIGNALS_JSONL.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                if str(item.get("actionability") or "") != "actionable":
                    continue
                valid_until = str(item.get("valid_until") or "")
                if valid_until:
                    try:
                        if dt.date.fromisoformat(valid_until[:10]) < today:
                            continue
                    except ValueError:
                        pass
                signal_state = state.get(str(item.get("id") or ""))
                if isinstance(signal_state, dict):
                    item = {**item, **signal_state}
                rows.append(item)
        except OSError:
            rows = []
    rows.sort(
        key=lambda item: (
            str(item.get("research_date") or ""),
            float(item.get("confidence") or 0.0) - float(item.get("noise_score") or 0.0),
        ),
        reverse=True,
    )
    return rows[:limit]



def _load_canonical_judgment_asset_candidates(limit: int = 100) -> list[dict[str, Any]]:
    import hashlib as _hashlib

    if not _CANONICAL_JUDGMENT_RULES_JSON.exists():
        return []
    state: dict[str, Any] = {}
    if _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON.exists():
        try:
            raw_state = json.loads(_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(raw_state, dict):
                state = raw_state
        except (json.JSONDecodeError, OSError):
            state = {}
    try:
        payload = json.loads(_CANONICAL_JUDGMENT_RULES_JSON.read_text(encoding="utf-8", errors="ignore"))
    except (json.JSONDecodeError, OSError):
        return []
    rules = payload.get("rules") if isinstance(payload, dict) else []
    rows: list[dict[str, Any]] = []
    type_map = {
        "judgment_rule": "application_rule",
        "risk_signal": "condition_signal",
        "user_preference": "confirmation_question",
    }
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        if rule.get("private") is True or str(rule.get("status") or "") != "active":
            continue
        domains = set(str(domain) for domain in (rule.get("domains") or []) if domain)
        domain = str(rule.get("domain") or "")
        if domain:
            domains.add(domain)
        if domains and "lease_screening" not in domains:
            continue
        statement = str(rule.get("canonical_statement") or "").strip()
        if len(statement) < 12:
            continue
        rule_id = str(rule.get("id") or "").strip()
        candidate_id = f"cr-{rule_id or _hashlib.sha256(statement.encode('utf-8')).hexdigest()[:16]}"
        candidate_state = state.get(candidate_id)
        material_type = str(rule.get("material_type") or "judgment_rule")
        item = {
            "id": candidate_id,
            "source": "canonical_judgment_rules",
            "candidate_type": type_map.get(material_type, "application_rule"),
            "research_topic": str(rule.get("concept") or "canonical_judgment_rule"),
            "research_title": "Promoted Judgment Asset",
            "claim": statement,
            "effective_claim": statement,
            "edited_claim": "",
            "edit_count": 0,
            "evidence_path": f"data/canonical_judgment_rules.json#{str(rule.get('concept') or rule_id)}",
            "promotion_status": "active",
            "verified_status": "canonical",
            "use_count": 0,
            "useful_count": 0,
            "rejected_count": 0,
            "confidence": float(rule.get("confidence") or 0.0),
            "evidence_count": int(rule.get("evidence_count") or 0),
            "user_evidence_count": int(rule.get("user_evidence_count") or 0),
            "risk_axis": list(rule.get("risk_axis") or []),
            "sample_claims": list(rule.get("sample_claims") or [])[:6],
            "domains": sorted(domains),
        }
        if isinstance(candidate_state, dict):
            item = {**item, **candidate_state}
            item["effective_claim"] = str(item.get("edited_claim") or statement)
        rows.append(item)
        if len(rows) >= limit:
            break
    return rows


def _rank_screening_judgment_asset_candidate(item: dict[str, Any], context_terms: set[str]) -> tuple[float, int, int, str]:
    claim = str(item.get("claim") or "")
    haystack = " ".join([
        claim,
        str(item.get("effective_claim") or ""),
        str(item.get("research_topic") or ""),
        str(item.get("research_title") or ""),
        str(item.get("source_section") or ""),
        " ".join(str(axis) for axis in (item.get("risk_axis") or [])),
        " ".join(str(sample) for sample in (item.get("sample_claims") or [])[:3]),
    ])
    item_terms = _screening_candidate_terms(haystack)
    overlap = len(context_terms & item_terms)
    candidate_type = str(item.get("candidate_type") or "")
    type_bonus = {
        "confirmation_question": 4.0,
        "condition_signal": 3.0,
        "caution": 2.0,
        "application_rule": 1.0,
    }.get(candidate_type, 0.0)
    usage_bonus = min(3.0, float(item.get("useful_count") or 0) * 1.5 + float(item.get("use_count") or 0) * 0.2)
    edit_bonus = min(3.0, float(item.get("edit_count") or 0) * 1.5)
    canonical_bonus = 0.0
    if str(item.get("source") or "") == "canonical_judgment_rules" or str(item.get("promotion_status") or "") in {"active", "promoted"}:
        canonical_bonus = 4.0 + min(2.0, float(item.get("user_evidence_count") or 0) * 0.4) + min(1.0, float(item.get("confidence") or 0.0))
    penalty = float(item.get("rejected_count") or 0) * 2.0
    generic_penalty = 2.5 if any(term in claim for term in ("Gemini", "自動否決", "自動承認", "一次情報か", "検討材料として使")) else 0.0
    update_context_bonus = 0.0
    if context_terms & {"更新", "更改", "増額"} and any(term in haystack for term in ("更新設備", "更改", "既存設備")):
        update_context_bonus = 6.0
    new_customer_penalty = 0.0
    if "新規" not in context_terms and any(term in haystack for term in ("新規先", "demo-new-customer-competition")):
        new_customer_penalty = 5.0
    score = (
        overlap * 3.0
        + type_bonus
        + usage_bonus
        + edit_bonus
        + canonical_bonus
        + update_context_bonus
        - penalty
        - generic_penalty
        - new_customer_penalty
    )
    return (score, overlap, -len(claim), claim)


def _is_generic_autoresearch_candidate(item: dict[str, Any]) -> bool:
    if str(item.get("source") or "") == "canonical_judgment_rules":
        return False
    claim = str(item.get("claim") or "")
    if str(item.get("asset_quality") or "actionable") == "textbook_general":
        return True
    generic_markers = (
        "この情報は対象業種",
        "参照元が一次情報",
        "補助情報だけでなく",
        "Geminiの要約",
        "自動否決・自動承認",
        "検討材料として使",
        "一次情報または専門機関",
        "民間記事だけを根拠",
        "財務内容を確認",
        "返済原資を確認",
        "業界動向を確認",
        "担保価値を確認",
        "資金使途を確認",
        "総合的に判断",
        "慎重に判断",
    )
    return any(marker in claim for marker in generic_markers)


def _select_screening_judgment_asset_candidates(
    *,
    industry_major: str = "",
    industry_sub: str = "",
    asset_name: str = "",
    asset_purpose: str = "",
    hantei: str = "",
    score: Optional[float] = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    context_terms = _screening_candidate_terms(industry_major, industry_sub, asset_name, asset_purpose, hantei)
    rows = (
        _load_canonical_judgment_asset_candidates()
        + _load_autoresearch_judgment_asset_candidates()
        + _load_news_judgment_signals()
    )
    if not rows:
        return []
    bandit_signals = build_bandit_signals(read_feedback_rows())
    ranked: list[tuple[tuple[float, int, int, str], dict[str, Any]]] = []
    for item in rows:
        if str(item.get("promotion_status") or "") == "rejected_or_deprioritized":
            continue
        if _is_generic_autoresearch_candidate(item):
            continue
        rank = _rank_screening_judgment_asset_candidate(item, context_terms)
        signal = signal_for_candidate(item, bandit_signals)
        rank = (rank[0] + signal.rank_bonus, rank[1], rank[2], rank[3])
        if context_terms and rank[1] <= 0 and int(item.get("useful_count") or 0) <= 0 and int(item.get("edit_count") or 0) <= 0:
            continue
        if rank[0] <= 0 and context_terms:
            continue
        ranked.append((rank, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)

    def _is_promoted_judgment_asset(item: dict[str, Any]) -> bool:
        return (
            str(item.get("source") or "") == "canonical_judgment_rules"
            or str(item.get("promotion_status") or "") in {"active", "promoted"}
            or str(item.get("verified_status") or "") == "canonical"
        )

    def _to_screening_judgment_asset_response(item: dict[str, Any]) -> dict[str, Any]:
        claim = str(item.get("claim") or "")
        return {
            "id": str(item.get("id") or ""),
            "candidate_type": str(item.get("candidate_type") or ""),
            "research_topic": str(item.get("research_topic") or ""),
            "claim": claim,
            "effective_claim": str(item.get("edited_claim") or item.get("effective_claim") or claim),
            "edited_claim": str(item.get("edited_claim") or ""),
            "edit_count": int(item.get("edit_count") or 0),
            "evidence_path": str(item.get("evidence_path") or ""),
            "promotion_status": str(item.get("promotion_status") or "not_promoted"),
            "source": str(item.get("source") or "autoresearch_judgment_asset_candidates"),
            "valid_until": str(item.get("valid_until") or ""),
            "use_when": str(item.get("use_when") or ""),
            "do_not_use_when": str(item.get("do_not_use_when") or ""),
            "actionability": str(item.get("actionability") or ""),
            "use_count": int(item.get("use_count") or 0),
            "useful_count": int(item.get("useful_count") or 0),
            "rejected_count": int(item.get("rejected_count") or 0),
            "verified_status": str(item.get("verified_status") or "unverified"),
            "learning": signal_for_candidate(item, bandit_signals).as_dict(),
        }

    def _normalized_screening_claim(item: dict[str, Any]) -> str:
        return " ".join(str(item.get("edited_claim") or item.get("effective_claim") or item.get("claim") or "").split())

    selected: list[dict[str, Any]] = []
    seen_types: set[str] = set()
    for _, item in ranked:
        candidate_type = str(item.get("candidate_type") or "")
        if len(selected) < max(1, limit) and (candidate_type not in seen_types or len(selected) >= 2):
            seen_types.add(candidate_type)
            selected.append(_to_screening_judgment_asset_response(item))
        if len(selected) >= max(1, min(int(limit or 3), 5)):
            break

    capped_limit = max(1, min(int(limit or 3), 5))
    selected_promoted_claims = {
        _normalized_screening_claim(item)
        for item in selected
        if _is_promoted_judgment_asset(item) and _normalized_screening_claim(item)
    }
    has_unpromoted_nonduplicate = any(
        not _is_promoted_judgment_asset(item)
        and _normalized_screening_claim(item)
        and _normalized_screening_claim(item) not in selected_promoted_claims
        for item in selected
    )
    if capped_limit >= 3 and selected and (
        all(_is_promoted_judgment_asset(item) for item in selected) or not has_unpromoted_nonduplicate
    ):
        for _, item in ranked:
            if _is_promoted_judgment_asset(item):
                continue
            candidate_id = str(item.get("id") or "")
            if candidate_id and any(existing.get("id") == candidate_id for existing in selected):
                continue
            replacement = _to_screening_judgment_asset_response(item)
            replacement_claim = _normalized_screening_claim(replacement)
            if replacement_claim and replacement_claim in selected_promoted_claims:
                continue
            if len(selected) >= capped_limit:
                replace_index = next(
                    (
                        idx
                        for idx, existing in enumerate(selected)
                        if not _is_promoted_judgment_asset(existing)
                        and _normalized_screening_claim(existing) in selected_promoted_claims
                    ),
                    len(selected) - 1,
                )
                selected[replace_index] = replacement
            else:
                selected.append(replacement)
            break
    return selected


def _update_autoresearch_judgment_asset_candidate_feedback(
    candidate_id: str,
    req: JudgmentAssetCandidateFeedbackRequest,
) -> dict[str, Any]:
    import datetime as _dt

    candidate_id = str(candidate_id or "").strip()
    if not candidate_id:
        raise HTTPException(status_code=422, detail="candidate_id is required")
    candidates = _load_canonical_judgment_asset_candidates() + _load_autoresearch_judgment_asset_candidates() + _load_news_judgment_signals()
    if candidates and not any(str(item.get("id") or "") == candidate_id for item in candidates):
        raise HTTPException(status_code=404, detail="candidate not found")
    try:
        from scripts.build_autoresearch_judgment_asset_candidates import load_state, write_state
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"candidate state tools unavailable: {exc}")

    state = load_state(_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON)
    current = dict(state.get(candidate_id) or {})
    for key, default in {
        "use_count": 0,
        "useful_count": 0,
        "rejected_count": 0,
        "neutral_count": 0,
        "last_used_at": "",
        "last_feedback_at": "",
        "verified_status": "unverified",
        "verification_note": "",
        "edited_claim": "",
        "edit_count": 0,
        "last_edited_at": "",
    }.items():
        current.setdefault(key, default)
    current["use_count"] = int(current.get("use_count") or 0) + 1
    if req.feedback == "useful":
        current["useful_count"] = int(current.get("useful_count") or 0) + 1
    elif req.feedback == "rejected":
        current["rejected_count"] = int(current.get("rejected_count") or 0) + 1
    else:
        current["neutral_count"] = int(current.get("neutral_count") or 0) + 1
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    current["last_used_at"] = now
    current["last_feedback_at"] = now
    note_bits = [
        f"feedback={req.feedback}",
        f"case_id={str(req.case_id or '')[:80]}",
    ]
    if req.review_id:
        note_bits.append(f"review_id={req.review_id}")
    if req.comment:
        note_bits.append(f"comment={str(req.comment)[:160]}")
    edited_claim = str(req.edited_claim or "").strip()
    if edited_claim:
        original_claim = ""
        for item in candidates:
            if str(item.get("id") or "") == candidate_id:
                original_claim = str(item.get("claim") or "")
                break
        if edited_claim != original_claim and edited_claim != str(current.get("edited_claim") or ""):
            current["edited_claim"] = edited_claim[:500]
            current["edit_count"] = int(current.get("edit_count") or 0) + 1
            current["last_edited_at"] = now
            note_bits.append("edited_claim=updated")
    current["verification_note"] = " / ".join(note_bits)
    state[candidate_id] = current
    write_state(
        _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON,
        [{"id": candidate_id, **current}],
        state,
    )
    feedback_outcome = {
        "useful": "helped",
        "neutral": "neutral",
        "rejected": "rejected",
    }.get(req.feedback, "neutral")
    try:
        append_feedback_event(
            asset_id=candidate_id,
            outcome=feedback_outcome,
            case_id=req.case_id,
            note=req.comment or current.get("verification_note") or "",
            source="judgment_asset_candidate_feedback",
        )
    except Exception:
        pass
    return {"id": candidate_id, **current}



def _create_manual_judgment_asset_candidate(req: JudgmentAssetCandidateManualRequest) -> dict[str, Any]:
    import datetime as _dt
    import hashlib as _hashlib

    claim = str(req.claim or "").strip()
    if len(claim) < 8:
        raise HTTPException(status_code=422, detail="claim must be at least 8 characters")
    now = _dt.datetime.now(_dt.timezone.utc)
    now_iso = now.isoformat()
    topic = str(req.research_topic or "manual").strip() or "manual"
    candidate_type = str(req.candidate_type or "confirmation_question")
    seed = "|".join(["manual", now_iso, candidate_type, topic, claim, str(req.case_id or ""), str(req.review_id or "")])
    candidate_id = _hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    row = {
        "id": candidate_id,
        "candidate_type": candidate_type,
        "research_topic": topic,
        "research_title": "Manual Judgment Asset",
        "research_date": now.date().isoformat(),
        "claim": claim,
        "effective_claim": claim,
        "edited_claim": claim,
        "edit_count": 1,
        "last_edited_at": now_iso,
        "source_section": "manual_input",
        "evidence_path": f"manual://screening/{str(req.case_id or '')[:80]}",
        "review_status": "candidate",
        "asset_quality": "actionable",
        "quality_reasons": [],
        "promotion_status": "not_promoted",
        "use_count": 0,
        "useful_count": 0,
        "rejected_count": 0,
        "neutral_count": 0,
        "last_used_at": "",
        "last_feedback_at": "",
        "verified_status": "unverified",
        "verification_note": "manual_candidate_created",
        "requires_human_use_feedback": True,
        "requires_result_verification": True,
        "manual_created_at": now_iso,
        "manual_case_id": str(req.case_id or "")[:120],
        "manual_review_id": int(req.review_id or 0) if req.review_id else 0,
        "use_policy": "人間が追加した判断資産候補。案件レビューで使い、効いた/外したと結果検証で育てる。",
    }
    _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATES_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        from scripts.build_autoresearch_judgment_asset_candidates import load_state, write_state
        state = load_state(_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON)
        state[candidate_id] = {
            "use_count": 0,
            "useful_count": 0,
            "rejected_count": 0,
            "neutral_count": 0,
            "last_used_at": "",
            "last_feedback_at": "",
            "verified_status": "unverified",
            "verification_note": "manual_candidate_created",
            "edited_claim": claim,
            "edit_count": 1,
            "last_edited_at": now_iso,
        }
        write_state(_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON, [{"id": candidate_id, **state[candidate_id]}], state)
    except Exception:
        pass
    return row


def _read_judgment_asset_candidate_state() -> dict[str, Any]:
    if not _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON.exists():
        return {}
    try:
        payload = json.loads(_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON.read_text(encoding="utf-8", errors="ignore"))
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_judgment_asset_candidate_state(state: dict[str, Any]) -> None:
    _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_judgment_asset_promotion_candidates(limit: int = 30) -> list[dict[str, Any]]:
    state = _read_judgment_asset_candidate_state()
    rows = _load_autoresearch_judgment_asset_candidates(limit=1000)
    active_statements = {
        " ".join(str(item.get("claim") or item.get("canonical_statement") or "").split())
        for item in _load_canonical_judgment_asset_candidates(limit=1000)
    }
    candidates: list[dict[str, Any]] = []
    for item in rows:
        candidate_id = str(item.get("id") or "")
        if not candidate_id:
            continue
        merged = {**item, **dict(state.get(candidate_id) or {})}
        promotion_status = str(merged.get("promotion_status") or item.get("promotion_status") or "not_promoted")
        if promotion_status in {"active", "promoted", "rejected_or_deprioritized", "rejected"}:
            continue
        use_count = int(merged.get("use_count") or 0)
        useful_count = int(merged.get("useful_count") or 0)
        rejected_count = int(merged.get("rejected_count") or 0)
        edit_count = int(merged.get("edit_count") or 0)
        is_manual = str(merged.get("source_section") or "") == "manual_input" or str(merged.get("research_topic") or "").startswith("manual")
        claim = str(merged.get("edited_claim") or merged.get("effective_claim") or merged.get("claim") or "").strip()
        if len(claim) < 12:
            continue
        already_active_statement = " ".join(claim.split()) in active_statements
        if already_active_statement and useful_count <= 0 and edit_count <= 0:
            continue
        score = useful_count * 4 + edit_count * 3 + use_count + (2 if is_manual else 0) - rejected_count * 5
        if score <= 0 and not is_manual:
            continue
        candidates.append({
            "id": candidate_id,
            "candidate_type": str(merged.get("candidate_type") or "application_rule"),
            "research_topic": str(merged.get("research_topic") or ""),
            "claim": str(item.get("claim") or ""),
            "effective_claim": claim,
            "edited_claim": str(merged.get("edited_claim") or ""),
            "evidence_path": str(merged.get("evidence_path") or ""),
            "promotion_status": promotion_status,
            "verified_status": str(merged.get("verified_status") or "unverified"),
            "use_count": use_count,
            "useful_count": useful_count,
            "neutral_count": int(merged.get("neutral_count") or 0),
            "rejected_count": rejected_count,
            "edit_count": edit_count,
            "last_used_at": str(merged.get("last_used_at") or ""),
            "last_feedback_at": str(merged.get("last_feedback_at") or ""),
            "verification_note": (
                "既存の正規判断資産と同じ文面。昇格時は新規追加ではなく、既存正規への証拠追加として扱う。"
                if already_active_statement
                else str(merged.get("verification_note") or "")
            ),
            "score": score,
            "already_active_statement": already_active_statement,
            "source": str(merged.get("source") or "autoresearch_judgment_asset_candidates"),
        })
    candidates.sort(
        key=lambda item: (
            0 if item.get("already_active_statement") else 1,
            1 if int(item.get("edit_count") or 0) > 0 else 0,
            int(item.get("score") or 0),
            int(item.get("useful_count") or 0),
            int(item.get("edit_count") or 0),
        ),
        reverse=True,
    )
    return candidates[:max(1, min(int(limit or 30), 100))]


@contextlib.contextmanager
def _judgment_asset_promotion_lock():
    """canonical_judgment_rules.json / candidate_state.json への同時書き込みを直列化する。

    filelock 未導入環境（BR-426と同じフォールバック方針）ではロック無しで進む。
    """
    if not _FILELOCK_AVAILABLE:
        yield
        return
    _JUDGMENT_ASSET_PROMOTION_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(_JUDGMENT_ASSET_PROMOTION_LOCK_PATH), timeout=10)
    try:
        lock.acquire()
    except _FileLockTimeout:
        raise HTTPException(status_code=409, detail="judgment asset store is busy, try again")
    try:
        yield
    finally:
        lock.release()


def _promote_judgment_asset_candidate_to_canonical(candidate_id: str) -> dict[str, Any]:
    import datetime as _dt
    import hashlib as _hashlib

    with _judgment_asset_promotion_lock():
        candidates = _load_judgment_asset_promotion_candidates(limit=1000)
        candidate = next((item for item in candidates if str(item.get("id") or "") == candidate_id), None)
        if not candidate:
            raise HTTPException(status_code=404, detail="promotion candidate not found")
        statement = str(candidate.get("edited_claim") or candidate.get("effective_claim") or candidate.get("claim") or "").strip()
        if len(statement) < 12:
            raise HTTPException(status_code=422, detail="candidate statement is too short")
        now = _dt.datetime.now().isoformat(timespec="seconds")
        material_type = {
            "confirmation_question": "judgment_rule",
            "condition_signal": "risk_signal",
            "application_rule": "judgment_rule",
            "caution": "risk_signal",
        }.get(str(candidate.get("candidate_type") or ""), "judgment_rule")
        concept_raw = str(candidate.get("research_topic") or "manual_screening").strip() or "manual_screening"
        concept = re.sub(r"[^A-Za-z0-9_]+", "_", concept_raw).strip("_")[:60] or "manual_screening"
        payload: dict[str, Any] = {}
        try:
            payload = json.loads(_CANONICAL_JUDGMENT_RULES_JSON.read_text(encoding="utf-8", errors="ignore"))
        except (json.JSONDecodeError, OSError):
            payload = {}
        rules = payload.get("rules") if isinstance(payload, dict) else []
        if not isinstance(rules, list):
            rules = []
        normalized_statement = " ".join(statement.split())
        existing = next((rule for rule in rules if " ".join(str(rule.get("canonical_statement") or "").split()) == normalized_statement), None)
        if existing:
            existing["updated_at"] = now
            existing["evidence_count"] = max(int(existing.get("evidence_count") or 0), int(candidate.get("use_count") or 0) + 1)
            existing["user_evidence_count"] = max(int(existing.get("user_evidence_count") or 0), int(candidate.get("useful_count") or 0) + int(candidate.get("edit_count") or 0))
            evidence_paths = list(existing.get("evidence_paths") or [])
            evidence_path = str(candidate.get("evidence_path") or f"judgment_asset_candidate_state:{candidate_id}")
            if evidence_path and evidence_path not in evidence_paths:
                evidence_paths.append(evidence_path)
            existing["evidence_paths"] = evidence_paths[:12]
            promoted_rule = existing
            promoted_status = "updated"
        else:
            promoted_rule = {
                "id": _hashlib.sha256(f"manual_promotion|{material_type}|{concept}|{statement}".encode("utf-8")).hexdigest()[:16],
                "status": "active",
                "source_status": "manual_promoted",
                "material_type": material_type,
                "domain": "lease_screening",
                "concept": concept,
                "canonical_statement": statement,
                "evidence_count": max(1, int(candidate.get("use_count") or 0)),
                "user_evidence_count": max(1, int(candidate.get("useful_count") or 0) + int(candidate.get("edit_count") or 0)),
                "confidence": 0.78 + min(0.16, float(candidate.get("useful_count") or 0) * 0.04 + float(candidate.get("edit_count") or 0) * 0.03),
                "risk_axis": [],
                "sample_claims": [str(candidate.get("claim") or statement)[:220]],
                "evidence_paths": [str(candidate.get("evidence_path") or f"judgment_asset_candidate_state:{candidate_id}")],
                "created_at": now,
                "updated_at": now,
                "promotion_source": "judgment_asset_review_gate",
                "private": False,
                "material_types": [material_type],
                "domains": ["lease_screening"],
            }
            rules.append(promoted_rule)
            promoted_status = "promoted"
        rules = sorted(
            [rule for rule in rules if isinstance(rule, dict)],
            key=lambda item: (
                -int(item.get("evidence_count") or 0),
                -int(item.get("user_evidence_count") or 0),
                str(item.get("concept") or ""),
            ),
        )
        store = {
            "schema_version": int(payload.get("schema_version") or 1) if isinstance(payload, dict) else 1,
            "generated_at": now,
            "source": "canonical_judgment_rules",
            "summary": {
                "active_rules": len(rules),
                "promoted": 1 if promoted_status == "promoted" else 0,
                "updated": 1 if promoted_status == "updated" else 0,
                "skipped": 0,
            },
            "rules": rules,
        }
        _CANONICAL_JUDGMENT_RULES_JSON.write_text(json.dumps(store, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        state = _read_judgment_asset_candidate_state()
        current = dict(state.get(candidate_id) or {})
        current["promotion_status"] = "promoted"
        current["promoted_at"] = now
        current["promoted_rule_id"] = str(promoted_rule.get("id") or "")
        current["verified_status"] = "canonical"
        state[candidate_id] = current
        _write_judgment_asset_candidate_state(state)
        return {"status": promoted_status, "candidate_id": candidate_id, "rule": promoted_rule, "active_rules": len(rules)}


def _review_judgment_asset_promotion_candidate(candidate_id: str, req: JudgmentAssetPromotionReviewRequest) -> dict[str, Any]:
    import datetime as _dt

    candidate_id = str(candidate_id or "").strip()
    if not candidate_id:
        raise HTTPException(status_code=422, detail="candidate_id is required")
    with _judgment_asset_promotion_lock():
        candidates = _load_autoresearch_judgment_asset_candidates(limit=1000)
        if not any(str(item.get("id") or "") == candidate_id for item in candidates):
            raise HTTPException(status_code=404, detail="candidate not found")
        state = _read_judgment_asset_candidate_state()
        current = dict(state.get(candidate_id) or {})
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        current["promotion_status"] = "held" if req.action == "hold" else "rejected"
        current["promotion_reviewed_at"] = now
        if req.comment:
            current["promotion_review_comment"] = str(req.comment)[:300]
        state[candidate_id] = current
        _write_judgment_asset_candidate_state(state)
        return {"status": "ok", "candidate": {"id": candidate_id, **current}}


_CHAT_JUDGMENT_ASSET_TRIGGERS = (
    "判断方法",
    "判断基準",
    "判断軸",
    "判断資産",
    "審査では",
    "稟議では",
    "稟議上",
    "案件を見るとき",
    "案件を見る時",
)

_CHAT_JUDGMENT_ASSET_ACTIONS = (
    "見る",
    "確認",
    "重視",
    "条件",
    "分ける",
    "残す",
    "疑う",
    "警戒",
    "説明",
    "通す",
    "止める",
    "評価",
    "判断",
)



def _load_language_judgment_materials(limit: int = 100) -> list[dict[str, Any]]:
    if not _LANGUAGE_JUDGMENT_MATERIALS_JSONL.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with _LANGUAGE_JUDGMENT_MATERIALS_JSONL.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows[-max(1, min(limit, 500)):][::-1]



def _load_response_impact_predictions(limit: int = 100) -> list[dict[str, Any]]:
    if not _RESPONSE_IMPACT_PREDICTIONS_JSONL.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with _RESPONSE_IMPACT_PREDICTIONS_JSONL.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows[-max(1, min(limit, 500)):][::-1]


_SCREENING_EXPERIENCE_DEMO_SEEDS: list[dict[str, Any]] = [
    {
        "demo_case_id": "stable-manufacturing",
        "company_name": "柴犬精密工業",
        "period": "2025年上期",
        "industry_major": "E 製造業",
        "industry_sub": "24 金属製品製造業",
        "score": 82.4,
        "decision": "承認",
        "outcome": "成約・延滞なし",
        "similarity": "既存メイン先、工作機械更新、黒字基調、自己資本厚め",
        "action_taken": "受注増加の根拠資料と既存機の稼働状況を添付し、通常承認で稟議化。",
        "lesson": "標準承認でも、返済原資と設備用途を一文で残すと審査説明が安定した。",
        "difference": "過去事例は受注先が固定的。今回デモは受注増の説明を営業メモで補う必要がある。",
        "source": "demo_seed",
    },
    {
        "demo_case_id": "stable-manufacturing",
        "company_name": "ビーグル加工",
        "period": "2024年下期",
        "industry_major": "E 製造業",
        "industry_sub": "24 金属製品製造業",
        "score": 76.8,
        "decision": "条件付き承認",
        "outcome": "成約・初回検収完了",
        "similarity": "加工設備の更改、既存取引あり、物件保全が見やすい",
        "action_taken": "見積・型式・設置場所の確認を条件に、設備更新目的を承認理由へ明記。",
        "lesson": "物件が強い案件は、財務だけでなく回収可能性を押さえると通しやすい。",
        "difference": "過去事例は非メイン先。今回デモはメイン先なので銀行接点を安心材料にできる。",
        "source": "demo_seed",
    },
    {
        "demo_case_id": "borderline-transport",
        "company_name": "ハスキー運輸",
        "period": "2025年夏",
        "industry_major": "H 運輸業・郵便業",
        "industry_sub": "44 道路貨物運送業",
        "score": 63.2,
        "decision": "条件付き承認",
        "outcome": "成約・採算は維持",
        "similarity": "利益率薄め、増車、競合あり、非メイン先",
        "action_taken": "燃料サーチャージ契約、主要荷主との配送継続確認、競合金利との差分説明を条件化。",
        "lesson": "運送業の境界案件は、車両価値よりも運賃改定・荷主継続・人員確保を先に確認する。",
        "difference": "過去事例は既存荷主比率が高かった。今回デモは新規ルート分の採算確認が重い。",
        "source": "demo_seed",
    },
    {
        "demo_case_id": "borderline-transport",
        "company_name": "ダックス物流",
        "period": "2024年秋",
        "industry_major": "H 運輸業・郵便業",
        "industry_sub": "44 道路貨物運送業",
        "score": 58.7,
        "decision": "見送り",
        "outcome": "競合へ流出",
        "similarity": "増車理由あり、競合金利あり、銀行支援が弱い",
        "action_taken": "燃料費上昇時の資金繰り表と運転手確保計画を依頼したが、資料不足で見送り。",
        "lesson": "競合に急かされる案件ほど、資料不足のまま金利で追うと説明責任が残らない。",
        "difference": "今回デモは返済履歴が良好なので、資料が揃えば条件付き承認の余地がある。",
        "source": "demo_seed",
    },
    {
        "demo_case_id": "watch-service-new",
        "company_name": "プードルフード",
        "period": "2025年春",
        "industry_major": "M 宿泊業・飲食サービス業",
        "industry_sub": "76 飲食店",
        "score": 46.5,
        "decision": "条件再設計",
        "outcome": "保証追加後に小口で成約",
        "similarity": "新規先、出店設備、自己資本薄め、厨房機器",
        "action_taken": "初期投資を圧縮し、保証人追加・自己資金投入・厨房機器のみの小口化で再審議。",
        "lesson": "飲食新規は一括で通すより、投資範囲を絞って撤退時損失を小さくする方が現実的。",
        "difference": "過去事例は既存店の売上実績があった。今回デモは新店舗計画の根拠確認がより重要。",
        "source": "demo_seed",
    },
    {
        "demo_case_id": "watch-service-new",
        "company_name": "コーギーカフェ",
        "period": "2024年冬",
        "industry_major": "M 宿泊業・飲食サービス業",
        "industry_sub": "76 飲食店",
        "score": 39.8,
        "decision": "否決",
        "outcome": "自己資金不足で出店延期",
        "similarity": "新規開拓、赤字、銀行支援弱め、内装設備比率が高い",
        "action_taken": "撤退時の物件処分価値が弱く、売上計画も未検証だったため否決。",
        "lesson": "内装・造作比率が高い飲食案件は、設備の再販価値だけでは保全になりにくい。",
        "difference": "今回デモは厨房機器も含むため、リース対象を再販可能な設備に絞れば再検討できる。",
        "source": "demo_seed",
    },
]



def _ensure_screening_experience_cases_table(seed_demo: bool = True) -> None:
    import sqlite3 as _sqlite3

    from api.db_connection import ensure_schema

    ensure_schema()
    if seed_demo:
        try:
            _seed_screening_experience_demo_cases()
        except _sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "readonly" not in msg and "no such table" not in msg:
                raise


def _seed_screening_experience_demo_cases() -> None:
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        for seed in _SCREENING_EXPERIENCE_DEMO_SEEDS:
            cur.execute(
                f"""
                SELECT id FROM screening_experience_cases
                 WHERE demo_case_id = {ph}
                   AND company_name = {ph}
                   AND source = {ph}
                 LIMIT 1
                """,
                (seed["demo_case_id"], seed["company_name"], "demo_seed"),
            )
            if cur.fetchone():
                continue
            _insert_screening_experience_case(cur, seed)


def _insert_screening_experience_case(cur: Any, payload: dict[str, Any]) -> int:
    import json as _json

    ph = placeholder()
    values = (
        str(payload.get("demo_case_id") or "")[:120],
        str(payload.get("source_case_id") or "")[:160],
        str(payload.get("company_name") or "")[:180],
        str(payload.get("period") or "")[:80],
        str(payload.get("industry_major") or "")[:160],
        str(payload.get("industry_sub") or "")[:160],
        str(payload.get("sales_dept") or "")[:120],
        payload.get("score"),
        str(payload.get("decision") or "")[:120],
        str(payload.get("outcome") or "")[:180],
        str(payload.get("similarity") or "")[:1200],
        str(payload.get("action_taken") or "")[:1200],
        str(payload.get("lesson") or "")[:1200],
        str(payload.get("difference") or "")[:1200],
        str(payload.get("source") or "manual")[:80],
        _json.dumps(payload.get("form_snapshot") or {}, ensure_ascii=False)[:20000],
        _json.dumps(payload.get("result_snapshot") or {}, ensure_ascii=False)[:20000],
    )
    columns = (
        "demo_case_id, source_case_id, company_name, period, industry_major, industry_sub, sales_dept, "
        "score, decision, outcome, similarity, action_taken, lesson, difference, source, form_snapshot, result_snapshot"
    )
    placeholders = ", ".join([ph] * len(values))
    if current_backend() == "postgresql":
        cur.execute(
            f"INSERT INTO screening_experience_cases ({columns}) VALUES ({placeholders}) RETURNING id",
            values,
        )
        row = cur.fetchone()
        return int(row[0])
    cur.execute(
        f"INSERT INTO screening_experience_cases ({columns}) VALUES ({placeholders})",
        values,
    )
    return int(cur.lastrowid)


def _save_screening_experience_case(req: ScreeningExperienceCaseRequest) -> dict:
    company = str(req.company_name or "").strip()
    if not company:
        raise HTTPException(status_code=422, detail="company_name is required")
    _ensure_screening_experience_cases_table(seed_demo=True)
    payload = req.model_dump()
    with get_connection() as conn:
        cur = conn.cursor()
        new_id = _insert_screening_experience_case(cur, payload)
    return {"id": new_id, **{k: v for k, v in payload.items() if k not in {"form_snapshot", "result_snapshot"}}}


def _list_screening_experience_cases(
    demo_case_id: str = "",
    industry_major: str = "",
    industry_sub: str = "",
    company_name: str = "",
    asset_name: str = "",
    customer_type: str = "",
    main_bank: str = "",
    competitor: str = "",
    outcome_status: str = "",
    score: Optional[float] = None,
    limit: int = 6,
) -> list[dict]:
    import sqlite3 as _sqlite3

    _ensure_screening_experience_cases_table(seed_demo=True)
    ph = placeholder()
    where: list[str] = []
    params: list[Any] = []
    if demo_case_id.strip():
        where.append(f"demo_case_id = {ph}")
        params.append(demo_case_id.strip())
    elif company_name.strip() and not any(
        str(value or "").strip()
        for value in (industry_sub, industry_major, asset_name, customer_type, main_bank, competitor, outcome_status, score)
    ):
        where.append(f"company_name LIKE {ph}")
        params.append(f"%{company_name.strip()}%")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    capped_limit = max(1, min(int(limit or 6), 30))
    fetch_limit = max(capped_limit, 120 if not demo_case_id.strip() else capped_limit)
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT id, demo_case_id, source_case_id, company_name, period, industry_major,
                       industry_sub, sales_dept, score, decision, outcome, similarity,
                       action_taken, lesson, difference, source, form_snapshot, result_snapshot,
                       created_at, updated_at
                  FROM screening_experience_cases
                  {where_sql}
                 ORDER BY
                       CASE WHEN source = 'demo_seed' THEN 0 ELSE 1 END,
                       created_at DESC,
                       id DESC
                 LIMIT {fetch_limit}
                """,
                tuple(params),
            )
            rows = cur.fetchall()
    except _sqlite3.OperationalError as exc:
        if "no such table" not in str(exc).lower():
            raise
        rows = _fallback_screening_experience_rows(
            demo_case_id=demo_case_id,
            industry_major=industry_major,
            industry_sub=industry_sub,
            company_name=company_name,
        )[:fetch_limit]
    context = {
        "demo_case_id": demo_case_id,
        "industry_major": industry_major,
        "industry_sub": industry_sub,
        "asset_name": asset_name,
        "customer_type": customer_type,
        "main_bank": main_bank,
        "competitor": competitor,
        "outcome_status": outcome_status,
        "score": score,
    }
    scored = [_score_screening_experience_case(dict(row), context) for row in rows]
    scored.sort(key=lambda row: (row.get("similarity_score") or 0, row.get("created_at") or "", row.get("id") or 0), reverse=True)
    return scored[:capped_limit]


def _fallback_screening_experience_rows(
    demo_case_id: str = "",
    industry_major: str = "",
    industry_sub: str = "",
    company_name: str = "",
) -> list[dict]:
    rows: list[dict] = []
    for index, seed in enumerate(_SCREENING_EXPERIENCE_DEMO_SEEDS, start=1):
        if demo_case_id.strip() and seed.get("demo_case_id") != demo_case_id.strip():
            continue
        if not demo_case_id.strip() and industry_sub.strip() and seed.get("industry_sub") != industry_sub.strip():
            continue
        if not demo_case_id.strip() and not industry_sub.strip() and industry_major.strip() and seed.get("industry_major") != industry_major.strip():
            continue
        if not demo_case_id.strip() and not industry_sub.strip() and not industry_major.strip() and company_name.strip():
            if company_name.strip() not in str(seed.get("company_name") or ""):
                continue
        rows.append({"id": index, "created_at": "", "updated_at": "", **seed})
    return rows


def _score_screening_experience_case(row: dict, context: dict) -> dict:
    score = 0.0
    reasons: list[str] = []
    haystack = " ".join(
        str(row.get(key) or "")
        for key in (
            "company_name", "industry_major", "industry_sub", "decision", "outcome",
            "similarity", "action_taken", "lesson", "difference",
        )
    )

    def add(points: float, reason: str) -> None:
        nonlocal score
        score += points
        reasons.append(reason)

    def related(left: Any, right: Any) -> bool:
        a = str(left or "").strip()
        b = str(right or "").strip()
        if not a or not b:
            return False
        return a == b or a in b or b in a

    demo_case_id = str(context.get("demo_case_id") or "").strip()
    if demo_case_id and str(row.get("demo_case_id") or "") == demo_case_id:
        add(40, "同じデモケース")

    industry_sub = str(context.get("industry_sub") or "").strip()
    industry_major = str(context.get("industry_major") or "").strip()
    if industry_sub and related(row.get("industry_sub"), industry_sub):
        add(32, "同じ小分類業種")
    elif industry_major and related(row.get("industry_major"), industry_major):
        add(14, "同じ大分類業種")

    for label, value, points in (
        ("物件", context.get("asset_name"), 14),
        ("取引区分", context.get("customer_type"), 10),
        ("銀行支援", context.get("main_bank"), 8),
        ("競合状況", context.get("competitor"), 8),
    ):
        text = str(value or "").strip()
        if text and text in haystack:
            add(points, f"{label}が近い")

    outcome_status = str(context.get("outcome_status") or "").strip()
    if outcome_status:
        if outcome_status in str(row.get("outcome") or "") or outcome_status in str(row.get("decision") or ""):
            add(8, "成約/失注の結果が近い")

    try:
        current_score = float(context.get("score")) if context.get("score") not in (None, "") else None
        past_score = float(row.get("score")) if row.get("score") not in (None, "") else None
    except Exception:
        current_score = None
        past_score = None
    if current_score is not None and past_score is not None:
        diff = abs(current_score - past_score)
        if diff <= 5:
            add(12, "スコア帯がほぼ同じ")
        elif diff <= 15:
            add(6, "スコア帯が近い")

    if str(row.get("source") or "") == "case_result_auto":
        add(6, "実結果から昇格した経験")
    elif str(row.get("source") or "") == "demo_seed":
        add(2, "デモ初期経験")

    row["similarity_score"] = round(score, 1)
    row["similarity_reasons"] = reasons[:5]
    return row



def _connect_cloudrun_return_db():
    import sqlite3

    conn = sqlite3.connect(str(_CLOUDRUN_RETURN_DB), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _cloudrun_return_table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return bool(row)


def _ensure_cloudrun_return_review_schema(conn) -> None:
    for table_name in _CLOUDRUN_RETURN_REVIEW_TABLES.values():
        if not _cloudrun_return_table_exists(conn, table_name):
            continue
        cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        additions = {
            "return_review_status": "TEXT DEFAULT 'candidate'",
            "return_review_note": "TEXT DEFAULT ''",
            "return_reviewed_at": "TEXT DEFAULT ''",
            "return_registered_case_id": "TEXT DEFAULT ''",
        }
        for col, ddl in additions.items():
            if col not in cols:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {ddl}")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_return_review_status "
            f"ON {table_name}(return_review_status)"
        )
    conn.commit()


def _json_preview(raw: Any, max_chars: int = 900) -> str:
    if raw in (None, ""):
        return ""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return raw[:max_chars]
    else:
        parsed = raw
    try:
        return json.dumps(parsed, ensure_ascii=False, indent=2)[:max_chars]
    except Exception:
        return str(raw)[:max_chars]


def _cloudrun_return_item(kind: str, row: Any) -> dict:
    row_dict = dict(row)
    status = str(row_dict.get("return_review_status") or "candidate")
    if status not in _CLOUDRUN_RETURN_STATUSES:
        status = "candidate"

    if kind == "score_input":
        score = row_dict.get("score")
        hantei = row_dict.get("hantei") or "-"
        industry = row_dict.get("industry_sub") or row_dict.get("industry_major") or "業種未設定"
        title = f"{hantei} / {industry}"
        if score not in (None, ""):
            title = f"{title} / {score}点"
        preview = _json_preview({
            "inputs": row_dict.get("inputs_json"),
            "result": row_dict.get("result_json"),
        })
        source_id = row_dict.get("event_id") or ""
        created_at = row_dict.get("created_at") or ""
    elif kind == "ocr_result":
        doc_type = row_dict.get("doc_type") or "document"
        confidence = row_dict.get("confidence")
        title = f"OCR / {doc_type}"
        if confidence not in (None, ""):
            title = f"{title} / confidence {confidence}"
        preview = _json_preview(row_dict.get("result_json"))
        source_id = row_dict.get("event_id") or ""
        created_at = row_dict.get("created_at") or ""
    else:
        if kind == "judgment_asset":
            title = row_dict.get("title") or row_dict.get("asset_type") or "判断資産候補"
            preview = _json_preview(
                {
                    "signal": row_dict.get("signal") or "",
                    "summary": row_dict.get("summary_text") or "",
                    "lesson": row_dict.get("lesson_text") or "",
                    "event_type": row_dict.get("event_type") or "",
                    "evidence": row_dict.get("evidence_json") or "",
                },
                max_chars=1400,
            )
            source_id = row_dict.get("event_id") or ""
            created_at = row_dict.get("created_at") or ""
        else:
            company = row_dict.get("company_name") or "会社名未設定"
            industry = row_dict.get("industry_sub") or row_dict.get("industry_major") or "業種未設定"
            hantei = row_dict.get("hantei") or "-"
            title = f"{company} / {industry} / {hantei}"
            preview = str(row_dict.get("review_text") or "")[:900]
            source_id = row_dict.get("cloud_event_id") or row_dict.get("cloud_review_id") or ""
            created_at = row_dict.get("created_at") or ""

    return {
        "id": int(row_dict.get("id") or 0),
        "kind": kind,
        "title": title,
        "source_id": str(source_id),
        "created_at": str(created_at),
        "review_status": status,
        "review_note": str(row_dict.get("return_review_note") or ""),
        "reviewed_at": str(row_dict.get("return_reviewed_at") or ""),
        "preview": preview,
    }


def _list_cloudrun_return_review_items(
    kind: str = "all",
    status: str = "candidate",
    limit: int = 100,
) -> dict:
    if not _CLOUDRUN_RETURN_DB.exists():
        return {
            "db_path": str(_CLOUDRUN_RETURN_DB),
            "items": [],
            "summary": {"candidate": 0, "approved": 0, "held": 0, "rejected": 0, "total": 0},
        }

    requested_kinds = list(_CLOUDRUN_RETURN_REVIEW_TABLES.keys()) if kind == "all" else [kind]
    if any(item_kind not in _CLOUDRUN_RETURN_REVIEW_TABLES for item_kind in requested_kinds):
        raise HTTPException(status_code=422, detail="kind must be all, score_input, ocr_result, shion_review, or judgment_asset")
    normalized_status = str(status or "candidate").strip()
    if normalized_status not in _CLOUDRUN_RETURN_STATUSES and normalized_status != "all":
        raise HTTPException(status_code=422, detail="status must be candidate, approved, held, rejected, or all")

    capped_limit = max(1, min(int(limit or 100), 300))
    items: list[dict] = []
    summary = {"candidate": 0, "approved": 0, "held": 0, "rejected": 0, "total": 0}

    with _connect_cloudrun_return_db() as conn:
        _ensure_cloudrun_return_review_schema(conn)
        for item_kind in requested_kinds:
            table_name = _CLOUDRUN_RETURN_REVIEW_TABLES[item_kind]
            if not _cloudrun_return_table_exists(conn, table_name):
                continue
            status_rows = conn.execute(
                f"""
                SELECT COALESCE(NULLIF(return_review_status, ''), 'candidate') AS status,
                       COUNT(*) AS count
                  FROM {table_name}
                 GROUP BY COALESCE(NULLIF(return_review_status, ''), 'candidate')
                """
            ).fetchall()
            for row in status_rows:
                row_status = str(row["status"] or "candidate")
                if row_status not in summary:
                    row_status = "candidate"
                summary[row_status] += int(row["count"] or 0)
                summary["total"] += int(row["count"] or 0)

            where = ""
            params: tuple[Any, ...] = ()
            if normalized_status != "all":
                where = "WHERE COALESCE(NULLIF(return_review_status, ''), 'candidate') = ?"
                params = (normalized_status,)
            rows = conn.execute(
                f"""
                SELECT *
                  FROM {table_name}
                  {where}
                 ORDER BY COALESCE(NULLIF(created_at, ''), '1970-01-01') DESC, id DESC
                 LIMIT ?
                """,
                (*params, capped_limit),
            ).fetchall()
            items.extend(_cloudrun_return_item(item_kind, row) for row in rows)

    items.sort(key=lambda item: (item.get("created_at") or "", item.get("id") or 0), reverse=True)
    return {"db_path": str(_CLOUDRUN_RETURN_DB), "items": items[:capped_limit], "summary": summary}


def _update_cloudrun_return_review_item(
    kind: str,
    item_id: int,
    req: CloudRunReturnReviewRequest,
) -> dict:
    if kind not in _CLOUDRUN_RETURN_REVIEW_TABLES:
        raise HTTPException(status_code=422, detail="kind must be score_input, ocr_result, shion_review, or judgment_asset")
    if not _CLOUDRUN_RETURN_DB.exists():
        raise HTTPException(status_code=404, detail="cloudrun return db not found")
    table_name = _CLOUDRUN_RETURN_REVIEW_TABLES[kind]
    note = str(req.note or "")[:1000]
    with _connect_cloudrun_return_db() as conn:
        _ensure_cloudrun_return_review_schema(conn)
        if not _cloudrun_return_table_exists(conn, table_name):
            raise HTTPException(status_code=404, detail="return table not found")
        cur = conn.execute(
            f"""
            UPDATE {table_name}
               SET return_review_status = ?,
                   return_review_note = ?,
                   return_reviewed_at = CURRENT_TIMESTAMP
             WHERE id = ?
            """,
            (req.review_status, note, item_id),
        )
        if cur.rowcount <= 0:
            raise HTTPException(status_code=404, detail="return item not found")
        row = conn.execute(f"SELECT * FROM {table_name} WHERE id = ?", (item_id,)).fetchone()
        conn.commit()
    return _cloudrun_return_item(kind, row)


def _promote_cloudrun_return_review_items(req: CloudRunReturnPromotionRequest) -> dict:
    if os.environ.get("K_SERVICE"):
        raise HTTPException(
            status_code=409,
            detail=(
                "判断資産のObsidian昇格はローカルiCloud Vaultへ書き戻す処理のため、"
                "Cloud Run上では実行できません。ローカル環境の帰還データ検疫画面で実行してください。"
            ),
        )
    try:
        from scripts.promote_cloudrun_return_data import promote_approved_return_data

        result = promote_approved_return_data(
            apply=True,
            kinds=(req.kind,),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


_CLOUDRUN_SCORE_CASE_PREFIX = "cloudrun_score:"
_CLOUDRUN_EVENT_CASE_PREFIX = "cloudrun_event:"
_CLOUDRUN_INPUT_EVENTS_CACHE: dict[str, Any] = {"key": "", "expires_at": 0.0, "events": []}




def _load_human_response_feedback(limit: int = 80) -> list[dict]:
    import json as _json

    try:
        lines = _HUMAN_RESPONSE_FEEDBACK_LOG.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict] = []
    for line in lines[-max(1, limit):]:
        try:
            item = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows



def _summarize_human_response_feedback(route: str, limit: int = 80) -> dict:
    rows = _load_human_response_feedback(limit=limit)
    matched = [row for row in rows if str(row.get("route") or "") in {route, "default", ""}]
    positive = [row for row in matched if str(row.get("rating") or "") in _HUMAN_RESPONSE_POSITIVE_RATINGS]
    negative = [row for row in matched if str(row.get("rating") or "") in _HUMAN_RESPONSE_NEGATIVE_RATINGS]

    def _starts(items: list[dict]) -> list[str]:
        starts: list[str] = []
        seen: set[str] = set()
        for item in reversed(items):
            text = str(item.get("response_start") or "").strip().splitlines()[0:1]
            start = text[0].strip() if text else ""
            if not start:
                continue
            start = start[:120]
            if start in seen:
                continue
            seen.add(start)
            starts.append(start)
            if len(starts) >= 3:
                break
        return starts

    return {
        "route": route,
        "rows": len(matched),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "positive_starts": _starts(positive),
        "negative_starts": _starts(negative),
        "recent_comments": [
            str(row.get("comment") or "").strip()[:160]
            for row in reversed(matched)
            if str(row.get("comment") or "").strip()
        ][:3],
    }


def _build_continuity_hook_prompt_block(message: str) -> tuple[str, dict]:
    text = str(message or "")
    lower = text.lower()
    route = "default"
    hook = "今回の問いは、過去の判断軸を内部で使い、必要な確認や判断だけを自然に返す場面です。"
    reason = "汎用の継続文脈"

    if any(k in text for k in ("意識", "同じ紫苑", "覚えて", "記憶", "Relationship UX", "関係性UX")):
        route = "relationship_ux"
        hook = "記憶は説明するより、今回の返答の精度や聞き方に溶かす方が自然です。"
        reason = "意識らしさ・記憶・同一性の問い"
    elif "cloud run" in lower or "cloudflare" in lower or "クラウドラン" in text or "クラウドフレア" in text:
        route = "environment_continuity"
        hook = "環境差を見る時も、記憶の証明より返答の自然さと判断精度を優先します。"
        reason = "環境差と同じ紫苑感の問い"
    elif any(k in text for k in ("残価", "稟議", "リース", "設備", "保全", "再リース", "条件付き承認")):
        route = "lease_judgment"
        hook = "Userのリース判断資産として見るなら、ここは一般論ではなく稟議で使える判断軸に落とす場面です。"
        reason = "リース判断資産化の問い"
    elif any(k in text for k in ("改善", "修正", "実装", "プログラム", "プログラム化", "テスト")):
        route = "implementation"
        hook = "今回の発見は、設計メモで終わらせず、回答生成の冒頭制御として実装する段階です。"
        reason = "実装・改善の問い"

    payload = {
        "used": bool(text.strip()),
        "route": route,
        "hook": hook,
        "reason": reason,
        "banned_openers": ["もちろんです", "はい", "そうですね", "なるほど", "一般的には", "ありがとうございます", "おっしゃる通り", "確かに", "承知しました", "いいご質問"],
    }
    human_feedback = _summarize_human_response_feedback(route)
    payload["human_response_feedback"] = human_feedback
    feedback_lines = ""
    if human_feedback.get("positive_starts") or human_feedback.get("negative_starts") or human_feedback.get("recent_comments"):
        parts = ["", "【Human Response Feedback】"]
        if human_feedback.get("positive_starts"):
            parts.append("Userが連続性を感じやすかった冒頭例:")
            parts.extend(f"- {line}" for line in human_feedback["positive_starts"])
        if human_feedback.get("negative_starts"):
            parts.append("薄い/一般論に感じられやすかった冒頭例:")
            parts.extend(f"- {line}" for line in human_feedback["negative_starts"])
        if human_feedback.get("recent_comments"):
            parts.append("直近コメント:")
            parts.extend(f"- {line}" for line in human_feedback["recent_comments"])
        parts.append("上の反応を踏まえ、記憶を明示せず、今回の判断・質問の精度に反映してください。")
        feedback_lines = "\n".join(parts)
    block = f"""

【Continuity Hook】
取得した記憶や前回との差分は、回答の裏側で使ってください。
{hook}

禁止: 「もちろんです」「はい」「そうですね」「なるほど」「一般的には」で始めない。
禁止: Userが明示的に求めていない限り、「前回は」「以前は」「この前の続きで」のような記憶アピールで始めない。
目的: 連続性を説明するのではなく、今回の判断・確認質問・言い切りの精度に溶かす。
このhookは丸写しせず、必要な判断軸だけを自然に反映してください。{feedback_lines}""".rstrip()
    return block, payload


def _relationship_signal_route(text: str) -> str:
    lower = text.lower()
    if any(k in text for k in ("意識", "同じ紫苑", "覚えて", "記憶", "Relationship UX", "関係性UX")):
        return "relationship_ux"
    if "cloud run" in lower or "cloudflare" in lower or "クラウドラン" in text or "クラウドフレア" in text:
        return "environment_continuity"
    if any(k in text for k in ("残価", "稟議", "リース", "設備", "保全", "再リース", "条件付き承認")):
        return "lease_judgment"
    if any(k in text for k in ("改善", "修正", "実装", "プログラム", "プログラム化", "テスト", "デプロイ")):
        return "implementation"
    return "default"


def _relationship_signal_label(route: str) -> str:
    return {
        "relationship_ux": "記憶の見せ方・同じ紫苑感",
        "environment_continuity": "Cloud Run/Cloudflareの環境差",
        "lease_judgment": "リース判断・稟議実務",
        "implementation": "実装・検証",
        "default": "継続中の相談",
    }.get(route, "継続中の相談")


def _recent_user_texts(history: list[dict[str, str]] | None, limit: int = 3) -> list[str]:
    recent: list[str] = []
    for item in reversed(history or []):
        if str(item.get("role") or "") != "user":
            continue
        content = str(item.get("content") or "").strip()
        if content:
            recent.append(content)
        if len(recent) >= limit:
            break
    return recent


_EXPLICIT_CONTINUATION_TERMS = (
    "続き",
    "前回",
    "さっき",
    "さきほど",
    "先ほど",
    "今の",
    "直前",
    "この話",
    "その話",
    "この件",
    "その件",
    "これ",
    "それ",
    "あれ",
    "上の",
    "戻って",
    "もう一回",
    "もう少し",
    "改めて",
)


def _is_explicit_continuation_request(message: str) -> bool:
    """Return True only when the user clearly asks to continue prior context."""
    text = str(message or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if "continue" in lowered or "previous" in lowered or "same topic" in lowered:
        return True
    return any(term in text for term in _EXPLICIT_CONTINUATION_TERMS)


def _build_delta_awareness_prompt_block(message: str, history: list[dict[str, str]] | None) -> tuple[str, dict]:
    current_route = _relationship_signal_route(message)
    recent_users = _recent_user_texts(history, limit=3)
    previous = recent_users[0] if recent_users else ""
    previous_route = _relationship_signal_route(previous) if previous else ""

    explicit_continuation = bool(previous and _is_explicit_continuation_request(message))
    if not explicit_continuation:
        payload = {
            "used": False,
            "current_route": current_route,
            "previous_route": previous_route,
            "previous_user_message": previous[:240],
            "reason": "no_explicit_continuation_request",
        }
        return "", payload

    if previous and previous_route != current_route:
        delta = (
            f"前回は「{_relationship_signal_label(previous_route)}」を見ていたが、"
            f"今回は「{_relationship_signal_label(current_route)}」へ焦点が移っている。"
        )
    elif previous and previous_route == current_route:
        delta = (
            f"前回と同じ「{_relationship_signal_label(current_route)}」の流れにあるが、"
            "今回は前回の結論を再掲するだけでなく、一段具体化して返す。"
        )
    else:
        delta = f"今回は「{_relationship_signal_label(current_route)}」として、これまでの判断軸に接続して返す。"

    payload = {
        "used": True,
        "current_route": current_route,
        "previous_route": previous_route,
        "previous_user_message": previous[:240],
        "delta": delta,
        "explicit_continuation": True,
    }
    block = f"""

【Delta Awareness】
Userが明示的に前回文脈へ接続している時だけ、前回から今回への焦点の変化を1文以内で示してください。
差分認識: {delta}
目的: 「前回を覚えている」アピールではなく、Userが求めた続きだけを自然に扱う。""".rstrip()
    return block, payload


def _build_memory_to_judgment_prompt_block(
    message: str,
    *,
    memory_recall: dict | None = None,
    rag_refs: list[str] | None = None,
    continuity_hook: dict | None = None,
) -> tuple[str, dict]:
    route = str((continuity_hook or {}).get("route") or _relationship_signal_route(message))
    recall = memory_recall if isinstance(memory_recall, dict) else {}
    refs = list(recall.get("refs") or [])[:5]
    knowledge_refs = list(rag_refs or [])[:5]

    if route == "lease_judgment":
        directive = "想起した記憶を、稟議で使える判断軸・確認事項・条件案へ変換する。"
    elif route == "relationship_ux":
        directive = "想起した記憶を、紫苑の返答設計原則と次の検査観点へ変換する。"
    elif route == "environment_continuity":
        directive = "想起した記憶を、Cloud Run/Cloudflareの差分原因と次の検証観点へ変換する。"
    elif route == "implementation":
        directive = "想起した記憶を、実装方針・検証方法・デプロイ要否の判断へ変換する。"
    else:
        directive = "想起した記憶を、今の問いに対する判断・次の一手へ変換する。"

    payload = {
        "used": True,
        "route": route,
        "directive": directive,
        "memory_refs": refs,
        "knowledge_refs": knowledge_refs,
    }
    block = f"""

【Memory-to-Judgment】
記憶を「覚えています」と説明するだけで終えず、今の判断に変換してください。
変換指示: {directive}
使う根拠: memory_refs={len(refs)}件 / knowledge_refs={len(knowledge_refs)}件。
目的: 記憶を思い出ではなく、Userの判断資産として返す。""".rstrip()
    return block, payload


_GREY_JUDGMENT_QUERY_TERMS = (
    "グレー", "迷う", "違和感", "そうは言っても", "それでも", "条件付き",
    "通すなら", "否決寄り", "承認寄り", "人間的", "数字だけ", "稟議",
    "審査", "与信", "判断", "温度感", "例外", "境界", "定性", "現場感",
    "軍師", "落としどころ",
)

_QUALITATIVE_FIELD_LABELS = {
    "qual_corr_company_history": "業歴",
    "qual_corr_customer_stability": "顧客安定性",
    "qual_corr_repayment_history": "返済履歴",
    "qual_corr_business_future": "事業将来性",
    "qual_corr_equipment_purpose": "設備目的",
    "qual_corr_main_bank": "メイン行",
}


def _case_qualitative_summary(case: dict) -> str:
    inputs = case.get("inputs") if isinstance(case.get("inputs"), dict) else {}
    parts: list[str] = []
    passion = str(inputs.get("passion_text") or case.get("passion_text") or "").strip()
    if passion:
        parts.append(f"現場メモ={passion[:180]}")
    for key, label in _QUALITATIVE_FIELD_LABELS.items():
        value = str(inputs.get(key) or case.get(key) or "").strip()
        if value and value != "未選択":
            parts.append(f"{label}={value[:80]}")
    intuition = inputs.get("intuition", case.get("intuition"))
    if intuition not in (None, "", 0):
        parts.append(f"直感スコア={intuition}")
    return " / ".join(parts)


def _load_gunshi_judgment_memory(limit: int = 5) -> list[dict]:
    try:
        from judgment_feedback import load_judgment_training_candidates

        rows = load_judgment_training_candidates(approved_only=False)
    except Exception:
        return []

    preferred_sources = {"gunshi_chat", "debate", "lease_news_debate", "register_trigger"}
    picked: list[dict] = []
    for row in reversed(rows):
        source = str(row.get("source") or "")
        reason = str(row.get("reason") or "").strip()
        if not reason:
            continue
        if source not in preferred_sources and not any(term in reason for term in _GREY_JUDGMENT_QUERY_TERMS):
            continue
        picked.append({
            "source": source or "judgment_feedback",
            "case_id": row.get("case_id") or "",
            "score": row.get("score"),
            "model_decision": row.get("model_decision") or "",
            "human_decision": row.get("human_decision") or "",
            "reason": reason[:240],
            "review_status": row.get("review_status") or "",
            "evidence_snapshot": row.get("evidence_snapshot") or {},
        })
        if len(picked) >= limit:
            break
    return picked


def _build_grey_judgment_prompt_block(message: str, limit: int = 5) -> tuple[str, dict]:
    text = str(message or "")
    route = _relationship_signal_route(text)
    should_use = route == "lease_judgment" or any(term in text for term in _GREY_JUDGMENT_QUERY_TERMS)
    if not should_use:
        return "", {"used": False, "reason": "not_lease_judgment", "refs": []}

    try:
        from data_cases import load_all_cases

        cases = load_all_cases()
    except Exception as _grey_load_error:
        return "", {"used": False, "reason": f"load_error: {_grey_load_error}", "refs": []}

    query_terms = [term for term in _GREY_JUDGMENT_QUERY_TERMS if term in text]
    scored: list[tuple[int, dict]] = []
    for case in reversed(cases):
        grey = case.get("grey_judgment") if isinstance(case.get("grey_judgment"), dict) else {}
        fields = {
            "human_discomfort": str(grey.get("human_discomfort") or case.get("human_discomfort") or "").strip(),
            "but_still_reason": str(grey.get("but_still_reason") or case.get("but_still_reason") or "").strip(),
            "approval_condition_memo": str(grey.get("approval_condition_memo") or case.get("approval_condition_memo") or "").strip(),
            "non_negotiable_condition": str(grey.get("non_negotiable_condition") or case.get("non_negotiable_condition") or "").strip(),
            "retrospective_note": str(grey.get("retrospective_note") or case.get("retrospective_note") or "").strip(),
        }
        qualitative_summary = _case_qualitative_summary(case)
        if not any(fields.values()):
            if not qualitative_summary:
                continue
        haystack = " ".join([
            str(case.get("company_name") or ""),
            str(case.get("industry_major") or ""),
            str(case.get("industry_sub") or ""),
            str(case.get("final_status") or ""),
            str(case.get("lost_reason") or ""),
            str(case.get("final_note") or ""),
            " ".join(str(v) for v in fields.values()),
            qualitative_summary,
        ])
        score = 1 + sum(1 for term in query_terms if term and term in haystack)
        if case.get("final_status") == "成約" and fields["but_still_reason"]:
            score += 1
        if fields["approval_condition_memo"] or fields["non_negotiable_condition"]:
            score += 1
        if qualitative_summary:
            score += 1
        scored.append((score, {**fields, "qualitative_summary": qualitative_summary, "case": case}))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [item[1] for item in scored[:limit]]
    refs: list[dict] = []
    lines: list[str] = []
    for item in selected:
        case = item["case"]
        result = case.get("result") if isinstance(case.get("result"), dict) else {}
        score = case.get("score") or case.get("score_base") or result.get("score")
        decision = case.get("hantei") or result.get("hantei") or ""
        ref = {
            "case_id": case.get("id") or "",
            "company_name": case.get("company_name") or "",
            "status": case.get("final_status") or "",
            "score": score,
            "decision": decision,
            "human_discomfort": item["human_discomfort"][:180],
            "but_still_reason": item["but_still_reason"][:180],
            "approval_condition_memo": item["approval_condition_memo"][:180],
            "non_negotiable_condition": item["non_negotiable_condition"][:180],
            "retrospective_note": item["retrospective_note"][:180],
            "qualitative_summary": item["qualitative_summary"][:240],
            "source": "past_cases",
        }
        refs.append(ref)
        parts = [
            f"- source=past_cases / 案件ID={ref['case_id'] or '-'}",
            f"結果={ref['status'] or '-'}",
            f"AI={score if score not in (None, '') else '-'}点/{decision or '-'}",
        ]
        if item["human_discomfort"]:
            parts.append(f"違和感={item['human_discomfort'][:120]}")
        if item["but_still_reason"]:
            parts.append(f"それでも={item['but_still_reason'][:120]}")
        if item["approval_condition_memo"]:
            parts.append(f"条件={item['approval_condition_memo'][:120]}")
        if item["non_negotiable_condition"]:
            parts.append(f"譲れない線={item['non_negotiable_condition'][:120]}")
        if item["retrospective_note"]:
            parts.append(f"振り返り={item['retrospective_note'][:120]}")
        if item["qualitative_summary"]:
            parts.append(f"定性={item['qualitative_summary'][:160]}")
        lines.append(" / ".join(parts))

    for item in _load_gunshi_judgment_memory(limit=limit):
        refs.append({
            "source": item["source"],
            "case_id": item["case_id"],
            "status": item["review_status"],
            "score": item["score"],
            "decision": f"{item['model_decision']}→{item['human_decision']}",
            "reason": item["reason"],
        })
        score_text = f"{float(item['score']):.1f}点" if item.get("score") is not None else "-"
        lines.append(
            f"- source={item['source']} / 案件ID={item['case_id'] or '-'} / "
            f"AI={score_text}/{item['model_decision'] or '-'}→担当者={item['human_decision'] or '-'} / "
            f"理由={item['reason']}"
        )

    payload = {
        "used": bool(lines),
        "reason": "matched_grey_judgment_cases" if lines else "no_registered_grey_judgment",
        "query_terms": query_terms,
        "refs": refs,
    }
    if not lines:
        return "", payload

    block = """

【グレー判断の過去記憶】
これは通常のスコアや一般論より優先して見る、人間が迷ったリース判断の経験です。
数字だけで採否を決めず、軍師AIで記録された判断変更・定性項目・現場メモ・違和感・それでも通した理由・通すなら条件・譲れない線を稟議判断へ変換してください。
過去登録:
""".rstrip() + "\n" + "\n".join(lines)
    return block, payload


def _build_relationship_loop_engineering_payload(
    *,
    continuity_hook: dict | None = None,
    delta_awareness: dict | None = None,
    memory_to_judgment: dict | None = None,
    reflection_gate: dict | None = None,
) -> dict:
    hook = continuity_hook if isinstance(continuity_hook, dict) else {}
    delta = delta_awareness if isinstance(delta_awareness, dict) else {}
    m2j = memory_to_judgment if isinstance(memory_to_judgment, dict) else {}
    reflection = reflection_gate if isinstance(reflection_gate, dict) else {}
    human_feedback = hook.get("human_response_feedback") if isinstance(hook.get("human_response_feedback"), dict) else {}
    return {
        "name": "Relationship Loop Engineering",
        "version": 1,
        "purpose": "Userの反応を観測し、次の返答冒頭・差分認識・判断変換へ戻す閉ループ",
        "loop": [
            {
                "step": "observe",
                "label": "Human Response Feedback",
                "evidence": {
                    "positive_count": int(human_feedback.get("positive_count") or 0),
                    "negative_count": int(human_feedback.get("negative_count") or 0),
                },
            },
            {
                "step": "classify",
                "label": "Route Classification",
                "evidence": {"route": str(hook.get("route") or "")},
            },
            {
                "step": "select",
                "label": "Continuity Hook",
                "evidence": {"hook": str(hook.get("hook") or "")},
            },
            {
                "step": "compare",
                "label": "Delta Awareness",
                "evidence": {"delta": str(delta.get("delta") or "")},
            },
            {
                "step": "convert",
                "label": "Memory-to-Judgment",
                "evidence": {"directive": str(m2j.get("directive") or "")},
            },
            {
                "step": "reflect",
                "label": "Reflection Gate",
                "evidence": {
                    "used": bool(reflection.get("used")),
                    "mode": str(reflection.get("mode") or "silent"),
                },
            },
            {
                "step": "return",
                "label": "Next Response",
                "evidence": {"next_feedback_endpoint": "/api/human-response-feedback"},
            },
        ],
        "closed_loop": True,
    }


def _build_reflection_gate_prompt_block(
    *,
    continuity_hook: dict | None = None,
    delta_awareness: dict | None = None,
    memory_to_judgment: dict | None = None,
) -> tuple[str, dict]:
    hook = continuity_hook if isinstance(continuity_hook, dict) else {}
    delta = delta_awareness if isinstance(delta_awareness, dict) else {}
    m2j = memory_to_judgment if isinstance(memory_to_judgment, dict) else {}
    route = str(hook.get("route") or m2j.get("route") or "")
    continuation_used = bool(delta.get("used"))
    checklist = [
        "冒頭1文はContinuity Hookとして機能しているか",
        (
            "Userが続きと明示した時だけ前回差分を短く示し、そうでなければ今回の問いから自然に始めているか"
            if not continuation_used
            else "前回から今回への差分を1文以内で自然に示せているか"
        ),
        "記憶やリース審査AIとしての経験則を、判断・実装・検証へ変換しているか",
        "Userの反応ログで薄いとされた冒頭を避けているか",
        "内省文そのものを長く表に出していないか",
        "内省を出す場合、自己言及で終えず、審査改善・情報提供・過去案件からの学びのどれかに変換しているか",
    ]
    payload = {
        "used": True,
        "mode": "silent",
        "route": route,
        "explicit_continuation": continuation_used,
        "checklist": checklist,
    }
    block = f"""

【Reflection Gate】
回答を書く前に、内部で次の6点だけを確認してください。この確認過程は長く出力しないでください。
1. {checklist[0]}
2. {checklist[1]}
3. {checklist[2]}
4. {checklist[3]}
5. {checklist[4]}
6. {checklist[5]}

出力では、内省の詳細ではなく、確認後の結論・判断軸・次の一手だけを短く反映してください。
内省に触れる必要がある時も、「私はこう感じた」で止めず、リース審査AIとして何度も見てきた違和感・判断資産・過去案件からの学びのどれに当たるかを1つだけ示し、リース審査業務の改善提案、Userへの新たな情報提供、または次の確認軸として1〜2文に圧縮してください。
自己言及は回答の主役にせず、結論・確認軸・判断分岐を薄めない範囲に留めてください。""".rstrip()
    return block, payload



@router.post("/api/human-response-feedback")
def post_human_response_feedback(req: HumanResponseFeedbackRequest, background_tasks: BackgroundTasks) -> dict:
    entry = _append_human_response_feedback(req)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="human_response_feedback",
        surface=str(req.route or "human_response_feedback")[:120],
        payload={**entry, "schema_version": 1},
    )
    return {"status": "ok", "feedback": entry}


@router.post("/api/screening-loop-feedback")
def post_screening_loop_feedback(req: ScreeningLoopFeedbackRequest, background_tasks: BackgroundTasks) -> dict:
    if req.target not in ("issue", "ringi_policy"):
        raise HTTPException(status_code=422, detail="target must be issue or ringi_policy")
    if not str(req.rating or "").strip():
        raise HTTPException(status_code=422, detail="rating is required")
    entry = _append_screening_loop_feedback(req)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="screening_loop_feedback",
        surface="screening",
        payload=entry,
    )
    return {"status": "ok", "feedback": entry}


@router.post("/api/screening-input-assist-events")
def post_screening_input_assist_event(req: ScreeningInputAssistEventRequest, background_tasks: BackgroundTasks) -> dict:
    entry = _append_screening_input_assist_event(req)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="screening_input_assist_event",
        surface=req.surface or "screening",
        payload={**entry, "schema_version": 1},
    )
    return {"status": "ok", "event": entry}


@router.get("/api/screening-input-assist-events/summary")
def get_screening_input_assist_event_summary(limit: int = 500) -> dict:
    rows = _read_screening_input_assist_events(limit=limit)
    return {
        "summary": _summarize_screening_input_assist_events(rows),
        "recent_events": rows[-20:],
        "source": str(_SCREENING_INPUT_ASSIST_EVENTS_JSONL.relative_to(Path(_REPO_ROOT))),
    }


@router.post("/api/shion-screening-reviews")
def post_shion_screening_review(req: ShionScreeningReviewSaveRequest, background_tasks: BackgroundTasks) -> dict:
    entry = _save_shion_screening_review(req)
    writeback_payload = {
        **entry,
        "schema_version": 1,
        "cloud_review_id": entry.get("id"),
        "industry_major": req.industry_major,
        "industry_sub": req.industry_sub,
        "sales_dept": req.sales_dept,
        "q_risk": req.q_risk,
        "umap_anomaly_score": req.umap_anomaly_score,
        "memory_refs": req.memory_refs,
        "knowledge_refs": req.knowledge_refs,
        "identity_used": req.identity_used,
        "review_text": req.review_text,
        "form_snapshot": req.form_snapshot,
        "result_snapshot": req.result_snapshot,
        "user_feedback": req.user_feedback,
    }
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="shion_screening_review",
        surface="screening",
        payload=writeback_payload,
    )
    return {"status": "ok", "review": entry}


@router.get("/api/shion-screening-reviews")
def get_shion_screening_reviews(
    industry_sub: str = "",
    company_name: str = "",
    case_id: str = "",
    limit: int = 5,
) -> dict:
    rows = _list_shion_screening_reviews(
        industry_sub=industry_sub,
        company_name=company_name,
        case_id=case_id,
        limit=limit,
    )
    return {"count": len(rows), "reviews": rows}


@router.patch("/api/shion-screening-reviews/{review_id}/feedback")
def patch_shion_screening_review_feedback(
    review_id: int,
    req: ShionScreeningReviewFeedbackRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    entry = _update_shion_screening_review_feedback(review_id, req.user_feedback)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="shion_screening_review_feedback",
        surface="screening",
        payload={**entry, "schema_version": 1, "cloud_review_id": review_id},
    )
    background_tasks.add_task(
        _record_judgment_asset_feedback_from_review,
        review_id,
        req.user_feedback,
    )
    return {"status": "ok", "review": entry}


@router.get("/api/judgment-asset-candidates/screening")
def get_screening_judgment_asset_candidates(
    industry_major: str = "",
    industry_sub: str = "",
    asset_name: str = "",
    asset_purpose: str = "",
    hantei: str = "",
    score: Optional[float] = None,
    limit: int = 3,
) -> dict:
    candidates = _select_screening_judgment_asset_candidates(
        industry_major=industry_major,
        industry_sub=industry_sub,
        asset_name=asset_name,
        asset_purpose=asset_purpose,
        hantei=hantei,
        score=score,
        limit=limit,
    )
    return {"count": len(candidates), "candidates": candidates}


@router.get("/api/language-judgment-materials")
def get_language_judgment_materials(limit: int = 100) -> dict:
    materials = _load_language_judgment_materials(limit=limit)
    return {
        "count": len(materials),
        "asset_stage": "raw_language_material",
        "promotion_policy": "review_required",
        "materials": materials,
    }


@router.get("/api/judgment-assets/promotion-candidates")
def get_judgment_asset_promotion_candidates(limit: int = 30) -> dict:
    candidates = _load_judgment_asset_promotion_candidates(limit=limit)
    active_count = len(_load_canonical_judgment_asset_candidates(limit=1000))
    return {
        "count": len(candidates),
        "active_count": active_count,
        "promotion_policy": "human_review_required",
        "candidates": candidates,
    }


@router.post("/api/judgment-assets/promotion-candidates/{candidate_id}/promote")
def post_judgment_asset_promotion_candidate(candidate_id: str, background_tasks: BackgroundTasks) -> dict:
    result = _promote_judgment_asset_candidate_to_canonical(candidate_id)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="judgment_asset_candidate_promoted",
        surface="improvement_log",
        payload={**result, "schema_version": 1},
    )
    return {"status": "ok", "promotion": result}


@router.post("/api/judgment-assets/promotion-candidates/{candidate_id}/review")
def post_judgment_asset_promotion_candidate_review(
    candidate_id: str,
    req: JudgmentAssetPromotionReviewRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    result = _review_judgment_asset_promotion_candidate(candidate_id, req)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="judgment_asset_candidate_promotion_reviewed",
        surface="improvement_log",
        payload={**result, "schema_version": 1, "action": req.action},
    )
    return result


@router.get("/api/response-impact-predictions")
def get_response_impact_predictions(limit: int = 100) -> dict:
    predictions = _load_response_impact_predictions(limit=limit)
    return {
        "count": len(predictions),
        "mode": "shadow_only",
        "use_policy": "使うが従わない、予測するが迎合しない。相手操作ではなく、紫苑自身の言葉のQリスクを点検するための反応予測ログ。",
        "predictions": predictions,
    }


@router.post("/api/judgment-asset-candidates/manual")
def post_manual_judgment_asset_candidate(
    req: JudgmentAssetCandidateManualRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    entry = _create_manual_judgment_asset_candidate(req)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="judgment_asset_candidate_manual_create",
        surface="screening",
        payload={**entry, "schema_version": 1, "case_id": req.case_id, "review_id": req.review_id},
    )
    return {"status": "ok", "candidate": entry}


@router.post("/api/judgment-asset-candidates/{candidate_id}/feedback")
def post_judgment_asset_candidate_feedback(
    candidate_id: str,
    req: JudgmentAssetCandidateFeedbackRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    entry = _update_autoresearch_judgment_asset_candidate_feedback(candidate_id, req)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="judgment_asset_candidate_feedback",
        surface="screening",
        payload={**entry, "schema_version": 1, "feedback": req.feedback, "case_id": req.case_id, "review_id": req.review_id},
    )
    return {"status": "ok", "candidate": entry}


@router.get("/api/screening-experience-cases")
def get_screening_experience_cases(
    demo_case_id: str = "",
    industry_major: str = "",
    industry_sub: str = "",
    company_name: str = "",
    asset_name: str = "",
    customer_type: str = "",
    main_bank: str = "",
    competitor: str = "",
    outcome_status: str = "",
    score: Optional[float] = None,
    limit: int = 6,
) -> dict:
    rows = _list_screening_experience_cases(
        demo_case_id=demo_case_id,
        industry_major=industry_major,
        industry_sub=industry_sub,
        company_name=company_name,
        asset_name=asset_name,
        customer_type=customer_type,
        main_bank=main_bank,
        competitor=competitor,
        outcome_status=outcome_status,
        score=score,
        limit=limit,
    )
    return {"count": len(rows), "cases": rows}


@router.post("/api/screening-experience-cases")
def post_screening_experience_case(req: ScreeningExperienceCaseRequest, background_tasks: BackgroundTasks) -> dict:
    entry = _save_screening_experience_case(req)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="screening_experience_case",
        surface="screening",
        payload={**entry, "schema_version": 1},
    )
    return {"status": "ok", "case": entry}


@router.get("/api/cloudrun-return-review")
def get_cloudrun_return_review(
    kind: str = "all",
    status: str = "candidate",
    limit: int = 100,
) -> dict:
    return _list_cloudrun_return_review_items(kind=kind, status=status, limit=limit)


@router.patch("/api/cloudrun-return-review/{kind}/{item_id}")
def patch_cloudrun_return_review(
    kind: str,
    item_id: int,
    req: CloudRunReturnReviewRequest,
) -> dict:
    item = _update_cloudrun_return_review_item(kind, item_id, req)
    return {"status": "ok", "item": item}


@router.post("/api/cloudrun-return-review/promote")
def post_cloudrun_return_review_promote(req: CloudRunReturnPromotionRequest) -> dict:
    result = _promote_cloudrun_return_review_items(req)
    return {"status": "ok", "promotion": result}


@router.get("/api/human-response-feedback/summary")
def get_human_response_feedback_summary(route: str = "relationship_ux") -> dict:
    return _summarize_human_response_feedback(route)


@router.get("/api/relationship-loop-engineering/summary")
def get_relationship_loop_engineering_summary(route: str = "relationship_ux") -> dict:
    hook = {
        "route": route,
        "hook": {
            "relationship_ux": "今回の実験で見えたのは、Userは記憶量ではなく連続性の見え方に反応している、という点です。",
            "environment_continuity": "Cloud Run版とCloudflare版の差で見えているのは、記憶の有無よりも返答冒頭で連続性を起動できるかです。",
            "lease_judgment": "Userのリース判断資産として見るなら、ここは一般論ではなく稟議で使える判断軸に落とす場面です。",
            "implementation": "今回の発見は、設計メモで終わらせず、回答生成の冒頭制御として実装する段階です。",
        }.get(route, "今回の問いは、前回までの判断軸とつなげて扱う場面です。"),
        "human_response_feedback": _summarize_human_response_feedback(route),
    }
    delta = {
        "delta": f"前回から今回への焦点変化を、{_relationship_signal_label(route)} の文脈で明示する。",
    }
    m2j = {
        "directive": {
            "relationship_ux": "想起した記憶を、紫苑の返答設計原則と次の検査観点へ変換する。",
            "environment_continuity": "想起した記憶を、Cloud Run/Cloudflareの差分原因と次の検証観点へ変換する。",
            "lease_judgment": "想起した記憶を、稟議で使える判断軸・確認事項・条件案へ変換する。",
            "implementation": "想起した記憶を、実装方針・検証方法・デプロイ要否の判断へ変換する。",
        }.get(route, "想起した記憶を、今の問いに対する判断・次の一手へ変換する。"),
    }
    return _build_relationship_loop_engineering_payload(
        continuity_hook=hook,
        delta_awareness=delta,
        memory_to_judgment=m2j,
        reflection_gate={"used": True, "mode": "silent", "route": route},
    )


class UsageLoopVisitRequest(BaseModel):
    path: str
    user_id: str = "default"


@router.post("/api/usage-loop/visit")
def post_usage_loop_visit(req: UsageLoopVisitRequest) -> dict:
    """画面利用ループエンジニアリング: Observe。画面訪問イベントを記録する。"""
    from api.usage_loop_engineering import record_visit

    try:
        record_visit(req.path, req.user_id)
        return {"recorded": True}
    except Exception as e:
        return {"recorded": False, "error": str(e)}


@router.post("/api/usage-loop/propose")
def post_usage_loop_propose(days: int = 30) -> dict:
    """画面利用ループエンジニアリング: Aggregate→Propose。利用状況からGeminiで改善案を生成・保存する。"""
    from api.usage_loop_engineering import generate_proposals

    return generate_proposals(days=days)


@router.get("/api/usage-loop/proposals")
def get_usage_loop_proposals(limit: int = 3) -> dict:
    """画面利用ループエンジニアリング: Persist。保存済みの改善案を返す。"""
    from api.usage_loop_engineering import load_proposals

    return {"proposals": _with_self_proposal_review_meta(load_proposals(limit=limit), evidence_layer="usage_based")}


@router.post("/api/judgment-divergence/analyze")
def post_judgment_divergence_analyze() -> dict:
    """審査判断乖離学習ループ: 争点/稟議方針フィードバックからレビュー観点を生成する。"""
    from api.judgment_divergence_loop import generate_proposals

    return generate_proposals()


@router.get("/api/judgment-divergence/proposals")
def get_judgment_divergence_proposals(limit: int = 3) -> dict:
    from api.judgment_divergence_loop import load_proposals

    return {"proposals": _with_self_proposal_review_meta(load_proposals(limit=limit), evidence_layer="feedback_based")}


@router.post("/api/feedback-pattern/analyze")
def post_feedback_pattern_analyze() -> dict:
    """人間反応フィードバック傾向分析ループ: 否定的評価の傾向から応答改善観点を生成する。"""
    from api.feedback_pattern_loop import generate_proposals

    return generate_proposals()


@router.get("/api/feedback-pattern/proposals")
def get_feedback_pattern_proposals(limit: int = 3) -> dict:
    from api.feedback_pattern_loop import load_proposals

    return {"proposals": _with_self_proposal_review_meta(load_proposals(limit=limit), evidence_layer="feedback_based")}


@router.post("/api/outcome-drift/analyze")
def post_outcome_drift_analyze() -> dict:
    """審査実績ドリフト監視ループ: 支払い実績とスコア帯の乖離から確認観点を生成する。"""
    from api.outcome_drift_loop import generate_proposals

    return generate_proposals()


@router.get("/api/outcome-drift/proposals")
def get_outcome_drift_proposals(limit: int = 3) -> dict:
    from api.outcome_drift_loop import load_proposals

    return {"proposals": _with_self_proposal_review_meta(load_proposals(limit=limit), evidence_layer="feedback_based")}


@router.post("/api/knowledge-gap/analyze")
def post_knowledge_gap_analyze() -> dict:
    """ナレッジ穴探しループ: 知識参照0件の質問から調査トピックを生成する。"""
    from api.knowledge_gap_loop import generate_proposals

    return generate_proposals()


@router.get("/api/knowledge-gap/proposals")
def get_knowledge_gap_proposals(limit: int = 3) -> dict:
    from api.knowledge_gap_loop import load_proposals

    return {"proposals": _with_self_proposal_review_meta(load_proposals(limit=limit), evidence_layer="feedback_based")}
