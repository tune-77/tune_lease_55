"""Usage logging and review inbox for Shion's agentic skill tools.

This module records when Shion uses agentic skill tools and mirrors reviewable
outputs into a small inbox. It never promotes memory, writes Obsidian, changes
RAG, or alters scoring.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from api.loop_engineering_common import DATA_DIR, append_jsonl, load_jsonl

USAGE_LOG_PATH = DATA_DIR / "agentic_skill_usage_log.jsonl"
REVIEW_INBOX_PATH = DATA_DIR / "agentic_skill_review_inbox.jsonl"
REVIEW_LOG_PATH = DATA_DIR / "agentic_skill_review_decisions.jsonl"

AGENTIC_SKILL_TOOL_NAMES = {
    "structure_judgment_asset_candidate",
    "validate_lease_source_summary",
    "convert_research_to_screening_insights",
    "build_screening_decision_flow",
    "write_scqa_report",
    "inspect_agentic_skill_flow",
    "propose_agentic_skill_next_actions",
    "search_shion_system_context",
    "propose_shion_system_improvement_focus",
}

REVIEWABLE_TOOL_NAMES = {
    "structure_judgment_asset_candidate",
    "convert_research_to_screening_insights",
    "build_screening_decision_flow",
}

VALID_REVIEW_DECISIONS = {"adopted", "revised", "held", "rejected"}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _text(value: Any, *, maximum: int = 1000) -> str:
    return str(value or "").strip()[:maximum]


def _short_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _term_tokens(text: str) -> set[str]:
    """英数字トークンと和文の漢字/カタカナ連を抽出する（過去レビューとの類似判定用）。"""
    raw = re.findall(r"[A-Za-z0-9_./-]{2,}|[一-龥ぁ-んァ-ヶー]{2,}", text or "")
    terms: set[str] = set(raw)
    for token in raw:
        terms.update(re.findall(r"[一-龥]{2,}|[ァ-ヶー]{2,}", token))
    return terms


def _safe_payload(value: Any, *, max_chars: int = 5000) -> Any:
    """Make ADK tool payloads JSON-serializable and bounded."""
    try:
        raw = json.dumps(value, ensure_ascii=False, default=str)
        loaded = json.loads(raw)
    except Exception:
        loaded = str(value)
    text = json.dumps(loaded, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return loaded
    return {"truncated": True, "preview": text[:max_chars]}


def _tool_response_payload(function_response: Any) -> Any:
    for attr in ("response", "result", "content"):
        if hasattr(function_response, attr):
            value = getattr(function_response, attr)
            if value is not None:
                return value
    if isinstance(function_response, dict):
        return function_response
    return str(function_response)


def _unwrap_tool_result(payload: Any) -> Any:
    """Unwrap common ADK/genai function response envelopes."""
    if not isinstance(payload, dict):
        return payload
    for key in ("result", "output", "return_value"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def record_agentic_skill_call(
    *,
    tool_name: str,
    args: Any = None,
    session_id: str = "",
    case_context: dict[str, Any] | None = None,
    usage_log_path: Path = USAGE_LOG_PATH,
) -> dict[str, Any] | None:
    """Record an agentic skill call event, if the tool is in scope."""
    if tool_name not in AGENTIC_SKILL_TOOL_NAMES:
        return None
    event = {
        "id": _short_hash(["call", tool_name, session_id, args, _now_iso()]),
        "event_type": "agentic_skill_call",
        "tool_name": tool_name,
        "session_id": session_id,
        "case_context": _safe_payload(case_context or {}, max_chars=1800),
        "args": _safe_payload(args or {}, max_chars=2200),
        "recorded_at": _now_iso(),
        "guardrail": "usage_log_only_no_side_effect",
    }
    append_jsonl(usage_log_path, event)
    return event


def _review_claim(tool_name: str, payload: Any) -> str:
    if not isinstance(payload, dict):
        return _text(payload, maximum=700)
    if tool_name == "structure_judgment_asset_candidate":
        candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
        return _text(
            candidate.get("core_judgment")
            or candidate.get("recommended_action")
            or candidate.get("source_note"),
            maximum=700,
        )
    if tool_name == "convert_research_to_screening_insights":
        checks = payload.get("top_checks") if isinstance(payload.get("top_checks"), list) else []
        if checks and isinstance(checks[0], dict):
            return _text(checks[0].get("question") or checks[0].get("why"), maximum=700)
        return _text(payload.get("comment_draft"), maximum=700)
    if tool_name == "build_screening_decision_flow":
        notes = payload.get("notes") if isinstance(payload.get("notes"), list) else []
        origins = payload.get("risk_origins") if isinstance(payload.get("risk_origins"), list) else []
        return _text(" / ".join([*(str(item) for item in origins[:3]), *(str(item) for item in notes[:2])]), maximum=700)
    return _text(payload, maximum=700)


def _review_type(tool_name: str) -> str:
    return {
        "structure_judgment_asset_candidate": "judgment_asset_candidate",
        "convert_research_to_screening_insights": "screening_check_candidate",
        "build_screening_decision_flow": "decision_flow_candidate",
    }.get(tool_name, "agentic_skill_candidate")


def find_prior_review_decisions(
    *,
    candidate_type: str,
    claim: str,
    decisions: tuple[str, ...] = ("rejected", "held"),
    limit: int = 3,
    review_inbox_path: Path = REVIEW_INBOX_PATH,
    review_log_path: Path = REVIEW_LOG_PATH,
) -> list[dict[str, Any]]:
    """似た主張を持つ過去のrejected/held決定を探す（読み取り専用）。

    review_decisions.jsonlは追記のみで書き換えない不変ログなので、ここでは
    「知見」として参照するだけで候補の作成・採否そのものは変えない
    （WikiSkill論文のWiki層＝ロールバックされない知識、に相当する参照）。
    """
    claim_terms = _term_tokens(claim)
    if not claim_terms:
        return []
    inbox_by_id = {
        str(row.get("id") or ""): row
        for row in load_jsonl(review_inbox_path, newest_first=False)
        if row.get("id")
    }
    scored: list[tuple[int, dict[str, Any]]] = []
    for review in load_jsonl(review_log_path, newest_first=False):
        decision = _text(review.get("decision"))
        if decision not in decisions:
            continue
        source = inbox_by_id.get(_text(review.get("inbox_id")), {})
        if candidate_type and _text(source.get("candidate_type")) != candidate_type:
            continue
        past_claim = _text(source.get("claim"), maximum=500)
        overlap = len(claim_terms & _term_tokens(past_claim))
        if overlap == 0:
            continue
        scored.append((overlap, {
            "inbox_id": review.get("inbox_id", ""),
            "decision": decision,
            "note": _text(review.get("note"), maximum=300),
            "reviewed_at": review.get("reviewed_at", ""),
            "past_claim": past_claim,
        }))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    max_items = max(1, min(int(limit or 3), 8))
    return [item for _, item in scored[:max_items]]


def record_agentic_skill_result(
    *,
    tool_name: str,
    result: Any,
    session_id: str = "",
    case_context: dict[str, Any] | None = None,
    usage_log_path: Path = USAGE_LOG_PATH,
    review_inbox_path: Path = REVIEW_INBOX_PATH,
    review_log_path: Path = REVIEW_LOG_PATH,
) -> dict[str, Any] | None:
    """Record an agentic skill result and mirror reviewable items to the inbox."""
    if tool_name not in AGENTIC_SKILL_TOOL_NAMES:
        return None
    payload = _safe_payload(_unwrap_tool_result(_tool_response_payload(result)), max_chars=5000)
    event = {
        "id": _short_hash(["result", tool_name, session_id, payload, _now_iso()]),
        "event_type": "agentic_skill_result",
        "tool_name": tool_name,
        "session_id": session_id,
        "case_context": _safe_payload(case_context or {}, max_chars=1800),
        "result": payload,
        "recorded_at": _now_iso(),
        "guardrail": "usage_log_only_no_side_effect",
    }
    if tool_name in REVIEWABLE_TOOL_NAMES:
        claim = _review_claim(tool_name, payload)
        if claim:
            candidate_type = _review_type(tool_name)
            prior_review_context = find_prior_review_decisions(
                candidate_type=candidate_type,
                claim=claim,
                review_inbox_path=review_inbox_path,
                review_log_path=review_log_path,
            )
            inbox_item = {
                "id": _short_hash(["review", tool_name, session_id, claim, payload]),
                "source_event_id": event["id"],
                "tool_name": tool_name,
                "candidate_type": candidate_type,
                "claim": claim,
                "payload": payload,
                "case_context": _safe_payload(case_context or {}, max_chars=1800),
                "status": "candidate",
                "review_decision": "",
                "review_note": "",
                "prior_review_context": prior_review_context,
                "created_at": event["recorded_at"],
                "promotion_policy": "human_review_required_no_automatic_scoring_or_rag_change",
            }
            append_jsonl(review_inbox_path, inbox_item)
            event["review_inbox_id"] = inbox_item["id"]
    append_jsonl(usage_log_path, event)
    return event


def agentic_skill_usage_summary(limit: int = 50, *, usage_log_path: Path = USAGE_LOG_PATH) -> dict[str, Any]:
    rows = load_jsonl(usage_log_path, limit=max(1, min(int(limit or 50), 300)), newest_first=True)
    by_tool: dict[str, int] = {}
    for row in rows:
        name = str(row.get("tool_name") or "")
        if name:
            by_tool[name] = by_tool.get(name, 0) + 1
    return {
        "count": len(rows),
        "by_tool": by_tool,
        "events": rows,
        "guardrail": "read_only_summary",
    }


def check_agentic_skill_flow(
    *,
    usage_log_path: Path = USAGE_LOG_PATH,
    review_inbox_path: Path = REVIEW_INBOX_PATH,
    review_log_path: Path = REVIEW_LOG_PATH,
) -> dict[str, Any]:
    """Check the end-to-end observation flow for agentic skill usage."""
    usage_rows = load_jsonl(usage_log_path, limit=None, newest_first=False)
    inbox_rows = load_jsonl(review_inbox_path, limit=None, newest_first=False)
    review_rows = load_jsonl(review_log_path, limit=None, newest_first=False)
    usage_ids = {str(row.get("id") or "") for row in usage_rows if row.get("id")}
    inbox_ids = {str(row.get("id") or "") for row in inbox_rows if row.get("id")}
    reviewed_ids = {str(row.get("inbox_id") or "") for row in review_rows if row.get("inbox_id")}
    result_rows = [row for row in usage_rows if row.get("event_type") == "agentic_skill_result"]
    reviewable_results = [row for row in result_rows if row.get("tool_name") in REVIEWABLE_TOOL_NAMES]
    linked_results = [row for row in reviewable_results if row.get("review_inbox_id")]
    orphan_inbox = [
        row for row in inbox_rows
        if str(row.get("source_event_id") or "") and str(row.get("source_event_id") or "") not in usage_ids
    ]
    dangling_reviews = [row for row in review_rows if str(row.get("inbox_id") or "") not in inbox_ids]
    open_inbox = [row for row in inbox_rows if str(row.get("id") or "") not in reviewed_ids]

    checks = [
        {
            "name": "usage_log",
            "status": "ok" if usage_log_path.exists() else "empty",
            "message": f"{len(usage_rows)} usage events",
        },
        {
            "name": "review_inbox",
            "status": "ok" if review_inbox_path.exists() else "empty",
            "message": f"{len(inbox_rows)} review candidates",
        },
        {
            "name": "review_decisions",
            "status": "ok" if review_log_path.exists() else "empty",
            "message": f"{len(review_rows)} review decisions",
        },
        {
            "name": "usage_to_inbox_link",
            "status": "ok" if len(linked_results) == len(reviewable_results) else "warn",
            "message": f"{len(linked_results)}/{len(reviewable_results)} reviewable results linked",
        },
        {
            "name": "orphan_inbox",
            "status": "ok" if not orphan_inbox else "warn",
            "message": f"{len(orphan_inbox)} inbox items reference missing usage events",
        },
        {
            "name": "dangling_reviews",
            "status": "ok" if not dangling_reviews else "warn",
            "message": f"{len(dangling_reviews)} reviews reference missing inbox items",
        },
    ]
    status = "ok"
    if any(check["status"] == "warn" for check in checks):
        status = "warn"
    if not usage_rows and not inbox_rows and not review_rows:
        status = "empty"
    return {
        "status": status,
        "checks": checks,
        "summary": {
            "usage_events": len(usage_rows),
            "result_events": len(result_rows),
            "reviewable_results": len(reviewable_results),
            "linked_reviewable_results": len(linked_results),
            "inbox_items": len(inbox_rows),
            "open_inbox_items": len(open_inbox),
            "review_decisions": len(review_rows),
        },
        "latest": {
            "usage_event": usage_rows[-1] if usage_rows else None,
            "inbox_item": inbox_rows[-1] if inbox_rows else None,
            "review_decision": review_rows[-1] if review_rows else None,
        },
        "guardrail": "flow_check_only_no_pipeline_no_scoring_no_rag_no_obsidian_write",
    }


def _latest_reviews(review_log_path: Path = REVIEW_LOG_PATH) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(review_log_path, newest_first=False):
        inbox_id = _text(row.get("inbox_id"))
        if inbox_id:
            latest[inbox_id] = row
    return latest


def load_agentic_skill_review_inbox(
    *,
    limit: int = 50,
    status: str | None = None,
    review_inbox_path: Path = REVIEW_INBOX_PATH,
    review_log_path: Path = REVIEW_LOG_PATH,
) -> list[dict[str, Any]]:
    rows = load_jsonl(review_inbox_path, limit=None, newest_first=True)
    latest = _latest_reviews(review_log_path)
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        review = latest.get(str(item.get("id") or ""))
        if review:
            item["review"] = review
            item["review_decision"] = review.get("decision", "")
            item["review_note"] = review.get("note", "")
            item["status"] = review.get("decision", item.get("status", "candidate"))
        if status and item.get("status") != status:
            continue
        out.append(item)
        if len(out) >= max(1, min(int(limit or 50), 200)):
            break
    return out


def review_agentic_skill_inbox_item(
    *,
    inbox_id: str,
    decision: str,
    note: str = "",
    edited_claim: str = "",
    reviewer: str = "user",
    review_inbox_path: Path = REVIEW_INBOX_PATH,
    review_log_path: Path = REVIEW_LOG_PATH,
) -> dict[str, Any]:
    clean_id = _text(inbox_id)
    clean_decision = _text(decision)
    if clean_decision not in VALID_REVIEW_DECISIONS:
        raise ValueError("decision must be adopted, revised, held, or rejected")
    source = next((item for item in load_jsonl(review_inbox_path, newest_first=False) if item.get("id") == clean_id), None)
    if not source:
        raise ValueError("inbox item not found")
    review = {
        "inbox_id": clean_id,
        "source_event_id": source.get("source_event_id", ""),
        "tool_name": source.get("tool_name", ""),
        "candidate_type": source.get("candidate_type", ""),
        "decision": clean_decision,
        "note": _text(note),
        "edited_claim": _text(edited_claim),
        "reviewer": _text(reviewer) or "user",
        "reviewed_at": _now_iso(),
        "policy": "human_review_only_no_automatic_scoring_or_rag_change",
    }
    append_jsonl(review_log_path, review)
    return {"status": "recorded", "review": review}


def public_usage_notice(tool_name: str, event: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a small SSE-friendly notice for visible agentic skill use."""
    if tool_name not in AGENTIC_SKILL_TOOL_NAMES:
        return None
    label = {
        "structure_judgment_asset_candidate": "判断資産候補を作成",
        "validate_lease_source_summary": "情報源を検証",
        "convert_research_to_screening_insights": "調査を審査確認点へ変換",
        "build_screening_decision_flow": "判断分岐を整理",
        "write_scqa_report": "SCQAで説明を整理",
        "inspect_agentic_skill_flow": "一連の流れを確認",
        "propose_agentic_skill_next_actions": "次の確認を提案",
        "search_shion_system_context": "紫苑システムを検索",
        "propose_shion_system_improvement_focus": "改善焦点を提案",
    }.get(tool_name, tool_name)
    return {
        "type": "agentic_skill_usage",
        "tool": tool_name,
        "label": label,
        "review_inbox_id": (event or {}).get("review_inbox_id", ""),
        "logged": bool(event),
    }
