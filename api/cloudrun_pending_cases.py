"""Cloud Run pending-case normalization helpers."""

from __future__ import annotations

import json
from typing import Any


CLOUDRUN_SCORE_CASE_PREFIX = "cloudrun_score:"
CLOUDRUN_EVENT_CASE_PREFIX = "cloudrun_event:"


def parse_cloudrun_score_case_id(case_id: str) -> int | None:
    raw = str(case_id or "").strip()
    if not raw.startswith(CLOUDRUN_SCORE_CASE_PREFIX):
        return None
    try:
        score_id = int(raw.removeprefix(CLOUDRUN_SCORE_CASE_PREFIX))
    except ValueError:
        return None
    return score_id if score_id > 0 else None


def parse_cloudrun_event_case_id(case_id: str) -> str:
    raw = str(case_id or "").strip()
    if not raw.startswith(CLOUDRUN_EVENT_CASE_PREFIX):
        return ""
    event_id = raw.removeprefix(CLOUDRUN_EVENT_CASE_PREFIX).strip()
    return event_id[:120]


def loads_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def cloudrun_score_pending_item(row: dict[str, Any]) -> dict[str, Any]:
    inputs = loads_dict(row.get("inputs_json"))
    result = loads_dict(row.get("result_json"))
    created_at = str(row.get("created_at") or "")
    company_no = str(inputs.get("company_no") or inputs.get("customer_no") or "").strip()
    company_name = str(inputs.get("company_name") or inputs.get("customer_name") or "").strip()
    industry = str(
        row.get("industry_sub")
        or result.get("industry_sub")
        or inputs.get("industry_sub")
        or row.get("industry_major")
        or result.get("industry_major")
        or inputs.get("industry_major")
        or ""
    ).strip()
    score = row.get("score")
    if score in (None, ""):
        score = result.get("score", result.get("score_base"))
    return {
        "id": f"{CLOUDRUN_SCORE_CASE_PREFIX}{int(row.get('id') or 0)}",
        "company_no": company_no,
        "company_name": company_name or "Cloud Run審査入力",
        "timestamp": created_at,
        "score": score,
        "industry": industry,
        "registration_date": created_at[:10] if len(created_at) >= 10 else "",
        "estimate_sent_date": created_at[:10] if len(created_at) >= 10 else "",
        "final_result_date": "",
        "_source": "cloudrun_score_inputs",
        "cloudrun_return_id": int(row.get("id") or 0),
        "cloudrun_event_id": str(row.get("event_id") or ""),
        "review_status": str(row.get("return_review_status") or "candidate"),
    }


def cloudrun_score_pending_item_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("event_type") not in {"score_calculated", "score_full_calculated"}:
        return None
    event_id = str(event.get("event_id") or "").strip()
    if not event_id:
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    if not inputs and not result:
        return None
    created_at = str(event.get("ts") or "")
    company_no = str(inputs.get("company_no") or inputs.get("customer_no") or "").strip()
    company_name = str(inputs.get("company_name") or inputs.get("customer_name") or "").strip()
    industry = str(
        result.get("industry_sub")
        or inputs.get("industry_sub")
        or result.get("industry_major")
        or inputs.get("industry_major")
        or ""
    ).strip()
    return {
        "id": f"{CLOUDRUN_EVENT_CASE_PREFIX}{event_id}",
        "company_no": company_no,
        "company_name": company_name or "Cloud Run審査入力",
        "timestamp": created_at,
        "score": result.get("score", result.get("score_base")),
        "industry": industry,
        "registration_date": created_at[:10] if len(created_at) >= 10 else "",
        "estimate_sent_date": created_at[:10] if len(created_at) >= 10 else "",
        "final_result_date": "",
        "_source": "cloudrun_gcs_input",
        "cloudrun_event_id": event_id,
        "review_status": "gcs_input",
    }


def list_cloudrun_score_pending_cases_from_events(events: list[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
    if not events:
        return []
    registered_event_ids: set[str] = set()
    for event in events:
        if event.get("event_type") not in {"case_result_registered", "cloudrun_pending_case_rejected"}:
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        for key in ("source_event_id", "cloudrun_event_id"):
            value = str(payload.get(key) or "").strip()
            if value:
                registered_event_ids.add(value)
        case_id = str(payload.get("case_id") or "").strip()
        if case_id.startswith(CLOUDRUN_EVENT_CASE_PREFIX):
            registered_event_ids.add(case_id.removeprefix(CLOUDRUN_EVENT_CASE_PREFIX))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in reversed(events):
        event_id = str(event.get("event_id") or "").strip()
        if not event_id or event_id in seen or event_id in registered_event_ids:
            continue
        item = cloudrun_score_pending_item_from_event(event)
        if not item:
            continue
        seen.add(event_id)
        rows.append(item)
        if len(rows) >= limit:
            break
    rows.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    return rows[:limit]
