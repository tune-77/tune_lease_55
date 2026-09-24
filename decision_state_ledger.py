"""Append-only decision state events for Shion's screening workflow.

The ledger is an observation-only sidecar. Recording failures never change the
result of the screening operation that produced the event, and ledger events do
not alter scoring, prompts, or judgment-asset promotion.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LEDGER_PATH = REPO_ROOT / "data" / "judgment_state_events.jsonl"

EVENT_ACTIONS: dict[str, set[str]] = {
    "case_context_changed": {"created", "updated", "corrected"},
    "assumption_changed": {"created", "strengthened", "weakened", "invalidated"},
    "decision_changed": {"proposed", "revised", "finalized", "reopened"},
    "commitment_changed": {"opened", "assigned", "completed", "cancelled", "overdue"},
    "outcome_recorded": {"observed", "corrected"},
    "judgment_asset_evaluated": {"used", "helped", "challenged", "rejected", "neutral"},
    "event_corrected": {"invalidated"},
}
ACTOR_TYPES = {"human", "shion", "system", "external_source"}
VISIBILITY_LEVELS = {"private", "screening_team", "sales_and_screening", "demo_safe"}
QUOTE_POLICIES = {"allowed", "summary_only", "aggregate_only", "no_reuse"}
DECISION_CODES = {
    "承認": "approve",
    "条件付": "conditional",
    "条件付き": "conditional",
    "要審議": "conditional",
    "否決": "reject",
    "否認": "reject",
    "保留": "hold",
}
_PII_KEYS = {
    "name",
    "company_name",
    "address",
    "phone",
    "email",
    "representative",
    "hojin_name",
    "asset_location",
}
_APPEND_LOCK = threading.Lock()
_WRITEBACK_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="judgment-state-writeback")


class DecisionStateEventError(ValueError):
    """Raised when an event violates the stable ledger contract."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_structured_pii(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if str(key).lower() in _PII_KEYS else _redact_structured_pii(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_structured_pii(item) for item in value]
    return value


def _clean_text(value: Any, *, limit: int = 1200) -> str:
    return " ".join(str(value or "").split())[:limit]


def canonical_decision(value: str) -> str:
    text = str(value or "").strip().replace("条件付き承認", "条件付").replace("条件付承認", "条件付")
    for label, code in DECISION_CODES.items():
        if label in text:
            return code
    return text.lower()


def stable_idempotency_key(*parts: Any) -> str:
    raw = "\n".join(str(part or "").strip() for part in parts)
    return "jsekey_" + hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:24]


def build_event(
    *,
    event_type: str,
    aggregate_type: str,
    aggregate_id: str,
    actor_type: str,
    actor_role: str,
    surface: str,
    transition_subject: str,
    transition_action: str,
    reason_summary: str,
    evidence_refs: list[dict[str, Any]],
    transition_from: Any = None,
    transition_to: Any = None,
    actor_id: str = "",
    occurred_at: str = "",
    idempotency_key: str = "",
    fragile_assumptions: list[str] | None = None,
    alternatives_rejected: list[dict[str, str]] | None = None,
    dissent: list[str] | None = None,
    constraints: list[str] | None = None,
    caused_by: list[str] | None = None,
    supersedes: list[str] | None = None,
    correlation_id: str = "",
    visibility: str = "screening_team",
    contains_pii: bool = False,
    human_review_required: bool = True,
    retention_class: str = "decision_audit",
    payload: dict[str, Any] | None = None,
    event_id: str = "",
    recorded_at: str = "",
) -> dict[str, Any]:
    """Build and validate one normalized immutable event."""
    recorded = recorded_at or _now_iso()
    occurred = occurred_at or recorded
    generated_id = event_id or "jse_" + uuid4().hex
    key = idempotency_key or stable_idempotency_key(
        aggregate_type,
        aggregate_id,
        event_type,
        transition_action,
        correlation_id,
        occurred,
    )
    event = {
        "schema_version": "1.0",
        "event_id": generated_id,
        "idempotency_key": key,
        "occurred_at": occurred,
        "recorded_at": recorded,
        "event_type": str(event_type or "").strip(),
        "aggregate": {
            "type": str(aggregate_type or "").strip(),
            "id": str(aggregate_id or "").strip(),
        },
        "actor": {
            "type": str(actor_type or "").strip(),
            "role": str(actor_role or "").strip(),
            "id": str(actor_id or "").strip(),
        },
        "surface": str(surface or "unknown").strip() or "unknown",
        "transition": {
            "subject": str(transition_subject or "").strip(),
            "action": str(transition_action or "").strip(),
            "from": transition_from,
            "to": transition_to,
        },
        "context": {
            "reason_summary": _clean_text(reason_summary),
            "fragile_assumptions": [_clean_text(item, limit=400) for item in fragile_assumptions or [] if _clean_text(item)],
            "alternatives_rejected": _redact_structured_pii(alternatives_rejected or []),
            "dissent": [_clean_text(item, limit=400) for item in dissent or [] if _clean_text(item)],
            "constraints": [_clean_text(item, limit=400) for item in constraints or [] if _clean_text(item)],
        },
        "relations": {
            "caused_by": [str(item) for item in caused_by or [] if str(item).strip()],
            "supersedes": [str(item) for item in supersedes or [] if str(item).strip()],
            "correlation_id": str(correlation_id or "").strip(),
        },
        "evidence_refs": _redact_structured_pii(evidence_refs or []),
        "governance": {
            "visibility": visibility,
            "contains_pii": bool(contains_pii),
            "human_review_required": bool(human_review_required),
            "retention_class": str(retention_class or "decision_audit"),
        },
        "payload": _redact_structured_pii(payload or {}),
    }
    validate_event(event)
    return event


def validate_event(event: dict[str, Any]) -> None:
    event_type = str(event.get("event_type") or "")
    if event_type not in EVENT_ACTIONS:
        raise DecisionStateEventError(f"unsupported event_type: {event_type}")
    event_id = str(event.get("event_id") or "")
    if not event_id.startswith("jse_"):
        raise DecisionStateEventError("event_id must start with jse_")
    if not str(event.get("idempotency_key") or "").strip():
        raise DecisionStateEventError("idempotency_key is required")
    aggregate = event.get("aggregate") or {}
    if not str(aggregate.get("type") or "").strip() or not str(aggregate.get("id") or "").strip():
        raise DecisionStateEventError("aggregate type and id are required")
    actor = event.get("actor") or {}
    actor_type = str(actor.get("type") or "")
    if actor_type not in ACTOR_TYPES:
        raise DecisionStateEventError(f"unsupported actor type: {actor_type}")
    transition = event.get("transition") or {}
    action = str(transition.get("action") or "")
    if action not in EVENT_ACTIONS[event_type]:
        raise DecisionStateEventError(f"unsupported action for {event_type}: {action}")
    if not str(transition.get("subject") or "").strip():
        raise DecisionStateEventError("transition subject is required")
    governance = event.get("governance") or {}
    if governance.get("visibility") not in VISIBILITY_LEVELS:
        raise DecisionStateEventError("unsupported visibility")
    evidence_refs = event.get("evidence_refs") or []
    if not isinstance(evidence_refs, list) or not evidence_refs:
        raise DecisionStateEventError("at least one evidence_ref is required")
    for ref in evidence_refs:
        if not isinstance(ref, dict) or not str(ref.get("kind") or "").strip():
            raise DecisionStateEventError("each evidence_ref requires kind")
        if str(ref.get("quote_policy") or "") not in QUOTE_POLICIES:
            raise DecisionStateEventError("unsupported evidence quote_policy")
    if event_type == "decision_changed":
        if not str((event.get("context") or {}).get("reason_summary") or "").strip():
            raise DecisionStateEventError("decision_changed requires reason_summary")
        if action == "finalized" and actor_type != "human":
            raise DecisionStateEventError("only a human actor can finalize a decision")
    if event_type == "event_corrected" and not (event.get("relations") or {}).get("supersedes"):
        raise DecisionStateEventError("event_corrected requires a superseded event id")


def load_events(path: Path = DEFAULT_LEDGER_PATH) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            validate_event(event)
        except (json.JSONDecodeError, DecisionStateEventError, TypeError):
            continue
        events.append(event)
    return events


def append_event(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    """Append one event, deduplicating by event id and idempotency key."""
    validate_event(event)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _APPEND_LOCK:
        existing = load_events(path)
        event_id = str(event["event_id"])
        key = str(event["idempotency_key"])
        for row in existing:
            if str(row.get("event_id")) == event_id or str(row.get("idempotency_key")) == key:
                return {
                    "recorded": False,
                    "duplicate": True,
                    "event_id": str(row.get("event_id") or event_id),
                    "event": row,
                }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return {"recorded": True, "duplicate": False, "event_id": event_id, "event": event}


def safe_append_event(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    """Best-effort append for business flows that must not fail with the ledger."""
    try:
        result = {"ok": True, **append_event(path, event)}
        result["cloudrun_writeback_queued"] = _queue_cloudrun_writeback(event) if result.get("recorded") else False
        return result
    except Exception as exc:
        _record_append_failure(path, event, exc)
        return {
            "ok": False,
            "recorded": False,
            "duplicate": False,
            "event_id": str(event.get("event_id") or ""),
            "error": str(exc),
        }


def _queue_cloudrun_writeback(event: dict[str, Any]) -> bool:
    """Persist a normalized copy off-container without delaying the caller."""
    if not os.environ.get("K_SERVICE", "").strip():
        return False

    def deliver() -> None:
        try:
            from api.cloudrun_writeback import record_cloudrun_input_event

            record_cloudrun_input_event(
                event_type="judgment_state_event",
                surface=str(event.get("surface") or "decision_state_ledger"),
                payload=event,
            )
        except Exception:
            pass

    _WRITEBACK_EXECUTOR.submit(deliver)
    return True


def _record_append_failure(path: Path, event: dict[str, Any], exc: Exception) -> None:
    try:
        failure_path = Path(path).parent / "judgment_state_event_failures.jsonl"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "recorded_at": _now_iso(),
            "event_id": str(event.get("event_id") or ""),
            "idempotency_key": str(event.get("idempotency_key") or ""),
            "error": str(exc)[:500],
        }
        with failure_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass


def build_decision_changed_event(
    *,
    case_id: str,
    before: str,
    after: str,
    reason: str,
    source: str,
    record_id: int | str,
    score: float | None = None,
    action: str = "revised",
    occurred_at: str = "",
) -> dict[str, Any]:
    correlation_id = f"judgment_feedback:{record_id}"
    return build_event(
        event_type="decision_changed",
        aggregate_type="case",
        aggregate_id=case_id,
        actor_type="human",
        actor_role="screening_officer",
        surface=source or "judgment_feedback",
        transition_subject="decision",
        transition_action=action,
        transition_from={"stance": canonical_decision(before), "label": before},
        transition_to={"stance": canonical_decision(after), "label": after},
        reason_summary=reason,
        evidence_refs=[
            {
                "kind": "judgment_feedback",
                "id": str(record_id),
                "locator": f"db://judgment_feedback/{record_id}",
                "quote_policy": "summary_only",
            }
        ],
        occurred_at=occurred_at,
        correlation_id=correlation_id,
        idempotency_key=stable_idempotency_key(correlation_id, "decision_changed", action),
        payload={"score": score} if score is not None else {},
    )


def build_judgment_asset_evaluated_event(entry: dict[str, Any]) -> dict[str, Any]:
    rule_id = str(entry.get("rule_id") or "").strip()
    outcome = str(entry.get("outcome") or "").strip().lower()
    case_id = str(entry.get("case_id") or "").strip() or f"unscoped:{rule_id}"
    used_at = str(entry.get("used_at") or "").strip()
    source = str(entry.get("source") or "manual").strip() or "manual"
    reason = str(entry.get("note") or "").strip() or f"判断資産を{outcome}と評価"
    correlation_id = "judgment_asset_feedback:" + stable_idempotency_key(
        rule_id, outcome, case_id, used_at, source
    )
    return build_event(
        event_type="judgment_asset_evaluated",
        aggregate_type="case",
        aggregate_id=case_id,
        actor_type="human",
        actor_role="screening_reviewer",
        surface=source,
        transition_subject=f"judgment_asset:{rule_id}",
        transition_action=outcome,
        transition_from=None,
        transition_to={"outcome": outcome},
        reason_summary=reason,
        evidence_refs=[
            {
                "kind": "judgment_asset",
                "id": rule_id,
                "locator": "data/canonical_judgment_rules.json",
                "quote_policy": "allowed",
            }
        ],
        occurred_at=used_at,
        correlation_id=correlation_id,
        idempotency_key=stable_idempotency_key(correlation_id),
        payload={"rule_id": rule_id, "review_id": entry.get("review_id")},
    )


def build_outcome_recorded_event(
    *,
    case_id: str,
    outcome: str,
    reason: str,
    source: str,
    record_id: int | str,
    occurred_at: str = "",
) -> dict[str, Any]:
    correlation_id = f"judgment_feedback:{record_id}"
    return build_event(
        event_type="outcome_recorded",
        aggregate_type="case",
        aggregate_id=case_id,
        actor_type="system",
        actor_role="outcome_registrar",
        surface=source or "outcome_registration",
        transition_subject="case_outcome",
        transition_action="observed",
        transition_from=None,
        transition_to={"status": canonical_decision(outcome), "label": outcome},
        reason_summary=reason,
        evidence_refs=[
            {
                "kind": "judgment_feedback",
                "id": str(record_id),
                "locator": f"db://judgment_feedback/{record_id}",
                "quote_policy": "summary_only",
            }
        ],
        occurred_at=occurred_at,
        correlation_id=correlation_id,
        idempotency_key=stable_idempotency_key(correlation_id, "outcome_recorded"),
        human_review_required=True,
    )
def audit_ledger(path: Path = DEFAULT_LEDGER_PATH) -> dict[str, Any]:
    """Return non-mutating structural health metrics for a JSONL ledger."""
    summary: dict[str, Any] = {
        "path": str(path),
        "total_lines": 0,
        "valid_events": 0,
        "invalid_lines": 0,
        "duplicate_event_ids": 0,
        "duplicate_idempotency_keys": 0,
        "orphan_corrections": 0,
        "event_type_counts": {},
    }
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return summary
    ids: set[str] = set()
    keys: set[str] = set()
    corrections: list[str] = []
    counts: dict[str, int] = {}
    for line in lines:
        if not line.strip():
            continue
        summary["total_lines"] += 1
        try:
            event = json.loads(line)
            validate_event(event)
        except (json.JSONDecodeError, DecisionStateEventError, TypeError):
            summary["invalid_lines"] += 1
            continue
        summary["valid_events"] += 1
        event_id = str(event.get("event_id") or "")
        key = str(event.get("idempotency_key") or "")
        if event_id in ids:
            summary["duplicate_event_ids"] += 1
        if key in keys:
            summary["duplicate_idempotency_keys"] += 1
        ids.add(event_id)
        keys.add(key)
        event_type = str(event.get("event_type") or "")
        counts[event_type] = counts.get(event_type, 0) + 1
        if event_type == "event_corrected":
            corrections.extend(str(item) for item in (event.get("relations") or {}).get("supersedes") or [])
    summary["orphan_corrections"] = sum(1 for target in corrections if target not in ids)
    summary["event_type_counts"] = dict(sorted(counts.items()))
    return summary


def deduplicate_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the first valid event for each idempotency key."""
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    for event in events:
        validate_event(event)
        event_id = str(event["event_id"])
        key = str(event["idempotency_key"])
        if event_id in seen_ids or key in seen_keys:
            continue
        seen_ids.add(event_id)
        seen_keys.add(key)
        result.append(event)
    return result
