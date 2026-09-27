"""Expose Jev's shadow memory judgment beside a separate human decision.

The Jev report remains read-only evidence. Human decisions are persisted in a
separate state file and never mutate, delete, or rerank the memory index.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from runtime_paths import get_data_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = get_data_dir()
REPORT_PATHS = (
    DATA_DIR / "jev_memory_review_shadow_latest.json",
    REPO_ROOT / ".cloudrun_bundle" / "data" / "jev_memory_review_shadow_latest.json",
    REPO_ROOT / "reports" / "jev_memory_review_shadow_latest.json",
)
STATE_PATH = DATA_DIR / "jev_memory_review_human_state.json"
AUDIT_PATH = DATA_DIR / "jev_memory_review_human_audit.jsonl"
GCS_STATE_OBJECT = os.environ.get(
    "JEV_MEMORY_REVIEW_GCS_STATE_OBJECT",
    "cloudrun-state/jev_memory_review_human_state.json",
).strip("/")

Decision = Literal["retain", "revise", "archive_candidate", "held"]
ALLOWED_DECISIONS = {"retain", "revise", "archive_candidate", "held"}
_STATE_LOCK = threading.RLock()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _report_path() -> Path | None:
    return next((path for path in REPORT_PATHS if path.exists()), None)


def _cloudrun_state_enabled() -> bool:
    return bool(os.environ.get("K_SERVICE", "").strip())


def _read_gcs_state() -> dict[str, Any]:
    from google.api_core.exceptions import NotFound  # type: ignore[import-untyped]
    from google.cloud import storage  # type: ignore[import-untyped]
    from api.cloudrun_writeback import _bucket_name

    try:
        text = storage.Client().bucket(_bucket_name()).blob(GCS_STATE_OBJECT).download_as_text()
    except NotFound:
        return {}
    payload = json.loads(text)
    return payload if isinstance(payload, dict) else {}


def _write_gcs_state(state: dict[str, Any]) -> None:
    from google.cloud import storage  # type: ignore[import-untyped]
    from api.cloudrun_writeback import _bucket_name
    from scripts.gcs_lock import GCSLock

    bucket_name = _bucket_name()
    with GCSLock(bucket_name=bucket_name, target_file=GCS_STATE_OBJECT, ttl_seconds=30):
        blob = storage.Client().bucket(bucket_name).blob(GCS_STATE_OBJECT)
        blob.upload_from_string(
            json.dumps(state, ensure_ascii=False, indent=2),
            content_type="application/json; charset=utf-8",
        )


def load_state(path: Path | None = None) -> dict[str, Any]:
    if path is None and _cloudrun_state_enabled():
        payload = _read_gcs_state()
    else:
        payload = _read_json(path or STATE_PATH)
    reviews = payload.get("reviews")
    if not isinstance(reviews, dict):
        reviews = {}
    return {"schema_version": 1, **payload, "reviews": reviews}


def _save_state(state: dict[str, Any], path: Path | None = None) -> None:
    target = path or STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    state["schema_version"] = 1
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    fd, temp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(state, ensure_ascii=False, indent=2))
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    if path is None and _cloudrun_state_enabled():
        _write_gcs_state(state)


def _number_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            result[str(key)] = round(float(raw), 3)
    return result


def _review_item(row: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    projected = row.get("projected") if isinstance(row.get("projected"), dict) else {}
    tags = projected.get("semantic_tags") if isinstance(projected.get("semantic_tags"), list) else []
    return {
        "memory_id": str(row.get("memory_id") or ""),
        "shadow_rank": int(row.get("shadow_rank") or 0),
        "content_preview": str(row.get("content_preview") or "")[:500],
        "used_count": int(row.get("used_count") or 0),
        "composite_priority": round(float(row.get("composite_priority") or 0.0), 3),
        "shadow_route": str(row.get("shadow_route") or ""),
        "scores": _number_map(row.get("scores")),
        "confidence": _number_map(row.get("confidence")),
        "semantic_tags": [str(tag) for tag in tags[:4]],
        "human_decision": str(review.get("decision") or "pending"),
        "human_note": str(review.get("note") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
    }


def _report_rows() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report_path = _report_path()
    if report_path is None:
        return {}, []
    report = _read_json(report_path)
    rows = [row for row in report.get("results") or [] if isinstance(row, dict)]
    return report, rows


def get_review_queue() -> dict[str, Any]:
    report, rows = _report_rows()
    if not report:
        return {
            "available": False,
            "status": "report_missing",
            "items": [],
            "summary": {"total": 0, "pending": 0, "by_decision": {}},
            "guardrail": "shadow_only_human_decision_state_no_memory_mutation",
        }
    state = load_state()
    reviews = state["reviews"]
    items = [
        _review_item(row, reviews.get(str(row.get("memory_id") or ""), {}))
        for row in rows
        if str(row.get("memory_id") or "")
        and not reviews.get(str(row.get("memory_id") or ""), {}).get("deleted_at")
    ]
    decisions = Counter(item["human_decision"] for item in items)
    return {
        "available": True,
        "status": str(report.get("status") or "unknown"),
        "generated_at": str(report.get("generated_at") or ""),
        "model": str(report.get("model") or ""),
        "confidence_threshold": float(report.get("confidence_threshold") or 0.0),
        "summary": {
            "total": len(items),
            "pending": decisions.get("pending", 0),
            "by_decision": dict(sorted(decisions.items())),
        },
        "items": items,
        "guardrail": "shadow_only_human_decision_state_no_memory_mutation",
    }


def save_human_decision(memory_id: str, *, decision: Decision, note: str = "") -> dict[str, Any]:
    clean_id = memory_id.strip()
    if decision not in ALLOWED_DECISIONS:
        raise ValueError("invalid decision")
    queue = get_review_queue()
    candidates = {str(item["memory_id"]): item for item in queue.get("items") or []}
    if clean_id not in candidates:
        raise KeyError(clean_id)
    now = datetime.now().isoformat(timespec="seconds")
    review = {"decision": decision, "note": note.strip()[:1000], "reviewed_at": now}
    with _STATE_LOCK:
        state = load_state()
        if state["reviews"].get(clean_id, {}).get("deleted_at"):
            raise KeyError(clean_id)
        state["reviews"][clean_id] = review
        _save_state(state)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": now, "memory_id": clean_id, **review}, ensure_ascii=False) + "\n")
    return {**candidates[clean_id], "human_decision": decision, "human_note": review["note"], "reviewed_at": now}


def delete_review_candidate(memory_id: str) -> dict[str, Any]:
    """Hide one review candidate without deleting the underlying memory/report row."""
    clean_id = memory_id.strip()
    _, rows = _report_rows()
    candidates = {
        str(row.get("memory_id") or ""): row
        for row in rows
        if str(row.get("memory_id") or "")
    }
    if clean_id not in candidates:
        raise KeyError(clean_id)

    now = datetime.now().isoformat(timespec="seconds")
    with _STATE_LOCK:
        state = load_state()
        previous = state["reviews"].get(clean_id, {})
        state["reviews"][clean_id] = {**previous, "deleted_at": now}
        _save_state(state)

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"ts": now, "memory_id": clean_id, "action": "delete_review_candidate"},
                ensure_ascii=False,
            )
            + "\n"
        )
    return {"memory_id": clean_id, "deleted_at": now, "memory_deleted": False}
