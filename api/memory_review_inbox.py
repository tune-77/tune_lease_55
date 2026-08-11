"""Read-only candidate inbox plus separate human review state for memory items."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from runtime_paths import get_data_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = get_data_dir()

REVIEW_STATE_PATH = DATA_DIR / "memory_review_inbox_state.json"
REVIEW_AUDIT_PATH = DATA_DIR / "memory_review_inbox_audit.jsonl"

SOURCE_PATHS: dict[str, Path] = {
    "judgment_materials_preview": DATA_DIR / "judgment_materials_preview.jsonl",
    "autoresearch_candidates": DATA_DIR / "autoresearch_judgment_asset_candidates.jsonl",
    "reflection_action_candidates": DATA_DIR / "reflection_action_candidates.jsonl",
    "prediction_error_candidates": DATA_DIR / "prediction_error_update_candidates.jsonl",
    "obsidian_memory_insight_candidates": DATA_DIR / "obsidian_memory_insight_candidates.jsonl",
}

Decision = Literal["adopted", "revised", "held", "rejected"]

TERMINAL_STATUSES = {"adopted", "revised", "held", "rejected"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_review_state(path: Path | None = None) -> dict[str, Any]:
    path = path or REVIEW_STATE_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "reviews": {}}
    if not isinstance(payload, dict):
        return {"schema_version": 1, "reviews": {}}
    reviews = payload.get("reviews")
    if not isinstance(reviews, dict):
        payload["reviews"] = {}
    return payload


def save_review_state(state: dict[str, Any], path: Path | None = None) -> None:
    path = path or REVIEW_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    state["schema_version"] = 1
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _source_item_id(row: dict[str, Any]) -> str:
    for key in ("id", "candidate_id", "canonical_key", "event_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    seed = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def make_inbox_id(source_name: str, row: dict[str, Any]) -> str:
    return f"{source_name}__{_source_item_id(row)}"


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def normalize_candidate(source_name: str, row: dict[str, Any], review: dict[str, Any] | None = None) -> dict[str, Any]:
    review = review or {}
    inbox_id = make_inbox_id(source_name, row)
    source_item_id = _source_item_id(row)
    claim = _first_text(row, ("claim", "canonical_statement", "proposed_change", "suggested_update", "body", "hypothesis"))
    title = _first_text(row, ("title", "research_title", "target_belief", "kind", "candidate_type"))
    evidence_path = _first_text(row, ("evidence_path", "source_path", "evidence"))
    if not evidence_path:
        paths = row.get("evidence_paths") or row.get("source_paths") or []
        if isinstance(paths, list) and paths:
            evidence_path = str(paths[0])
    status = str(review.get("status") or "candidate")
    edited_claim = str(review.get("edited_claim") or row.get("edited_claim") or "")
    return {
        "inbox_id": inbox_id,
        "source": source_name,
        "source_item_id": source_item_id,
        "status": status,
        "title": title or source_item_id,
        "claim": claim,
        "edited_claim": edited_claim,
        "note": str(review.get("note") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "candidate_type": _first_text(row, ("candidate_type", "material_type", "action_schema", "error_type")),
        "quality": _first_text(row, ("asset_quality", "quality", "priority", "confidence")),
        "topic": _first_text(row, ("research_topic", "concept", "target", "domain")),
        "date_hint": _first_text(row, ("research_date", "date", "target_date", "created_at", "generated_at")),
        "evidence_path": evidence_path,
        "raw_status": _first_text(row, ("status", "review_status", "promotion_status", "human_decision_status", "promote_status")),
        "source_path": str(SOURCE_PATHS.get(source_name, Path(""))),
    }


def load_candidates() -> list[dict[str, Any]]:
    state = load_review_state()
    reviews = state.get("reviews") if isinstance(state.get("reviews"), dict) else {}
    items: list[dict[str, Any]] = []
    for source_name, path in SOURCE_PATHS.items():
        for row in _read_jsonl(path):
            inbox_id = make_inbox_id(source_name, row)
            review = reviews.get(inbox_id) if isinstance(reviews.get(inbox_id), dict) else {}
            items.append(normalize_candidate(source_name, row, review))
    return items


def summarize_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = Counter(str(item.get("status") or "candidate") for item in items)
    by_source = Counter(str(item.get("source") or "unknown") for item in items)
    return {
        "total": len(items),
        "open": sum(1 for item in items if str(item.get("status") or "candidate") == "candidate"),
        "by_status": dict(sorted(by_status.items())),
        "by_source": dict(sorted(by_source.items())),
    }


def list_inbox(
    *,
    status: str = "candidate",
    source: str = "all",
    q: str = "",
    limit: int = 30,
    offset: int = 0,
) -> dict[str, Any]:
    items = load_candidates()
    filtered = items
    if status != "all":
        filtered = [item for item in filtered if str(item.get("status") or "candidate") == status]
    if source != "all":
        filtered = [item for item in filtered if str(item.get("source") or "") == source]
    query = q.strip().lower()
    if query:
        filtered = [
            item
            for item in filtered
            if query
            in " ".join(
                str(item.get(key) or "")
                for key in ("title", "claim", "topic", "candidate_type", "evidence_path")
            ).lower()
        ]
    filtered.sort(
        key=lambda item: (
            str(item.get("date_hint") or ""),
            str(item.get("source") or ""),
            str(item.get("source_item_id") or ""),
        ),
        reverse=True,
    )
    limit = max(1, min(int(limit or 30), 100))
    offset = max(0, int(offset or 0))
    return {
        "summary": summarize_items(items),
        "filtered_total": len(filtered),
        "limit": limit,
        "offset": offset,
        "items": filtered[offset : offset + limit],
        "review_policy": "state_only_no_source_mutation",
        "state_path": _display_path(REVIEW_STATE_PATH),
    }


def review_candidate(
    inbox_id: str,
    *,
    decision: Decision,
    note: str = "",
    edited_claim: str = "",
) -> dict[str, Any]:
    candidates = {str(item["inbox_id"]): item for item in load_candidates()}
    if inbox_id not in candidates:
        raise KeyError(inbox_id)
    if decision == "revised" and not edited_claim.strip():
        raise ValueError("edited_claim is required when decision is revised")
    now = datetime.now().isoformat(timespec="seconds")
    state = load_review_state()
    reviews = state.setdefault("reviews", {})
    current = {
        "status": decision,
        "note": note.strip()[:1000],
        "edited_claim": edited_claim.strip()[:4000],
        "reviewed_at": now,
        "source": candidates[inbox_id]["source"],
        "source_item_id": candidates[inbox_id]["source_item_id"],
    }
    reviews[inbox_id] = current
    save_review_state(state)
    REVIEW_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now, "inbox_id": inbox_id, **current}, ensure_ascii=False) + "\n")
    return {
        **candidates[inbox_id],
        "status": decision,
        "note": current["note"],
        "edited_claim": current["edited_claim"],
        "reviewed_at": now,
    }
