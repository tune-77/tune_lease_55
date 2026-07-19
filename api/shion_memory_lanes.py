"""Separated memory lanes for ordinary Shion continuity.

This module builds a small read-only summary of Shion's memory surfaces without
mixing them into judgment assets. It is meant for UI/status display and later
prompt routing, not for automatic promotion.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from runtime_paths import get_data_path

from api.shion_tasks import TASK_LOG_PATH, list_tasks


PERSONAL_MEMORY_PATH = Path(get_data_path("user_personal_memory.md"))
SHION_MEMORY_INDEX_PATH = Path(get_data_path("shion_memory_index.json"))

LANE_POLICY = {
    "personal_memory": "direct_conversation_continuity_only",
    "task_memory": "what_to_do_not_why_it_matters",
    "dialogue_memory": "continuity_and_preferences_not_business_evidence",
    "judgment_memory": "business_decision_support_after_review",
    "value_memory": "safety_and_human_judgment_boundary",
    "technical_memory": "operations_context_only",
}


def _clean_text(value: Any, *, limit: int = 220) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:limit].rstrip()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_personal_memory(path: Path, *, include_sensitive: bool = False, limit: int = 6) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    memories: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    section = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            section = stripped.removeprefix("## ").strip()
            continue
        if section not in {"Personal Facts", "Captured Personal Memories"}:
            continue
        if not stripped.startswith("- ["):
            continue
        if "[sensitive" in stripped and not include_sensitive:
            continue
        if "[candidate" in stripped:
            continue
        memories.append(
            {
                "kind": "personal_fact" if section == "Personal Facts" else "captured_personal_memory",
                "section": section,
                "text": _clean_text(stripped.lstrip("- "), limit=180),
                "source_path": str(path),
            }
        )
        if len(memories) >= limit:
            break
    return memories


def _memory_records(index_path: Path) -> list[dict[str, Any]]:
    data = _read_json(index_path)
    records = data.get("records") if isinstance(data.get("records"), list) else []
    return [record for record in records if isinstance(record, dict)]


def _record_allowed(record: dict[str, Any], *, include_private: bool = False) -> bool:
    if not include_private and (record.get("private") is True or record.get("status") == "private"):
        return False
    if include_private and (record.get("private") is True or record.get("status") == "private"):
        return True
    return str(record.get("status") or "active") in {"active", "stale", "revised"}


def _summarize_memory_index(index_path: Path, *, include_private: bool = False, sample_limit: int = 5) -> dict[str, Any]:
    records = [record for record in _memory_records(index_path) if _record_allowed(record, include_private=include_private)]
    by_type = Counter(str(record.get("memory_type") or "unknown") for record in records)
    by_status = Counter(str(record.get("status") or "active") for record in records)
    samples_by_type: dict[str, list[dict[str, Any]]] = {}
    priority_types = ("value_memory", "judgment_memory", "dialogue_memory", "factual_memory", "technical_memory")
    for memory_type in priority_types:
        selected = [
            {
                "id": record.get("id"),
                "status": record.get("status", "active"),
                "content": _clean_text(record.get("content"), limit=180),
                "source_path": record.get("source_path", ""),
            }
            for record in records
            if record.get("memory_type") == memory_type and record.get("content")
        ][:sample_limit]
        if selected:
            samples_by_type[memory_type] = selected
    return {
        "count": len(records),
        "by_type": dict(sorted(by_type.items())),
        "by_status": dict(sorted(by_status.items())),
        "samples_by_type": samples_by_type,
    }


def build_memory_lanes(
    *,
    include_private: bool = False,
    include_sensitive_personal: bool = False,
    sample_limit: int = 5,
    personal_path: Path | None = None,
    index_path: Path | None = None,
    task_path: Path | None = None,
) -> dict[str, Any]:
    """Return a compact separated-memory summary for Shion."""
    personal = _read_personal_memory(
        personal_path or PERSONAL_MEMORY_PATH,
        include_sensitive=include_sensitive_personal,
        limit=sample_limit,
    )
    tasks = list_tasks(status="open", limit=sample_limit, path=task_path or TASK_LOG_PATH)
    memory_index = _summarize_memory_index(
        index_path or SHION_MEMORY_INDEX_PATH,
        include_private=include_private,
        sample_limit=sample_limit,
    )
    return {
        "schema_version": 1,
        "mode": "separated_memory_lanes",
        "policy": LANE_POLICY,
        "guardrail": "ordinary_memory_does_not_automatically_become_judgment_asset",
        "lanes": {
            "personal_memory": {
                "count": len(personal),
                "use": LANE_POLICY["personal_memory"],
                "items": personal,
            },
            "task_memory": {
                "count": len(tasks),
                "use": LANE_POLICY["task_memory"],
                "items": tasks,
            },
            "indexed_memory": {
                "use": "route_dependent_recall_only",
                **memory_index,
            },
        },
    }
