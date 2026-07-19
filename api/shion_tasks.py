"""Small append-only task ledger for Shion.

This is intentionally separate from Shion memory:

- memory: why a policy or preference matters
- task: what should be done, when, and whether it is done

The ledger is event-sourced JSONL so local and Cloud Run paths can merge or
replay without requiring a database migration.
"""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from runtime_paths import get_data_path


SCHEMA_VERSION = 1
TASK_LOG_PATH = Path(get_data_path("shion_tasks.jsonl"))
VALID_STATUSES = {"open", "done", "cancelled"}
_LOCK = threading.RLock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _task_id(title: str, created_at: str) -> str:
    raw = f"{title}|{created_at}|{uuid.uuid4().hex}"
    return "task_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _clean_text(value: Any, *, limit: int = 240) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    return text[:limit].rstrip()


def _normalize_due_at(value: str | None) -> str:
    raw = _clean_text(value, limit=80)
    if not raw:
        return ""
    # Keep date-only values valid and easy to display.
    if len(raw) == 10:
        datetime.fromisoformat(raw)
        return raw
    datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return raw


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _read_events(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or TASK_LOG_PATH
    if not target.exists():
        return []
    events: list[dict[str, Any]] = []
    with _LOCK:
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("task_id"):
                events.append(item)
    return events


def _apply_events(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for event in events:
        task_id = str(event.get("task_id") or "")
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if not task_id:
            continue
        current = dict(tasks.get(task_id) or {})
        if event_type == "task_created":
            current = {
                "id": task_id,
                "title": payload.get("title", ""),
                "note": payload.get("note", ""),
                "due_at": payload.get("due_at", ""),
                "status": "open",
                "source": payload.get("source", "manual"),
                "reminder": bool(payload.get("reminder", False)),
                "tags": payload.get("tags") if isinstance(payload.get("tags"), list) else [],
                "created_at": event.get("ts", ""),
                "updated_at": event.get("ts", ""),
            }
        elif current and event_type == "task_updated":
            for key in ("title", "note", "due_at", "source", "reminder", "tags"):
                if key in payload:
                    current[key] = payload[key]
            current["updated_at"] = event.get("ts", "")
        elif current and event_type == "task_status_changed":
            status = payload.get("status")
            if status in VALID_STATUSES:
                current["status"] = status
                current["completed_at"] = event.get("ts", "") if status == "done" else current.get("completed_at", "")
                current["updated_at"] = event.get("ts", "")
        if current:
            tasks[task_id] = current
    return tasks


def create_task(
    *,
    title: str,
    due_at: str | None = None,
    note: str = "",
    source: str = "manual",
    reminder: bool = False,
    tags: list[str] | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    clean_title = _clean_text(title, limit=160)
    if not clean_title:
        raise ValueError("title is required")
    created_at = _now_iso()
    task_id = _task_id(clean_title, created_at)
    payload = {
        "title": clean_title,
        "note": _clean_text(note, limit=500),
        "due_at": _normalize_due_at(due_at),
        "source": _clean_text(source, limit=80) or "manual",
        "reminder": bool(reminder),
        "tags": [_clean_text(tag, limit=40) for tag in (tags or []) if _clean_text(tag, limit=40)][:8],
    }
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_type": "task_created",
        "task_id": task_id,
        "ts": created_at,
        "payload": payload,
    }
    _append_event(path or TASK_LOG_PATH, event)
    return _apply_events([event])[task_id]


def list_tasks(
    *,
    status: Literal["open", "done", "cancelled", "all"] = "open",
    limit: int = 50,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    tasks = list(_apply_events(_read_events(path)).values())
    if status != "all":
        tasks = [task for task in tasks if task.get("status") == status]
    tasks.sort(key=lambda task: ((task.get("due_at") or "9999-99-99"), task.get("created_at") or ""), reverse=False)
    return tasks[: max(1, min(int(limit or 50), 200))]


def update_task(
    task_id: str,
    *,
    title: str | None = None,
    due_at: str | None = None,
    note: str | None = None,
    source: str | None = None,
    reminder: bool | None = None,
    tags: list[str] | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    tasks = _apply_events(_read_events(path))
    if task_id not in tasks:
        raise KeyError(task_id)
    payload: dict[str, Any] = {}
    if title is not None:
        clean_title = _clean_text(title, limit=160)
        if not clean_title:
            raise ValueError("title cannot be empty")
        payload["title"] = clean_title
    if due_at is not None:
        payload["due_at"] = _normalize_due_at(due_at)
    if note is not None:
        payload["note"] = _clean_text(note, limit=500)
    if source is not None:
        payload["source"] = _clean_text(source, limit=80) or "manual"
    if reminder is not None:
        payload["reminder"] = bool(reminder)
    if tags is not None:
        payload["tags"] = [_clean_text(tag, limit=40) for tag in tags if _clean_text(tag, limit=40)][:8]
    if not payload:
        return tasks[task_id]
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_type": "task_updated",
        "task_id": task_id,
        "ts": _now_iso(),
        "payload": payload,
    }
    _append_event(path or TASK_LOG_PATH, event)
    tasks = _apply_events(_read_events(path))
    return tasks[task_id]


def set_task_status(task_id: str, status: Literal["open", "done", "cancelled"], *, path: Path | None = None) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    tasks = _apply_events(_read_events(path))
    if task_id not in tasks:
        raise KeyError(task_id)
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_type": "task_status_changed",
        "task_id": task_id,
        "ts": _now_iso(),
        "payload": {"status": status},
    }
    _append_event(path or TASK_LOG_PATH, event)
    tasks = _apply_events(_read_events(path))
    return tasks[task_id]
