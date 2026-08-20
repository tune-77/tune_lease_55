"""Append-only Shion -> Codex agent consultation queue.

Shion may notice an operational anomaly, but it must not execute Codex agents
or mutate code from chat. This queue records a sanitized consultation request
that Codex or a scheduled review can pick up later.
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
QUEUE_PATH = Path(get_data_path("shion_agent_consultation_queue.jsonl"))
VALID_STATUSES = {"open", "in_review", "done", "cancelled"}
VALID_PRIORITIES = {"low", "medium", "high"}
VALID_AGENT_NAMES = {
    "judgment-asset-auditor",
    "ledger-consistency-auditor",
    "scoring-auditor",
    "rule-validator",
    "migration-validator",
    "data-quality-checker",
    "security-checker",
    "report-stylist",
    "build-runner",
    "api-health-checker",
    "log-file-analyzer",
    "code-reviewer",
    "test-runner",
    "test-result-analyzer",
    "file-searcher",
}
_LOCK = threading.RLock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_text(value: Any, *, limit: int = 500) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    return text[:limit].rstrip()


def _queue_id(agent: str, title: str, created_at: str) -> str:
    raw = f"{agent}|{title}|{created_at}|{uuid.uuid4().hex}"
    return "agentq_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _read_events(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or QUEUE_PATH
    if not target.exists():
        return []
    events: list[dict[str, Any]] = []
    with _LOCK:
        for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("queue_id"):
                events.append(item)
    return events


def _apply_events(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for event in events:
        queue_id = str(event.get("queue_id") or "")
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if not queue_id:
            continue
        current = dict(items.get(queue_id) or {})
        if event_type == "consultation_requested":
            current = {
                "id": queue_id,
                "agent": payload.get("agent", ""),
                "title": payload.get("title", ""),
                "reason": payload.get("reason", ""),
                "observed_signal": payload.get("observed_signal", ""),
                "suggested_scope": payload.get("suggested_scope", ""),
                "priority": payload.get("priority", "medium"),
                "status": "open",
                "source": payload.get("source", "shion_dialogue"),
                "created_at": event.get("ts", ""),
                "updated_at": event.get("ts", ""),
                "report_target": payload.get("report_target", ""),
                "safety_policy": "queue_only_no_agent_execution_no_code_change",
            }
        elif current and event_type == "consultation_status_changed":
            status = payload.get("status")
            if status in VALID_STATUSES:
                current["status"] = status
                current["resolution_note"] = payload.get("note", current.get("resolution_note", ""))
                current["updated_at"] = event.get("ts", "")
        if current:
            items[queue_id] = current
    return items


def request_consultation(
    *,
    agent: str,
    title: str,
    reason: str,
    observed_signal: str = "",
    suggested_scope: str = "",
    priority: str = "medium",
    source: str = "shion_dialogue",
    path: Path | None = None,
    record_writeback: bool = False,
) -> dict[str, Any]:
    clean_agent = _clean_text(agent, limit=80)
    if clean_agent not in VALID_AGENT_NAMES:
        raise ValueError(f"unknown agent: {clean_agent}")
    clean_title = _clean_text(title, limit=160)
    clean_reason = _clean_text(reason, limit=500)
    if not clean_title:
        raise ValueError("title is required")
    if not clean_reason:
        raise ValueError("reason is required")
    clean_priority = _clean_text(priority, limit=20) or "medium"
    if clean_priority not in VALID_PRIORITIES:
        clean_priority = "medium"
    created_at = _now_iso()
    queue_id = _queue_id(clean_agent, clean_title, created_at)
    payload = {
        "agent": clean_agent,
        "title": clean_title,
        "reason": clean_reason,
        "observed_signal": _clean_text(observed_signal, limit=400),
        "suggested_scope": _clean_text(suggested_scope, limit=400),
        "priority": clean_priority,
        "source": _clean_text(source, limit=80) or "shion_dialogue",
        "report_target": f".claude/reports/{_agent_report_dir(clean_agent)}/latest.md",
    }
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_type": "consultation_requested",
        "queue_id": queue_id,
        "ts": created_at,
        "payload": payload,
    }
    _append_event(path or QUEUE_PATH, event)
    item = _apply_events([event])[queue_id]
    if path is None:
        _log_action_ledger(item)
        if record_writeback:
            _record_cloudrun_consultation_event(item)
    return item


def list_consultations(
    *,
    status: Literal["open", "in_review", "done", "cancelled", "all"] = "open",
    agent: str = "all",
    limit: int = 50,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    items = list(_apply_events(_read_events(path)).values())
    if status != "all":
        items = [item for item in items if item.get("status") == status]
    if agent != "all":
        items = [item for item in items if item.get("agent") == agent]
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda item: (priority_rank.get(str(item.get("priority")), 1), str(item.get("created_at") or "")))
    return items[: max(1, min(int(limit or 50), 200))]


def set_consultation_status(
    queue_id: str,
    status: Literal["open", "in_review", "done", "cancelled"],
    *,
    note: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    items = _apply_events(_read_events(path))
    if queue_id not in items:
        raise KeyError(queue_id)
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_type": "consultation_status_changed",
        "queue_id": queue_id,
        "ts": _now_iso(),
        "payload": {"status": status, "note": _clean_text(note, limit=300)},
    }
    _append_event(path or QUEUE_PATH, event)
    return _apply_events(_read_events(path))[queue_id]


def build_agent_consultation_prompt_block(limit: int = 3, *, path: Path | None = None) -> str:
    items = list_consultations(status="open", limit=limit, path=path)
    lines = [
        "【AGENT相談キュー（read-only）】",
        "紫苑はCodex AGENTを直接実行しない。異常・断線・監査が必要な時は request_agent_consultation で相談票を作る。",
        "相談票は後でCodexが読み、必要なAGENTを実行して .claude/reports/.../latest.md に結果を書く。",
    ]
    if items:
        lines.append("未処理の相談票:")
        for item in items[:limit]:
            lines.append(f"- {item['id']} / {item['agent']} / {item['priority']}: {item['title']}")
    else:
        lines.append("未処理の相談票: なし")
    return "\n".join(lines)


def _agent_report_dir(agent: str) -> str:
    return {
        "judgment-asset-auditor": "judgment-asset-audit",
        "ledger-consistency-auditor": "ledger-consistency",
        "scoring-auditor": "scoring-audit",
        "rule-validator": "rule-validation",
        "migration-validator": "migration",
        "data-quality-checker": "data-quality",
        "security-checker": "security",
        "report-stylist": "report-stylist",
        "build-runner": "build",
        "api-health-checker": "api-health",
        "log-file-analyzer": "log-analysis",
        "code-reviewer": "code-review",
        "test-runner": "test-results",
        "test-result-analyzer": "test-results",
        "file-searcher": "file-searcher",
    }.get(agent, agent)


def _log_action_ledger(item: dict[str, Any], *, path: Path | None = None) -> None:
    try:
        from api.shion_action_ledger import log_action

        log_action(
            "agent_consultation_requested",
            summary=f"{item.get('agent')}: {item.get('title')}",
            observed_sources=[str(path or QUEUE_PATH)],
            risk_level="medium" if item.get("priority") == "high" else "low",
            requires_user_approval=True,
            target=str(item.get("report_target") or ""),
            result="queued",
        )
    except Exception:
        pass


def _record_cloudrun_consultation_event(item: dict[str, Any]) -> None:
    try:
        from api.cloudrun_writeback import record_cloudrun_input_event

        record_cloudrun_input_event(
            event_type="shion_agent_consultation_requested",
            surface="shion_agent_consultation",
            payload={**item, "schema_version": SCHEMA_VERSION},
        )
    except Exception:
        pass
