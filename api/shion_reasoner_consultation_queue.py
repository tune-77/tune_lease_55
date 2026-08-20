"""Append-only Shion -> external reasoner consultation queue.

Shion can ask for a second opinion from Codex, Gemini, or Claude without
executing code, sending raw customer data, or changing PR/data state. The queue
stores sanitized summaries only; a human/Codex workflow can decide whether and
how to forward the request.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from runtime_paths import get_data_path


SCHEMA_VERSION = 1
QUEUE_PATH = Path(get_data_path("shion_reasoner_consultation_queue.jsonl"))
VALID_STATUSES = {"open", "in_review", "answered", "done", "cancelled"}
VALID_PRIORITIES = {"low", "medium", "high"}
VALID_TARGETS = {"codex", "gemini", "claude", "auto"}
VALID_PURPOSES = {
    "hypothesis_check",
    "judgment_review",
    "code_review",
    "design_review",
    "fallback",
    "long_context_review",
}
VALID_PRIVACY_LEVELS = {"safe_summary_only", "local_only", "do_not_send"}
SAFETY_POLICY = "queue_only_safe_summary_no_raw_customer_data_no_code_execution"
_LOCK = threading.RLock()
_SENSITIVE_PATTERNS = (
    (re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"(?<!\d)(?:0\d{1,4}[-ー－]?\d{1,4}[-ー－]?\d{3,4})(?!\d)"), "[PHONE]"),
    (re.compile(r"(?<!\d)\d{13}(?!\d)"), "[CORPORATE_NUMBER]"),
    (
        re.compile(
            r"(?:株式会社|有限会社|合同会社)[^\s、。]{1,30}|"
            r"[^\s、。]{1,30}(?:株式会社|有限会社|合同会社)"
        ),
        "[COMPANY]",
    ),
    (re.compile(r"(?<![\d.])\d{5,}(?:\.\d+)?(?![\d.])"), "[LARGE_NUMBER]"),
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_text(value: Any, *, limit: int = 800, sanitize: bool = True) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if sanitize:
        for pattern, replacement in _SENSITIVE_PATTERNS:
            text = pattern.sub(replacement, text)
    return text[:limit].rstrip()


def _queue_id(target: str, purpose: str, created_at: str) -> str:
    raw = f"{target}|{purpose}|{created_at}|{uuid.uuid4().hex}"
    return "reasonq_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


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
        if event_type == "reasoner_consultation_requested":
            current = {
                "id": queue_id,
                "target": payload.get("target", "auto"),
                "purpose": payload.get("purpose", "hypothesis_check"),
                "title": payload.get("title", ""),
                "question": payload.get("question", ""),
                "shion_hypothesis": payload.get("shion_hypothesis", ""),
                "context_summary": payload.get("context_summary", ""),
                "privacy_level": payload.get("privacy_level", "safe_summary_only"),
                "priority": payload.get("priority", "medium"),
                "status": "open",
                "source": payload.get("source", "shion_dialogue"),
                "created_at": event.get("ts", ""),
                "updated_at": event.get("ts", ""),
                "safety_policy": SAFETY_POLICY,
            }
        elif current and event_type == "reasoner_consultation_status_changed":
            status = payload.get("status")
            if status in VALID_STATUSES:
                current["status"] = status
                current["resolution_note"] = payload.get("note", current.get("resolution_note", ""))
                current["answered_by"] = payload.get("answered_by", current.get("answered_by", ""))
                current["answer_summary"] = payload.get("answer_summary", current.get("answer_summary", ""))
                current["updated_at"] = event.get("ts", "")
        if current:
            items[queue_id] = current
    return items


def request_reasoner_consultation(
    *,
    target: str,
    purpose: str,
    title: str,
    question: str,
    shion_hypothesis: str,
    context_summary: str = "",
    privacy_level: str = "safe_summary_only",
    priority: str = "medium",
    source: str = "shion_dialogue",
    path: Path | None = None,
    record_writeback: bool = False,
) -> dict[str, Any]:
    clean_target = _clean_text(target, limit=40, sanitize=False) or "auto"
    if clean_target not in VALID_TARGETS:
        clean_target = "auto"
    clean_purpose = _clean_text(purpose, limit=80, sanitize=False) or "hypothesis_check"
    if clean_purpose not in VALID_PURPOSES:
        clean_purpose = "hypothesis_check"
    clean_privacy = _clean_text(privacy_level, limit=40, sanitize=False) or "safe_summary_only"
    if clean_privacy not in VALID_PRIVACY_LEVELS:
        clean_privacy = "safe_summary_only"
    clean_priority = _clean_text(priority, limit=20, sanitize=False) or "medium"
    if clean_priority not in VALID_PRIORITIES:
        clean_priority = "medium"

    clean_title = _clean_text(title, limit=160)
    clean_question = _clean_text(question, limit=1400)
    clean_hypothesis = _clean_text(shion_hypothesis, limit=1400)
    clean_context = _clean_text(context_summary, limit=1800)
    if not clean_title:
        raise ValueError("title is required")
    if not clean_question:
        raise ValueError("question is required")
    if not clean_hypothesis:
        raise ValueError("shion_hypothesis is required")

    created_at = _now_iso()
    queue_id = _queue_id(clean_target, clean_purpose, created_at)
    payload = {
        "target": clean_target,
        "purpose": clean_purpose,
        "title": clean_title,
        "question": clean_question,
        "shion_hypothesis": clean_hypothesis,
        "context_summary": clean_context,
        "privacy_level": clean_privacy,
        "priority": clean_priority,
        "source": _clean_text(source, limit=80, sanitize=False) or "shion_dialogue",
    }
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_type": "reasoner_consultation_requested",
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


def list_reasoner_consultations(
    *,
    status: Literal["open", "in_review", "answered", "done", "cancelled", "all"] = "open",
    target: str = "all",
    limit: int = 50,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    items = list(_apply_events(_read_events(path)).values())
    if status != "all":
        items = [item for item in items if item.get("status") == status]
    if target != "all":
        items = [item for item in items if item.get("target") == target]
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda item: (priority_rank.get(str(item.get("priority")), 1), str(item.get("created_at") or "")))
    return items[: max(1, min(int(limit or 50), 200))]


def set_reasoner_consultation_status(
    queue_id: str,
    status: Literal["open", "in_review", "answered", "done", "cancelled"],
    *,
    note: str = "",
    answered_by: str = "",
    answer_summary: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    items = _apply_events(_read_events(path))
    if queue_id not in items:
        raise KeyError(queue_id)
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_type": "reasoner_consultation_status_changed",
        "queue_id": queue_id,
        "ts": _now_iso(),
        "payload": {
            "status": status,
            "note": _clean_text(note, limit=300),
            "answered_by": _clean_text(answered_by, limit=80, sanitize=False),
            "answer_summary": _clean_text(answer_summary, limit=900),
        },
    }
    _append_event(path or QUEUE_PATH, event)
    return _apply_events(_read_events(path))[queue_id]


def build_reasoner_consultation_prompt_block(limit: int = 3, *, path: Path | None = None) -> str:
    items = list_reasoner_consultations(status="open", limit=limit, path=path)
    lines = [
        "【外部推論相談キュー（read-only）】",
        "紫苑はCodex/Gemini/Claudeを直接暴走実行しない。低確信度・別視点・Codex停止時は request_reasoner_consultation で安全要約の相談票を作る。",
        "生の顧客名・社名・財務数値・秘密は渡さず、safe_summary_only を既定にする。",
    ]
    if items:
        lines.append("未処理の外部推論相談票:")
        for item in items[:limit]:
            lines.append(f"- {item['id']} / {item['target']} / {item['priority']}: {item['title']}")
    else:
        lines.append("未処理の外部推論相談票: なし")
    return "\n".join(lines)


def _log_action_ledger(item: dict[str, Any], *, path: Path | None = None) -> None:
    try:
        from api.shion_action_ledger import log_action

        log_action(
            "reasoner_consultation_requested",
            summary=f"{item.get('target')}: {item.get('title')}",
            observed_sources=[str(path or QUEUE_PATH)],
            risk_level="medium" if item.get("priority") == "high" else "low",
            requires_user_approval=item.get("privacy_level") != "local_only",
            target=str(item.get("target") or ""),
            result="queued",
        )
    except Exception:
        pass


def _record_cloudrun_consultation_event(item: dict[str, Any]) -> None:
    try:
        from api.cloudrun_writeback import record_cloudrun_input_event

        record_cloudrun_input_event(
            event_type="shion_reasoner_consultation_requested",
            surface="shion_reasoner_consultation",
            payload={**item, "schema_version": SCHEMA_VERSION},
        )
    except Exception:
        pass
