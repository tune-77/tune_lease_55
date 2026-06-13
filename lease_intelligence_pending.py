"""Track investigation promises Shion makes and execute them on the next turn."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any

from runtime_paths import get_data_path

PENDING_PATH = get_data_path("shion_pending_tasks.json")

_PROMISE_PATTERNS = [
    r"調べ(?:てみ|ます|ました|てみます|てみました|てきます|させていただき)",
    r"確認(?:し|して|します|しました|してみます|させていただき)",
    r"検索(?:し|して|します|しました)",
    r"調査(?:し|して|します|しました|してみます|いたします)",
    r"後で(?:確認|調査|調べ)",
    r"次回(?:確認|調査|調べ)",
    r"ご報告(?:し|します|いたします)",
    r"報告(?:し|します|いたします)",
    r"進捗.*(?:報告|お知らせ)",
    r"改めて.*(?:報告|確認|調べ)",
]
_PROMISE_RE = re.compile("|".join(_PROMISE_PATTERNS))


def _load() -> list[dict]:
    try:
        with open(PENDING_PATH, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(tasks: list[dict]) -> None:
    try:
        with open(PENDING_PATH, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def extract_and_save_promises(user_message: str, shion_reply: str) -> list[str]:
    """If Shion's reply contains an investigation promise, record the topic as pending."""
    if not _PROMISE_RE.search(shion_reply):
        return []
    topic = user_message.strip()[:300]
    task_id = str(uuid.uuid4())[:8]
    tasks = _load()
    tasks.append({
        "id": task_id,
        "topic": topic,
        "promised_at": datetime.now().isoformat(),
        "status": "pending",
    })
    _save(tasks)
    return [task_id]


def get_pending_tasks() -> list[dict[str, Any]]:
    return [t for t in _load() if t.get("status") == "pending"]


def mark_done(task_ids: list[str]) -> None:
    if not task_ids:
        return
    tasks = _load()
    id_set = set(task_ids)
    for t in tasks:
        if t.get("id") in id_set:
            t["status"] = "done"
            t["done_at"] = datetime.now().isoformat()
    _save(tasks)
