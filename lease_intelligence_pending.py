"""Track investigation promises Shion makes and execute them on the next turn.
Also writes countermeasures to the improvement dispatch queue for the daily pipeline.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from runtime_paths import get_data_path

PENDING_PATH = get_data_path("shion_pending_tasks.json")
DISPATCH_QUEUE_PATH = Path.home() / "Library" / "Logs" / "tunelease" / "dispatch_queue.jsonl"

_COUNTERMEASURE_RE = re.compile(
    r"\*{0,2}③\s*対応策\*{0,2}[^\n]*\n(.*?)(?=\n\*{0,2}[①-⑩]|\Z)",
    re.DOTALL,
)

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


def _extract_countermeasure_block(shion_reply: str) -> str:
    """Extract the ③対応策 section from Shion's reply."""
    m = _COUNTERMEASURE_RE.search(shion_reply)
    if not m:
        return ""
    return m.group(1).strip()


def _lines_to_candidates(block: str, user_message: str) -> list[dict]:
    """Convert countermeasure text into dispatch_queue candidate entries."""
    candidates = []
    # Each bullet or numbered line becomes one candidate
    lines = [l.strip().lstrip("-・•*0123456789.）) ").strip() for l in block.splitlines()]
    lines = [l for l in lines if len(l) > 5]
    if not lines:
        # Whole block as single candidate
        lines = [block[:120]]
    for line in lines[:5]:  # max 5 candidates per reply
        candidates.append({
            "id": f"SHION-{str(uuid.uuid4())[:6].upper()}",
            "title": line[:80],
            "category": "shion",
            "reason": f"紫苑が調査した結果の対応策。元の問い: {user_message[:60]}",
            "source": "shion_dialogue",
        })
    return candidates


def save_countermeasures_to_dispatch(user_message: str, shion_reply: str) -> int:
    """If Shion's reply has a ③対応策 section, append it to dispatch_queue.jsonl.
    Returns the number of candidates written (0 if none found).
    """
    block = _extract_countermeasure_block(shion_reply)
    if not block:
        return 0
    candidates = _lines_to_candidates(block, user_message)
    if not candidates:
        return 0
    entry = {
        "type": "improvement_candidates",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "source": "shion",
        "candidates": candidates,
        "message": "紫苑の調査から生成された対応策です。着手・保留・破棄を決めてください。",
    }
    try:
        DISPATCH_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DISPATCH_QUEUE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        return 0
    return len(candidates)
