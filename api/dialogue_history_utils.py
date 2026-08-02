"""Lease-intelligence dialogue input/history bounding helpers."""

from __future__ import annotations

from typing import Any


def is_long_dialogue_input(message: str, file_type: str | None = None) -> bool:
    text = str(message or "")
    return len(text) >= 1800 or text.count("\n") >= 18 or bool(file_type)


def clip_dialogue_history_text(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    head = max_chars // 2
    tail = max_chars - head
    return (
        value[:head].rstrip()
        + "\n\n...（長文対話のため過去発言を中略）...\n\n"
        + value[-tail:].lstrip()
    )


def compact_dialogue_history(
    history: list[dict[str, Any]],
    *,
    max_messages: int = 24,
    max_chars_per_message: int = 1200,
    total_budget: int = 16000,
) -> list[dict[str, Any]]:
    selected = list(history or [])[-max_messages:]
    compacted: list[dict[str, Any]] = []
    remaining = total_budget
    for msg in reversed(selected):
        if remaining <= 0:
            break
        role = str(msg.get("role") or "")
        content = clip_dialogue_history_text(str(msg.get("content") or ""), min(max_chars_per_message, remaining))
        if not content:
            continue
        compacted.append({**msg, "role": role, "content": content})
        remaining -= len(content)
    return list(reversed(compacted))
