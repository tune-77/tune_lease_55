"""Common current datetime context for AI prompts."""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - Python without zoneinfo
    ZoneInfo = None  # type: ignore


CURRENT_DATETIME_LABEL = "【現在日時】"


def _jst() -> dt.tzinfo:
    if ZoneInfo is not None:
        return ZoneInfo("Asia/Tokyo")
    return dt.timezone(dt.timedelta(hours=9), name="JST")


def current_datetime_prompt_block(now: dt.datetime | None = None) -> str:
    """Return a compact JST datetime block for AI system prompts."""
    current = now or dt.datetime.now(_jst())
    if current.tzinfo is None:
        current = current.replace(tzinfo=_jst())
    current = current.astimezone(_jst())
    weekday = ["月", "火", "水", "木", "金", "土", "日"][current.weekday()]
    return (
        f"{CURRENT_DATETIME_LABEL}{current.strftime('%Y年%m月%d日 %H:%M')} "
        f"(JST, {current.date().isoformat()}, {weekday}曜日)"
    )


def current_time_reply_if_requested(message: str, now: dt.datetime | None = None) -> str | None:
    """Return a deterministic JST answer for simple current-time questions."""
    text = re.sub(r"\s+", "", str(message or ""))
    if not text:
        return None
    asks_time = any(
        key in text
        for key in (
            "今何時",
            "いま何時",
            "今の時間",
            "現在時刻",
            "今時刻",
            "いまの時間",
        )
    )
    if not asks_time:
        return None
    current = now or dt.datetime.now(_jst())
    if current.tzinfo is None:
        current = current.replace(tzinfo=_jst())
    current = current.astimezone(_jst())
    weekday = ["月", "火", "水", "木", "金", "土", "日"][current.weekday()]
    return (
        f"今はJSTで{current.strftime('%H:%M')}です。"
        f"{current.strftime('%Y年%m月%d日')}（{weekday}曜日）です。"
    )


def with_current_datetime_context(system_prompt: str | None) -> str:
    """Append current datetime context unless the prompt already has it."""
    text = str(system_prompt or "").strip()
    if CURRENT_DATETIME_LABEL in text:
        return text
    block = current_datetime_prompt_block()
    return f"{text}\n\n{block}".strip() if text else block


def ensure_current_datetime_message(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Ensure OpenAI/Ollama-style messages contain current datetime in a system message.

    The input list is not mutated.
    """
    copied = [dict(m) for m in (messages or [])]
    if any(CURRENT_DATETIME_LABEL in str(m.get("content") or "") for m in copied):
        return copied

    for msg in copied:
        if msg.get("role") == "system":
            msg["content"] = with_current_datetime_context(str(msg.get("content") or ""))
            return copied

    return [{"role": "system", "content": current_datetime_prompt_block()}, *copied]
