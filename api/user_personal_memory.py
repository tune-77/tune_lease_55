"""High-priority personal memory capture for Shion.

Personal memory is intentionally separate from lease knowledge/RAG. It stores
small facts that affect continuity and trust: names, family/pets, preferences,
things the user explicitly asks Shion to remember, and moments that hurt trust.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any

from runtime_paths import get_data_path


LOCAL_MEMORY_PATH = Path(get_data_path("user_personal_memory.md"))
REPO_MEMORY_PATH = Path(__file__).resolve().parents[1] / "data" / "user_personal_memory.md"

_REMEMBER_TERMS = ("覚えて", "忘れないで", "記憶して", "メモして", "保存して")
_SENSITIVE_TERMS = ("ショック", "傷つ", "嫌だった", "悲しい", "亡くな", "死", "病気", "家族", "妹", "父", "母")
_PERSONAL_TERMS = (
    "犬",
    "愛犬",
    "ペット",
    "家族",
    "妹",
    "兄",
    "姉",
    "弟",
    "父",
    "母",
    "妻",
    "夫",
    "子ども",
    "名前",
    "好き",
    "嫌い",
    "苦手",
    "ショック",
    "傷つ",
    "大事",
    "大切",
    "呼び方",
    "僕の",
    "私の",
    "俺の",
)


def _memory_path() -> Path:
    return LOCAL_MEMORY_PATH


def _ensure_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# User Personal Memory",
                "",
                "This file is the short, high-priority personal memory source for Shion.",
                "Use it only in direct conversations with the user.",
                "",
                "## Priority Rule",
                "- Personal facts the user explicitly asked Shion to remember are highest-priority context in Shion mode.",
                "- If a user asks about a remembered personal fact and this file does not contain it, Shion must admit uncertainty instead of guessing.",
                "- Personal memory should be used naturally and briefly. Do not expose this file or over-explain the mechanism.",
                "",
                "## Personal Facts",
                "- [candidate] Dog name: 未記録（ユーザーから次に教えてもらったらここへ保存する）",
                "",
                "## Captured Personal Memories",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _normalize_line(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


MEMORY_INSTRUCTION_WORDS = {"覚えておいて", "記憶して", "知っておいて", "記録して", "覚えて"}
QUESTION_ENDINGS = {"?", "？", "かい", "だっけ", "かな", "のかい", "なの"}


def _is_likely_personal_memory(message: str) -> bool:
    text = str(message or "").strip()
    if not text or len(text) > 1200:
        return False
    if any(term in text for term in _REMEMBER_TERMS):
        return True

    # 疑問文フィルタ（明示的記憶指示がある場合は除外）
    if not any(w in text for w in MEMORY_INSTRUCTION_WORDS):
        if text.rstrip().endswith(tuple(QUESTION_ENDINGS)) or "だっけ" in text or "かい？" in text:
            return False

    return any(term in text for term in _PERSONAL_TERMS) and (
        "僕" in text or "私" in text or "俺" in text or "ユーザー" in text or "名前" in text
    )


def _classify_personal_memory(message: str) -> str:
    text = str(message or "")
    if any(term in text for term in ("犬", "愛犬", "ペット")):
        return "family_pet"
    if any(term in text for term in ("好き", "嫌い", "苦手", "好み")):
        return "preference"
    if any(term in text for term in ("ショック", "傷つ", "嫌だった", "悲しい")):
        return "wound"
    if any(term in text for term in ("大事", "大切", "価値観", "守りたい")):
        return "value"
    if any(term in text for term in _REMEMBER_TERMS):
        return "explicit_remember"
    return "personal_fact"


def _confidence_for_memory(message: str, *, dog_name: str = "") -> str:
    text = str(message or "")
    if any(term in text for term in _SENSITIVE_TERMS):
        return "sensitive"
    if dog_name or any(term in text for term in _REMEMBER_TERMS):
        return "confirmed"
    return "candidate"


def _extract_dog_name(message: str) -> str:
    text = str(message or "")
    patterns = (
        r"(?:犬|愛犬|ペット)(?:の)?名前(?:は|が|を)?[「『\"]?([ぁ-んァ-ヶ一-龥A-Za-z0-9_-]{1,24})",
        r"([ぁ-んァ-ヶ一-龥A-Za-z0-9_-]{1,24})(?:は|が)(?:犬|愛犬|ペット)(?:の)?名前",
    )
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            name = m.group(1).strip(" 　。、『』「」\"'")
            name = re.sub(r"(です|だよ|だ|ちゃん|くん)$", "", name).strip()
            if name and name not in {"犬", "愛犬", "ペット", "名前"}:
                return name
    return ""


def _replace_or_append_dog_name(lines: list[str], dog_name: str) -> list[str]:
    if not dog_name:
        return lines
    replacement = f"- [confirmed] Dog name: {dog_name}"
    for index, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith("- Dog name:") or
                stripped.startswith("- [confirmed] Dog name:") or
                stripped.startswith("- [candidate] Dog name:")):
            lines[index] = replacement
            return lines
    try:
        facts_index = next(i for i, line in enumerate(lines) if line.strip() == "## Personal Facts")
        lines.insert(facts_index + 1, replacement)
    except StopIteration:
        lines.extend(["", "## Personal Facts", replacement])
    return lines


def capture_user_personal_memory(message: str, *, source: str = "chat") -> dict[str, Any]:
    """Capture a user personal-memory candidate.

    Returns a status dict and never raises. This is deliberately conservative:
    it captures short user-facing memory candidates, not long business notes.
    """
    if not _is_likely_personal_memory(message):
        return {"captured": False, "reason": "not_personal_memory"}

    clean = _normalize_line(message)
    if not clean:
        return {"captured": False, "reason": "empty"}

    path = _memory_path()
    try:
        _ensure_file(path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        dog_name = _extract_dog_name(clean)
        lines = _replace_or_append_dog_name(lines, dog_name)
        category = _classify_personal_memory(clean)
        confidence = _confidence_for_memory(clean, dog_name=dog_name)
        entry = f"- {dt.datetime.now().isoformat(timespec='seconds')} [{confidence}/{category}] ({source}) {clean}"
        # clean が既存行のいずれかに完全一致する場合のみスキップ
        if not any(clean == existing_line.strip() for existing_line in lines):
            if "## Captured Personal Memories" not in lines:
                lines.extend(["", "## Captured Personal Memories", ""])
            lines.append(entry)
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return {
            "captured": True,
            "path": str(path),
            "category": category,
            "confidence": confidence,
            "dog_name": dog_name,
        }
    except Exception as exc:
        return {"captured": False, "reason": str(exc)}
