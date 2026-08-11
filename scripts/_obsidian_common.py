"""Shared helpers duplicated verbatim (or near-verbatim) across scripts/*obsidian* files.

This module only extracts functions whose behavior was already identical across
callers. It does not change any caller's output, CLI, or default values —
each script imports and aliases these under its original private name.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any


def strip_frontmatter(text: str) -> str:
    return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def parse_date(value: str | None) -> date:
    return date.fromisoformat(value) if value else date.today()


def now_local() -> datetime:
    return datetime.now().astimezone()


def emotion_bar(value: float, max_val: float = 100.0) -> str:
    filled = int(round(value / max_val * 10))
    return "█" * filled + "░" * (10 - filled)


def read_json_or_none(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def list_lines(items: list[Any]) -> list[str]:
    if not items:
        return ["- なし"]
    return [f"- {item}" for item in items]
