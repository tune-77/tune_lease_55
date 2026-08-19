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

import yaml

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Folders that are operational/self-generated rather than personal knowledge
# notes. Vault-wide scans (tagging, taxonomy audit, theme radar) should skip
# these so curator output isn't built from its own reports or Shion's
# internal logs (same self-reference concern as mana_obsidian_curator.py).
VAULT_SCAN_EXCLUDE_DIR_NAMES: frozenset[str] = frozenset(
    {
        "Private Reflection",
        "Dialogue",
        "AI Chat",
        "Cloud SQL Summaries",
        "Vertex Distilled",
        "Content Strategy Journal",
    }
)


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


def parse_note_frontmatter(text: str) -> dict[str, Any]:
    """Parse a note's YAML frontmatter. Returns {} for missing/invalid blocks."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    try:
        parsed = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def note_tags(frontmatter: dict[str, Any]) -> list[str]:
    raw = frontmatter.get("tags")
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(tag).strip() for tag in raw if str(tag or "").strip()]


def iter_vault_markdown_files(
    vault: Path,
    *,
    exclude_dir_names: frozenset[str] = VAULT_SCAN_EXCLUDE_DIR_NAMES,
):
    """Yield vault markdown files, skipping operational/self-generated folders."""
    for path in sorted(vault.rglob("*.md")):
        if any(part in exclude_dir_names for part in path.relative_to(vault).parts[:-1]):
            continue
        yield path
