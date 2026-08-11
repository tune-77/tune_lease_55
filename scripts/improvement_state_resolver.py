"""Shared state resolution for improvement PM ledger entries.

The improvement PM flow has several consumers (API display, report sync,
recursive self-improvement). They must agree on which ledger row is effective
when the same canonical key is seen multiple times.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


TERMINAL_LEDGER_STATUSES = {
    "applied",
    "approved",
    "deleted",
    "rejected",
    "deferred",
    "parked",
    "suppressed",
    "rule_registered",
    "apply_failed",
}

PARKED_LEDGER_STATUSES = {"parked", "deferred", "deleted", "rejected", "suppressed"}


def normalize_status(value: object) -> str:
    parts = str(value or "").split()
    return parts[0].strip().lower() if parts else ""


def normalize_title(title: object) -> str:
    return re.sub(r"\s+", "", str(title or "").strip().lower())


def is_terminal_status(status: object) -> bool:
    return normalize_status(status) in TERMINAL_LEDGER_STATUSES


def prefer_ledger_entry(current: dict | None, candidate: dict) -> dict:
    """Return the effective row between two rows for the same alias.

    Human/resolution decisions must survive later automatic needs_review rows.
    If both rows are terminal, the later row wins because it represents a newer
    explicit resolution or audit correction (for example applied -> apply_failed).
    """
    if current is None:
        return candidate
    current_terminal = is_terminal_status(current.get("status"))
    candidate_terminal = is_terminal_status(candidate.get("status"))
    if candidate_terminal and not current_terminal:
        return candidate
    if current_terminal and not candidate_terminal:
        return current
    return candidate


def iter_ledger_entries(ledger_path: Path) -> Iterable[dict]:
    if not ledger_path.exists():
        return
    try:
        lines = ledger_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            yield entry


def entry_aliases(entry: dict) -> set[str]:
    return {
        alias
        for alias in (
            str(entry.get("key") or "").strip(),
            str(entry.get("canonical_key") or "").strip(),
            str(entry.get("rev_id") or "").strip(),
        )
        if alias
    }


def load_latest_by_alias(ledger_path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for entry in iter_ledger_entries(ledger_path):
        if not normalize_status(entry.get("status")):
            continue
        for alias in entry_aliases(entry):
            latest[alias] = prefer_ledger_entry(latest.get(alias), entry)
    return latest


def load_latest_status_by_key(ledger_path: Path) -> dict[str, str]:
    return {
        key: str(entry.get("status") or "")
        for key, entry in load_latest_by_alias(ledger_path).items()
    }


def load_latest_status_by_title(ledger_path: Path) -> dict[str, str]:
    latest: dict[str, dict] = {}
    for entry in iter_ledger_entries(ledger_path):
        title = normalize_title(entry.get("title"))
        if not title or not normalize_status(entry.get("status")):
            continue
        latest[title] = prefer_ledger_entry(latest.get(title), entry)
    return {
        title: str(entry.get("status") or "")
        for title, entry in latest.items()
    }


def status_to_report_bucket(status: object) -> str:
    normalized = normalize_status(status)
    if normalized == "applied":
        return "applied"
    if normalized in PARKED_LEDGER_STATUSES:
        return "parked"
    return ""
