#!/usr/bin/env python3
"""Record field feedback for canonical judgment assets.

This writes only a local JSONL ledger. It does not promote, demote, inject into
prompts, or change scoring by itself.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_JSON = PROJECT_ROOT / "data" / "canonical_judgment_rules.json"
DEFAULT_FEEDBACK_JSONL = PROJECT_ROOT / "data" / "judgment_asset_usage_feedback.jsonl"
VALID_OUTCOMES = {"used", "helped", "challenged", "rejected", "neutral"}


def build_feedback_entry(
    *,
    rule_id: str,
    outcome: str,
    case_id: str = "",
    note: str = "",
    source: str = "manual",
    used_at: str = "",
) -> dict[str, str]:
    normalized_rule_id = rule_id.strip()
    normalized_outcome = outcome.strip().lower()
    if not normalized_rule_id:
        raise ValueError("rule_id is required")
    if normalized_outcome not in VALID_OUTCOMES:
        raise ValueError(f"outcome must be one of: {', '.join(sorted(VALID_OUTCOMES))}")
    return {
        "schema_version": "1",
        "rule_id": normalized_rule_id,
        "outcome": normalized_outcome,
        "case_id": case_id.strip(),
        "note": note.strip(),
        "source": source.strip() or "manual",
        "used_at": used_at.strip() or datetime.now().isoformat(timespec="seconds"),
    }


def append_feedback(path: Path, entry: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _canonical_rules(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    rules = canonical.get("rules")
    if isinstance(rules, list):
        return [item for item in rules if isinstance(item, dict)]
    rules = canonical.get("canonical_rules")
    if isinstance(rules, list):
        return [item for item in rules if isinstance(item, dict)]
    return []


def load_active_rules(path: Path = DEFAULT_CANONICAL_JSON) -> dict[str, dict[str, Any]]:
    rules: dict[str, dict[str, Any]] = {}
    for rule in _canonical_rules(_read_json(path)):
        rule_id = str(rule.get("id") or "").strip()
        if not rule_id:
            continue
        if rule.get("status") != "active":
            continue
        if rule.get("private") is True:
            continue
        rules[rule_id] = rule
    return rules


def resolve_rule_id(value: str, active_rules: dict[str, dict[str, Any]]) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("rule_id is required")
    if normalized in active_rules:
        return normalized

    concept_matches = [
        rule_id
        for rule_id, rule in active_rules.items()
        if str(rule.get("concept") or "").strip() == normalized
    ]
    if len(concept_matches) == 1:
        return concept_matches[0]
    if len(concept_matches) > 1:
        raise ValueError(f"concept is ambiguous; use one of: {', '.join(sorted(concept_matches))}")

    raise ValueError(f"unknown active rule_id or concept: {normalized}")


def format_active_rules(active_rules: dict[str, dict[str, Any]]) -> str:
    if not active_rules:
        return "active judgment assets: 0"
    lines = [f"active judgment assets: {len(active_rules)}"]
    for rule_id, rule in sorted(active_rules.items(), key=lambda item: str(item[1].get("concept") or item[0])):
        concept = str(rule.get("concept") or "").strip() or "-"
        statement = str(rule.get("canonical_statement") or "").strip()
        lines.append(f"- {rule_id}  {concept}: {statement}")
    return "\n".join(lines)


def run_growth_report() -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "judgment_asset_growth_report.py")],
        check=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rule_id", nargs="?", help="Active rule id, or an active rule concept")
    parser.add_argument("outcome", nargs="?", choices=sorted(VALID_OUTCOMES))
    parser.add_argument("--case-id", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--source", default="manual")
    parser.add_argument("--used-at", default="")
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--feedback-jsonl", type=Path, default=DEFAULT_FEEDBACK_JSONL)
    parser.add_argument("--list-active", action="store_true", help="Show active judgment assets and exit")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the entry without writing")
    parser.add_argument(
        "--recompute-growth",
        action="store_true",
        help="Rebuild the judgment asset growth report after recording",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    active_rules = load_active_rules(args.canonical_json)
    if args.list_active:
        print(format_active_rules(active_rules))
        return 0

    if not args.rule_id or not args.outcome:
        raise SystemExit("rule_id and outcome are required unless --list-active is used")

    resolved_rule_id = resolve_rule_id(args.rule_id, active_rules)
    rule = active_rules[resolved_rule_id]
    entry = build_feedback_entry(
        rule_id=resolved_rule_id,
        outcome=args.outcome,
        case_id=args.case_id,
        note=args.note,
        source=args.source,
        used_at=args.used_at,
    )
    if args.dry_run:
        print(json.dumps(entry, ensure_ascii=False, sort_keys=True))
        return 0

    append_feedback(args.feedback_jsonl, entry)
    concept = str(rule.get("concept") or "").strip() or "-"
    print(f"recorded {entry['outcome']} for {entry['rule_id']} ({concept}) -> {args.feedback_jsonl}")
    if args.recompute_growth:
        run_growth_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
