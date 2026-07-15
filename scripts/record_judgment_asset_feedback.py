#!/usr/bin/env python3
"""Record field feedback for canonical judgment assets.

This writes only a local JSONL ledger. It does not promote, demote, inject into
prompts, or change scoring by itself.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rule_id")
    parser.add_argument("outcome", choices=sorted(VALID_OUTCOMES))
    parser.add_argument("--case-id", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--source", default="manual")
    parser.add_argument("--used-at", default="")
    parser.add_argument("--feedback-jsonl", type=Path, default=DEFAULT_FEEDBACK_JSONL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entry = build_feedback_entry(
        rule_id=args.rule_id,
        outcome=args.outcome,
        case_id=args.case_id,
        note=args.note,
        source=args.source,
        used_at=args.used_at,
    )
    append_feedback(args.feedback_jsonl, entry)
    print(f"recorded {entry['outcome']} for {entry['rule_id']} -> {args.feedback_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
