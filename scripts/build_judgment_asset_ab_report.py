#!/usr/bin/env python3
"""Build A/B candidates for judgment assets.

This report pairs active judgment assets that appear to compete in the same
concept/domain. It is observational only and never changes the active store.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_JSON = PROJECT_ROOT / "data" / "canonical_judgment_rules.json"
DEFAULT_FEEDBACK_JSONL = PROJECT_ROOT / "data" / "judgment_asset_usage_feedback.jsonl"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "judgment_asset_ab_latest.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "judgment_asset_ab_latest.md"
GUARDRAIL = "ab_observation_only_no_auto_winner_no_prompt_no_scoring_change"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
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


def _rules(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    rules = canonical.get("rules")
    if not isinstance(rules, list):
        rules = canonical.get("canonical_rules")
    return [
        r for r in (rules or [])
        if isinstance(r, dict) and r.get("status") == "active" and r.get("private") is not True and r.get("id")
    ]


def _outcome(row: dict[str, Any]) -> str:
    value = str(row.get("outcome") or row.get("status") or "").strip().lower()
    if value in {"used", "helped", "challenged", "rejected", "neutral"}:
        return value
    return ""


def build_report(canonical: dict[str, Any], feedback_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rules = _rules(canonical)
    feedback: dict[str, Counter[str]] = defaultdict(Counter)
    for row in feedback_rows:
        rid = str(row.get("rule_id") or row.get("judgment_asset_id") or row.get("asset_id") or "").strip()
        outcome = _outcome(row)
        if rid and outcome:
            feedback[rid][outcome] += 1

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        key = str(rule.get("concept") or "").strip()
        if not key:
            domains = rule.get("domains") if isinstance(rule.get("domains"), list) else []
            key = str(domains[0]) if domains else "ungrouped"
        groups[key].append(rule)

    pairs: list[dict[str, Any]] = []
    for concept, items in groups.items():
        if len(items) < 2:
            continue
        ranked = sorted(items, key=lambda r: _ab_score(r, feedback), reverse=True)
        a, b = ranked[0], ranked[1]
        pairs.append(
            {
                "concept": concept,
                "asset_a": _asset_summary(a, feedback),
                "asset_b": _asset_summary(b, feedback),
                "test_plan": "次の類似案件でA/Bの両方を稟議確認項目へ変換し、helped/challenged/neutralで記録する。",
                "winner_policy": "helped率とchallenged/rejectedの少なさで見る。自動で勝者確定しない。",
            }
        )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "guardrail": GUARDRAIL,
        "summary": {
            "active_rules": len(rules),
            "candidate_pairs": len(pairs),
            "grouped_concepts": sum(1 for items in groups.values() if len(items) >= 2),
        },
        "pairs": pairs,
    }


def _ab_score(rule: dict[str, Any], feedback: dict[str, Counter[str]]) -> float:
    rid = str(rule.get("id") or "")
    counts = feedback.get(rid, Counter())
    confidence = float(rule.get("confidence") or 0.0)
    return confidence * 10 + counts.get("helped", 0) * 4 + counts.get("used", 0) - counts.get("challenged", 0) * 3 - counts.get("rejected", 0) * 5


def _asset_summary(rule: dict[str, Any], feedback: dict[str, Counter[str]]) -> dict[str, Any]:
    rid = str(rule.get("id") or "")
    return {
        "rule_id": rid,
        "concept": str(rule.get("concept") or ""),
        "statement": str(rule.get("canonical_statement") or rule.get("statement") or "")[:220],
        "confidence": rule.get("confidence"),
        "feedback": dict(feedback.get(rid, Counter())),
        "ab_score": round(_ab_score(rule, feedback), 3),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Judgment Asset A/B Report",
        "",
        f"- Guardrail: `{report.get('guardrail')}`",
        f"- Candidate pairs: {(report.get('summary') or {}).get('candidate_pairs', 0)}",
    ]
    for pair in report.get("pairs") or []:
        a = pair.get("asset_a") or {}
        b = pair.get("asset_b") or {}
        lines.extend([
            "",
            f"## {pair.get('concept')}",
            f"- A: `{a.get('rule_id')}` score={a.get('ab_score')} {a.get('statement')}",
            f"- B: `{b.get('rule_id')}` score={b.get('ab_score')} {b.get('statement')}",
            f"- Plan: {pair.get('test_plan')}",
        ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--feedback-jsonl", type=Path, default=DEFAULT_FEEDBACK_JSONL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = build_report(_read_json(args.canonical_json), _read_jsonl(args.feedback_jsonl))
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(report), encoding="utf-8")
    print(f"wrote={args.output_json}")
    print(f"wrote={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
