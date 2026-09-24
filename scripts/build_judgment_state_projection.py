#!/usr/bin/env python3
"""Replay Decision State Ledger events into rebuildable case projections."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from decision_state_ledger import DEFAULT_LEDGER_PATH, audit_ledger, deduplicate_events, load_events


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "judgment_state_current.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "judgment_state_audit_latest.md"


def _sort_key(event: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(event.get("occurred_at") or ""),
        str(event.get("recorded_at") or ""),
        str(event.get("event_id") or ""),
    )


def _active_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    corrected: set[str] = set()
    for event in events:
        if event.get("event_type") == "event_corrected":
            corrected.update(str(item) for item in (event.get("relations") or {}).get("supersedes") or [])
    return [event for event in events if str(event.get("event_id") or "") not in corrected], corrected


def project_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    unique = deduplicate_events(sorted(events, key=_sort_key))
    active, corrected = _active_events(unique)
    cases: dict[str, dict[str, Any]] = {}
    findings: list[dict[str, str]] = []

    def case_state(case_id: str) -> dict[str, Any]:
        if case_id not in cases:
            cases[case_id] = {
                "case_id": case_id,
                "as_of_event_id": "",
                "current_decision": None,
                "assumptions": {},
                "commitments": {},
                "latest_outcome": None,
                "judgment_asset_evaluations": [],
                "event_count": 0,
                "timeline": [],
            }
        return cases[case_id]

    for event in active:
        event_type = str(event.get("event_type") or "")
        if event_type == "event_corrected":
            continue
        aggregate = event.get("aggregate") or {}
        if aggregate.get("type") != "case":
            continue
        case_id = str(aggregate.get("id") or "")
        state = case_state(case_id)
        transition = event.get("transition") or {}
        action = str(transition.get("action") or "")
        actor = event.get("actor") or {}
        context = event.get("context") or {}
        state["as_of_event_id"] = event.get("event_id")
        state["event_count"] += 1
        state["timeline"].append(
            {
                "event_id": event.get("event_id"),
                "occurred_at": event.get("occurred_at"),
                "event_type": event_type,
                "action": action,
                "subject": transition.get("subject"),
                "reason_summary": context.get("reason_summary"),
                "actor_type": actor.get("type"),
                "from": transition.get("from"),
                "to": transition.get("to"),
                "evidence_refs": event.get("evidence_refs") or [],
            }
        )

        if event_type == "decision_changed":
            if action == "finalized" and actor.get("type") != "human":
                findings.append({"case_id": case_id, "event_id": str(event.get("event_id")), "issue": "non_human_final"})
                continue
            target = transition.get("to") or {}
            state["current_decision"] = {
                **(target if isinstance(target, dict) else {"value": target}),
                "status": "final" if action == "finalized" else action,
                "decided_by": actor.get("type"),
                "reason_summary": context.get("reason_summary"),
                "event_id": event.get("event_id"),
            }
        elif event_type == "assumption_changed":
            subject = str(transition.get("subject") or "")
            state["assumptions"][subject] = {
                "id": subject,
                "status": action,
                "value": transition.get("to"),
                "reason_summary": context.get("reason_summary"),
                "event_id": event.get("event_id"),
            }
        elif event_type == "commitment_changed":
            subject = str(transition.get("subject") or "")
            state["commitments"][subject] = {
                "id": subject,
                "status": action,
                "value": transition.get("to"),
                "reason_summary": context.get("reason_summary"),
                "event_id": event.get("event_id"),
            }
        elif event_type == "outcome_recorded":
            state["latest_outcome"] = {
                "status": action,
                "value": transition.get("to"),
                "reason_summary": context.get("reason_summary"),
                "event_id": event.get("event_id"),
            }
        elif event_type == "judgment_asset_evaluated":
            state["judgment_asset_evaluations"].append(
                {
                    "rule_id": (event.get("payload") or {}).get("rule_id"),
                    "outcome": action,
                    "event_id": event.get("event_id"),
                    "occurred_at": event.get("occurred_at"),
                }
            )

    for state in cases.values():
        state["assumptions"] = list(state["assumptions"].values())
        all_commitments = list(state["commitments"].values())
        state["open_commitments"] = [
            item for item in all_commitments if item.get("status") in {"opened", "assigned", "overdue"}
        ]
        state["commitment_history"] = all_commitments
        del state["commitments"]

    return {
        "schema_version": "1.0",
        "source": "decision_state_ledger",
        "summary": {
            "valid_unique_events": len(unique),
            "active_events": len(active),
            "corrected_events": len(corrected),
            "case_count": len(cases),
            "finding_count": len(findings),
        },
        "findings": findings,
        "cases": [cases[key] for key in sorted(cases)],
    }


def write_projection(path: Path, projection: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(projection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def render_audit_report(projection: dict[str, Any], audit: dict[str, Any]) -> str:
    summary = projection.get("summary") or {}
    cases = projection.get("cases") or []
    open_commitments = sum(len(case.get("open_commitments") or []) for case in cases)
    finalized = sum(
        1 for case in cases if (case.get("current_decision") or {}).get("status") == "final"
    )
    without_decision = sum(1 for case in cases if not case.get("current_decision"))
    issue_counts = Counter(str(item.get("issue") or "unknown") for item in projection.get("findings") or [])
    lines = [
        "# Judgment State Ledger Audit",
        "",
        "## Summary",
        "",
        f"- Valid events: {audit.get('valid_events', 0)}",
        f"- Invalid lines: {audit.get('invalid_lines', 0)}",
        f"- Duplicate event IDs: {audit.get('duplicate_event_ids', 0)}",
        f"- Duplicate idempotency keys: {audit.get('duplicate_idempotency_keys', 0)}",
        f"- Orphan corrections: {audit.get('orphan_corrections', 0)}",
        f"- Cases: {summary.get('case_count', 0)}",
        f"- Cases with a finalized human decision: {finalized}",
        f"- Cases without a decision event: {without_decision}",
        f"- Open commitments: {open_commitments}",
        "",
        "## Event Types",
        "",
    ]
    for event_type, count in (audit.get("event_type_counts") or {}).items():
        lines.append(f"- {event_type}: {count}")
    if not audit.get("event_type_counts"):
        lines.append("- No events yet")
    lines.extend(["", "## Findings", ""])
    if issue_counts:
        for issue, count in sorted(issue_counts.items()):
            lines.append(f"- {issue}: {count}")
    else:
        lines.append("- No projection findings")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This report is observation-only. It does not change screening scores, prompts, decisions, or judgment-asset promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = audit_ledger(args.ledger)
    projection = project_events(load_events(args.ledger))
    write_projection(args.output, projection)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_audit_report(projection, audit), encoding="utf-8")
    print(f"events={audit['valid_events']}")
    print(f"cases={projection['summary']['case_count']}")
    print(f"output={args.output}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
