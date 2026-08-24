#!/usr/bin/env python3
"""Build Shion Memory Sentinel report.

This is an observability-only consolidation layer over the existing memory
reports. It does not promote, delete, rewrite, rerank, or change prompts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.shion_memory_system_audit import run_shion_memory_sentinel

DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports" / "shion_memory_sentinel_latest.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "shion_memory_sentinel_latest.md"


def markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Shion Memory Sentinel",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Guardrail: `{report.get('guardrail')}`",
        f"- Memory records: {summary.get('memory_records', 0)}",
        f"- Long-term domain coverage: {summary.get('long_term_domain_coverage', 0)}",
        f"- Long-term use_when coverage: {summary.get('long_term_use_when_coverage', 0)}",
        f"- Usage events: {summary.get('usage_events', 0)}",
        f"- Used memory ids: {summary.get('used_memory_ids', 0)}",
        f"- Likely helpful: {summary.get('likely_helpful_memory_ids', 0)}",
        f"- Needs feedback: {summary.get('needs_feedback_memory_ids', 0)}",
        f"- Possible noise: {summary.get('possible_noise_memory_ids', 0)}",
        f"- Open human reviews: {summary.get('open_human_review_records', 0)}",
        f"- Open human review batches: {summary.get('open_human_review_batches', 0)}",
        f"- Contradiction candidates: {summary.get('contradiction_candidates', 0)}",
        "",
        "## Source Reports",
    ]
    for source in report.get("source_reports") or []:
        lines.append(
            f"- `{source.get('path')}` status={source.get('status')} generated_at={source.get('generated_at') or '-'}"
        )

    lines.extend(["", "## Watch Signals"])
    signals = report.get("signals") or []
    if signals:
        for signal in signals:
            lines.append(
                f"- `{signal.get('level')}` {signal.get('area')}: {signal.get('metric')} - {signal.get('reason')}"
            )
    else:
        lines.append("- No active memory sentinel signal.")

    lines.extend(["", "## Feedback Triage"])
    triage = report.get("feedback_triage") or {}
    batches = triage.get("top_batches") or []
    if batches:
        for batch in batches:
            lines.append(
                f"- `{batch.get('domain')}` {batch.get('memory_layer')}/{batch.get('memory_type')}: "
                f"{batch.get('count')} records, used={batch.get('used_count')}"
            )
            for sample in batch.get("samples") or []:
                lines.append(f"  - `{sample.get('id')}` used={sample.get('used_count')} {sample.get('content')}")
    else:
        lines.append("- No needs-feedback batches.")

    lines.extend(["", "## Next Actions"])
    actions = report.get("next_actions") or []
    if actions:
        for action in actions:
            lines.append(f"- {action}")
    else:
        lines.append("- No immediate action.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = run_shion_memory_sentinel(limit=args.limit)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(report), encoding="utf-8")
    print(
        "Shion Memory Sentinel: "
        f"status={report['status']} signals={report['signal_count']} -> {args.output_md}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
