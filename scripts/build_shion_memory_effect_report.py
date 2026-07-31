#!/usr/bin/env python3
"""Build an effect report for Shion memories.

The report explains which memory layers are being used, which memories are
stale/revised, and which records need review. It is observational only: no
prompt, scoring, RAG rank, Obsidian, or judgment asset store is changed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_ROOT / "data" / "shion_memory_index.json"
DEFAULT_USAGE_LOG = PROJECT_ROOT / "data" / "shion_memory_usage_log.jsonl"
DEFAULT_REVISIONS = PROJECT_ROOT / "data" / "shion_memory_revisions.jsonl"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "shion_memory_effect_latest.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "shion_memory_effect_latest.md"
GUARDRAIL = "observability_only_no_prompt_no_rag_rank_no_scoring_no_auto_promotion"


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


def build_report(index: dict[str, Any], usage_rows: list[dict[str, Any]], revision_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    records = [r for r in index.get("records") or [] if isinstance(r, dict)]
    by_id = {str(r.get("id") or ""): r for r in records if r.get("id")}
    usage_by_id: Counter[str] = Counter()
    layer_usage: Counter[str] = Counter()
    type_usage: Counter[str] = Counter()
    latest_question: dict[str, str] = {}
    impact_hint_count = 0

    for row in usage_rows:
        refs = [str(ref) for ref in row.get("refs") or [] if ref]
        if row.get("impact_hints"):
            impact_hint_count += 1
        for ref in refs:
            usage_by_id[ref] += 1
            record = by_id.get(ref) or {}
            layer_usage[str(record.get("memory_layer") or "unknown")] += 1
            type_usage[str(record.get("memory_type") or "unknown")] += 1
            question = str(row.get("question") or "").strip()
            if question:
                latest_question[ref] = question

    status_counts = Counter(str(r.get("status") or "active") for r in records)
    layer_counts = Counter(str(r.get("memory_layer") or "unknown") for r in records)
    unused_persistent = [
        _record_summary(r, usage_by_id, latest_question)
        for r in records
        if str(r.get("memory_layer") or "") == "persistent" and usage_by_id[str(r.get("id") or "")] == 0
    ]
    review_candidates = [
        _record_summary(r, usage_by_id, latest_question)
        for r in records
        if str(r.get("status") or "") in {"stale", "revised"}
    ][:30]
    top_used = [
        _record_summary(by_id[rid], usage_by_id, latest_question)
        for rid, _count in usage_by_id.most_common(20)
        if rid in by_id
    ]
    revisions = revision_rows or []
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "guardrail": GUARDRAIL,
        "summary": {
            "records": len(records),
            "usage_events": len(usage_rows),
            "used_memory_ids": len(usage_by_id),
            "usage_events_with_impact_hints": impact_hint_count,
            "by_layer": dict(sorted(layer_counts.items())),
            "usage_by_layer": dict(sorted(layer_usage.items())),
            "usage_by_type": dict(sorted(type_usage.items())),
            "by_status": dict(sorted(status_counts.items())),
            "revision_events": len(revisions),
        },
        "top_used": top_used,
        "review_candidates": review_candidates,
        "unused_persistent": unused_persistent,
        "next_actions": [
            "usage_events_with_impact_hints が増えるほど、回答がどの記憶に影響されたか追跡しやすい。",
            "stale/revised は削除せず、必要なら scripts/revise_shion_memory.py で後継記憶を登録する。",
            "unused_persistent は強い原則なのに使われていないため、強すぎる/不要/参照条件が狭すぎる可能性を見る。",
        ],
    }


def _record_summary(record: dict[str, Any], usage_by_id: Counter[str], latest_question: dict[str, str]) -> dict[str, Any]:
    rid = str(record.get("id") or "")
    return {
        "id": rid,
        "memory_layer": str(record.get("memory_layer") or ""),
        "memory_type": str(record.get("memory_type") or ""),
        "status": str(record.get("status") or "active"),
        "source_path": str(record.get("source_path") or ""),
        "used_count": int(usage_by_id.get(rid, 0)),
        "last_used_at": str(record.get("last_used_at") or ""),
        "latest_question": latest_question.get(rid, ""),
        "content": " ".join(str(record.get("content") or "").split())[:180],
    }


def markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Shion Memory Effect Report",
        "",
        f"- Guardrail: `{report.get('guardrail')}`",
        f"- Records: {summary.get('records', 0)}",
        f"- Usage events: {summary.get('usage_events', 0)}",
        f"- Used memory ids: {summary.get('used_memory_ids', 0)}",
        f"- Impact-hint events: {summary.get('usage_events_with_impact_hints', 0)}",
        f"- Usage by layer: {summary.get('usage_by_layer', {})}",
        "",
        "## Top Used",
    ]
    for item in report.get("top_used") or []:
        lines.append(f"- {item.get('id')} [{item.get('memory_layer')}/{item.get('status')}] used={item.get('used_count')} {item.get('content')}")
    lines.extend(["", "## Review Candidates"])
    for item in report.get("review_candidates") or []:
        lines.append(f"- {item.get('id')} [{item.get('status')}] {item.get('content')}")
    lines.extend(["", "## Unused Persistent"])
    for item in report.get("unused_persistent") or []:
        lines.append(f"- {item.get('id')} {item.get('content')}")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--usage-log", type=Path, default=DEFAULT_USAGE_LOG)
    parser.add_argument("--revisions", type=Path, default=DEFAULT_REVISIONS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = build_report(_read_json(args.index), _read_jsonl(args.usage_log), _read_jsonl(args.revisions))
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
