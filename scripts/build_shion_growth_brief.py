#!/usr/bin/env python3
"""Build a short daily growth brief for Shion.

This reads already-generated sidecar reports and writes a compact summary for
hands-off cultivation. It does not change memory, prompts, scoring, or judgment
assets.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_EFFECT = PROJECT_ROOT / "reports" / "shion_memory_effect_latest.json"
DEFAULT_JUDGMENT_AB = PROJECT_ROOT / "reports" / "judgment_asset_ab_latest.json"
DEFAULT_PERSISTENT_AUDIT = PROJECT_ROOT / "reports" / "persistent_memory_audit_latest.json"
DEFAULT_FIELD_REVIEW = PROJECT_ROOT / "reports" / "judgment_asset_field_review_latest.json"
DEFAULT_GRAPH_EFFECT = PROJECT_ROOT / "reports" / "obsidian_graph_judgment_effect_latest.json"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "shion_growth_brief_latest.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "shion_growth_brief_latest.md"
GUARDRAIL = "brief_only_no_auto_promotion_no_prompt_no_scoring_change"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def build_brief(
    *,
    target_date: str,
    memory_effect: dict[str, Any],
    judgment_ab: dict[str, Any],
    persistent_audit: dict[str, Any],
    field_review: dict[str, Any],
    graph_effect: dict[str, Any] | None = None,
) -> dict[str, Any]:
    memory_summary = memory_effect.get("summary") if isinstance(memory_effect.get("summary"), dict) else {}
    ab_summary = judgment_ab.get("summary") if isinstance(judgment_ab.get("summary"), dict) else {}
    audit_summary = persistent_audit.get("summary") if isinstance(persistent_audit.get("summary"), dict) else {}
    field_summary = field_review.get("summary") if isinstance(field_review.get("summary"), dict) else {}
    graph_effect = graph_effect or {}
    graph_summary = graph_effect.get("summary") if isinstance(graph_effect.get("summary"), dict) else {}
    top_used = memory_effect.get("top_used") if isinstance(memory_effect.get("top_used"), list) else []
    review_candidates = memory_effect.get("review_candidates") if isinstance(memory_effect.get("review_candidates"), list) else []
    ab_pairs = judgment_ab.get("pairs") if isinstance(judgment_ab.get("pairs"), list) else []

    actions: list[str] = []
    if int(audit_summary.get("findings") or 0):
        actions.append("永続記憶監査に指摘あり。PERSISTENT_MEMORY.md を確認する。")
    if int(ab_summary.get("candidate_pairs") or 0):
        actions.append("判断資産A/B候補あり。次の類似案件で helped/challenged を記録する。")
    if review_candidates:
        actions.append("stale/revised 記憶あり。必要なら revise_shion_memory.py で後継記憶を登録する。")
    if int(memory_summary.get("usage_events_with_impact_hints") or 0) == 0:
        actions.append("新しい回答から impact_hints が記録されるか確認する。")
    if int(graph_summary.get("buckets", {}).get("complex_but_unproven", 0) or 0):
        actions.append("Obsidianグラフに未検証の複雑ノードあり。効いたノードだけ入口化する。")
    if not actions:
        actions.append("今日は観測継続。強い自動昇格は不要。")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date": target_date,
        "guardrail": GUARDRAIL,
        "summary": {
            "memory_records": int(memory_summary.get("records") or 0),
            "used_memory_ids": int(memory_summary.get("used_memory_ids") or 0),
            "memory_usage_events": int(memory_summary.get("usage_events") or 0),
            "memory_impact_hint_events": int(memory_summary.get("usage_events_with_impact_hints") or 0),
            "judgment_asset_ab_pairs": int(ab_summary.get("candidate_pairs") or 0),
            "persistent_audit_findings": int(audit_summary.get("findings") or 0),
            "judgment_assets_grow": int(field_summary.get("grow") or 0),
            "judgment_assets_review": int(field_summary.get("review") or 0),
            "judgment_assets_sleeping": int(field_summary.get("sleeping") or 0),
            "obsidian_graph_effective_hubs": int(graph_summary.get("buckets", {}).get("effective_hub", 0) or 0),
            "obsidian_graph_complex_unproven": int(graph_summary.get("buckets", {}).get("complex_but_unproven", 0) or 0),
            "obsidian_graph_notes_with_usage_signal": int(graph_summary.get("notes_with_usage_signal") or 0),
        },
        "top_used_memories": top_used[:3],
        "memory_review_candidates": review_candidates[:3],
        "judgment_asset_ab_candidates": ab_pairs[:3],
        "obsidian_graph_answer": str(graph_effect.get("answer") or ""),
        "actions": actions,
    }


def markdown(brief: dict[str, Any]) -> str:
    summary = brief.get("summary") or {}
    lines = [
        "# Shion Growth Brief",
        "",
        f"- Date: {brief.get('date')}",
        f"- Guardrail: `{brief.get('guardrail')}`",
        f"- Memory: records={summary.get('memory_records', 0)}, used_ids={summary.get('used_memory_ids', 0)}, usage_events={summary.get('memory_usage_events', 0)}, impact_hints={summary.get('memory_impact_hint_events', 0)}",
        f"- Judgment assets: A/B pairs={summary.get('judgment_asset_ab_pairs', 0)}, grow={summary.get('judgment_assets_grow', 0)}, review={summary.get('judgment_assets_review', 0)}, sleeping={summary.get('judgment_assets_sleeping', 0)}",
        f"- Obsidian graph: effective_hubs={summary.get('obsidian_graph_effective_hubs', 0)}, complex_unproven={summary.get('obsidian_graph_complex_unproven', 0)}, usage_nodes={summary.get('obsidian_graph_notes_with_usage_signal', 0)}",
        f"- Graph answer: {brief.get('obsidian_graph_answer')}",
        f"- Persistent audit findings: {summary.get('persistent_audit_findings', 0)}",
        "",
        "## Actions",
    ]
    for action in brief.get("actions") or []:
        lines.append(f"- {action}")
    lines.extend(["", "## Top Used Memories"])
    for item in brief.get("top_used_memories") or []:
        lines.append(f"- {item.get('id')} [{item.get('memory_layer')}/{item.get('status')}] used={item.get('used_count')} {item.get('content')}")
    lines.extend(["", "## Judgment Asset A/B"])
    for item in brief.get("judgment_asset_ab_candidates") or []:
        a = item.get("asset_a") or {}
        b = item.get("asset_b") or {}
        lines.append(f"- {item.get('concept')}: {a.get('rule_id')} vs {b.get('rule_id')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--memory-effect", type=Path, default=DEFAULT_MEMORY_EFFECT)
    parser.add_argument("--judgment-ab", type=Path, default=DEFAULT_JUDGMENT_AB)
    parser.add_argument("--persistent-audit", type=Path, default=DEFAULT_PERSISTENT_AUDIT)
    parser.add_argument("--field-review", type=Path, default=DEFAULT_FIELD_REVIEW)
    parser.add_argument("--graph-effect", type=Path, default=DEFAULT_GRAPH_EFFECT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    brief = build_brief(
        target_date=args.date,
        memory_effect=_read_json(args.memory_effect),
        judgment_ab=_read_json(args.judgment_ab),
        persistent_audit=_read_json(args.persistent_audit),
        field_review=_read_json(args.field_review),
        graph_effect=_read_json(args.graph_effect),
    )
    if args.dry_run:
        print(json.dumps(brief, ensure_ascii=False, indent=2))
        return 0
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(brief, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(brief), encoding="utf-8")
    print(f"wrote={args.output_json}")
    print(f"wrote={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
