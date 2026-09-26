#!/usr/bin/env python3
"""Summarize Jev(TypeSafe) shadow-mode agreement for Obsidian curation/reduction.

Reads the jsonl logs written by:
  - api/shion_obsidian_curator.py (TYPESAFE_CURATION_MODE=shadow)
  - scripts/analyze_rag_staleness.py (TYPESAFE_STALENESS_MODE=shadow)

Read-only: does not touch the Vault, ChromaDB, or the ledger. Existence of
this report is the basis for deciding whether Jev is later allowed to
influence curator auto-actions or staleness reduction candidates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CURATION_LOG = PROJECT_ROOT / "data" / "obsidian_curation_shadow_log.jsonl"
DEFAULT_STALENESS_LOG = PROJECT_ROOT / "data" / "rag_staleness_shadow_log.jsonl"


def read_entries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            rows.append(entry)
    return rows


def summarize_curation(entries: list[dict[str, Any]]) -> dict[str, Any]:
    applied = [e for e in entries if e.get("status") != "fallback"]
    fallback = [e for e in entries if e.get("status") == "fallback"]
    include_any = sum(1 for e in applied if "include" in (e.get("jev_routes") or []))
    return {
        "samples": len(entries),
        "applied_count": len(applied),
        "fallback_count": len(fallback),
        "proposals_with_a_related_hub": include_any,
        "related_hub_rate": round(include_any / len(applied), 3) if applied else 0.0,
    }


def summarize_staleness(entries: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [e for e in entries if e.get("status") != "fallback" and e.get("jev_route")]
    fallback = [e for e in entries if e.get("status") == "fallback"]
    agree = [e for e in judged if e.get("agreement") is True]
    disagree = [e for e in judged if e.get("agreement") is False]
    return {
        "samples": len(entries),
        "judged_count": len(judged),
        "fallback_count": len(fallback),
        "agree_count": len(agree),
        "disagree_count": len(disagree),
        "agreement_rate": round(len(agree) / len(judged), 3) if judged else 0.0,
    }


def render(curation: dict[str, Any], staleness: dict[str, Any]) -> str:
    return (
        "# Jev (TypeSafe) Obsidian Curation Shadow Report\n\n"
        "## Curator: connect_used_isolate relatedness\n\n"
        f"- Samples: {curation['samples']} (applied={curation['applied_count']}, "
        f"fallback={curation['fallback_count']})\n"
        f"- Jev found a related hub for {curation['proposals_with_a_related_hub']} proposals "
        f"({curation['related_hub_rate'] * 100:.1f}%)\n\n"
        "## Staleness: reduction candidate agreement\n\n"
        f"- Samples: {staleness['samples']} (judged={staleness['judged_count']}, "
        f"fallback={staleness['fallback_count']})\n"
        f"- Agreement with keyword heuristic: {staleness['agree_count']}/{staleness['judged_count']} "
        f"({staleness['agreement_rate'] * 100:.1f}%)\n"
        f"- Disagreement: {staleness['disagree_count']}\n\n"
        "agreement_rate が十分高くない限り、Jevの判定を自動実行（curatorのリンク追加・"
        "ChromaDB削減候補の確定）へ昇格させるべきではない。\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curation-log", default=str(DEFAULT_CURATION_LOG))
    parser.add_argument("--staleness-log", default=str(DEFAULT_STALENESS_LOG))
    parser.add_argument("--json", action="store_true", help="Markdown ではなく JSON で出力")
    args = parser.parse_args()

    curation = summarize_curation(read_entries(Path(args.curation_log)))
    staleness = summarize_staleness(read_entries(Path(args.staleness_log)))
    if args.json:
        print(json.dumps({"curation": curation, "staleness": staleness}, ensure_ascii=False, indent=2))
    else:
        print(render(curation, staleness))


if __name__ == "__main__":
    main()
