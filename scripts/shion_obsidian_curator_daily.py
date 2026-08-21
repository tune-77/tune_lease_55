#!/usr/bin/env python3
"""Build Shion ADK Obsidian Curator daily report.

This is a read-only daily diagnostic. It calls the same bounded tools that are
registered on Shion's ADK agent and writes report artifacts only:

- reports/shion_obsidian_curator_daily_latest.json
- reports/shion_obsidian_curator_daily_latest.md

It does not edit Obsidian, rebuild ChromaDB, change prompts, or apply links.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api.shion_obsidian_curator import (  # noqa: E402
    review_obsidian_vault_health,
    suggest_obsidian_curation_actions,
)

DEFAULT_OUTPUT_JSON = _REPO_ROOT / "reports" / "shion_obsidian_curator_daily_latest.json"
DEFAULT_OUTPUT_MD = _REPO_ROOT / "reports" / "shion_obsidian_curator_daily_latest.md"
DEFAULT_THEMES = (
    "補助金 未採択",
    "条件付き承認 違和感",
    "残価 再販リスク",
)


def _clean(value: Any, limit: int = 220) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "..."
    return text


def build_daily_report(*, report_date: str, themes: list[str], limit: int) -> dict[str, Any]:
    health = review_obsidian_vault_health(limit=limit)
    theme_reports = [
        suggest_obsidian_curation_actions(theme, limit=min(3, limit))
        for theme in themes
        if str(theme or "").strip()
    ]

    top_actions: list[dict[str, Any]] = []
    for proposal in health.get("proposals") or []:
        if not isinstance(proposal, dict):
            continue
        top_actions.append({
            "source": "vault_health",
            "type": proposal.get("type"),
            "target": proposal.get("path") or proposal.get("directory"),
            "reason": proposal.get("reason"),
            "suggested_action": proposal.get("suggested_action") or "人間レビュー後に少数リンク/検索語だけ追加",
            "requires_human_approval": True,
        })
        if len(top_actions) >= limit:
            break

    for theme_report in theme_reports:
        theme = theme_report.get("theme")
        for suggestion in theme_report.get("suggestions") or []:
            if not isinstance(suggestion, dict):
                continue
            top_actions.append({
                "source": "theme",
                "theme": theme,
                "type": "theme_curation_candidate",
                "target": suggestion.get("path"),
                "reason": "検索/チャットで拾えない時の候補",
                "suggested_links": suggestion.get("suggested_links") or [],
                "suggested_search_terms": suggestion.get("suggested_search_terms") or [],
                "requires_human_approval": True,
            })
            if len(top_actions) >= limit:
                break
        if len(top_actions) >= limit:
            break

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date": report_date,
        "mode": "read_only_daily_obsidian_curator",
        "guardrail": "read_only_no_vault_write_no_chroma_reindex_no_prompt_change",
        "health": health,
        "theme_reports": theme_reports,
        "top_actions": top_actions[:limit],
    }


def markdown(report: dict[str, Any]) -> str:
    health = report.get("health") if isinstance(report.get("health"), dict) else {}
    summary = health.get("summary") if isinstance(health.get("summary"), dict) else {}
    lines = [
        "# Shion Obsidian Curator Daily",
        "",
        f"- Date: `{report.get('date')}`",
        f"- Mode: `{report.get('mode')}`",
        f"- Guardrail: `{report.get('guardrail')}`",
        f"- Graph buckets: {summary.get('graph_buckets', {})}",
        f"- Degree 0 count: {summary.get('degree0_count', 0)}",
        f"- Top degree 0 dirs: {summary.get('top_degree0_dirs', {})}",
        "",
        "## Top Actions",
    ]
    actions = report.get("top_actions") or []
    if not actions:
        lines.append("- No action. Keep observing.")
    for item in actions:
        target = _clean(item.get("target"))
        reason = _clean(item.get("reason"))
        theme = f" theme={_clean(item.get('theme'), 80)}" if item.get("theme") else ""
        links = item.get("suggested_links") or []
        terms = item.get("suggested_search_terms") or []
        lines.append(f"- `{target}`{theme}: {reason}")
        if links:
            lines.append(f"  - suggested_links: {', '.join(_clean(v, 80) for v in links[:3])}")
        if terms:
            lines.append(f"  - suggested_search_terms: {', '.join(_clean(v, 60) for v in terms[:5])}")

    lines.extend(["", "## Theme Checks"])
    for theme_report in report.get("theme_reports") or []:
        theme = _clean(theme_report.get("theme"), 120)
        status = _clean(theme_report.get("status"), 80)
        suggestions = theme_report.get("suggestions") or []
        lines.append(f"### {theme}")
        lines.append(f"- status: `{status}` / suggestions={len(suggestions)}")
        for suggestion in suggestions[:3]:
            lines.append(
                f"- `{_clean(suggestion.get('path'), 160)}` score={suggestion.get('route_score')} "
                f"degree={suggestion.get('degree')}"
            )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--theme", action="append", dest="themes", default=[])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args(argv)

    themes = args.themes or list(DEFAULT_THEMES)
    report = build_daily_report(report_date=args.date, themes=themes, limit=max(1, min(10, args.limit)))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(report), encoding="utf-8")
    print(f"wrote={args.output_json}")
    print(f"wrote={args.output_md}")
    print(f"top_actions={len(report.get('top_actions') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
