#!/usr/bin/env python3
"""Generate a monthly prompt-feedback report.

The report is written to `reports/` and, if the iCloud Obsidian Vault is
available, appended to `Projects/tune_lease_55/AI Chat/Monthly Review/`.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from prompt_feedback_metrics import DEFAULT_LOG_PATH, build_summary, load_jsonl, render_markdown

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_LATEST_MD = REPORTS_DIR / "prompt_feedback_monthly_latest.md"
DEFAULT_LATEST_JSON = REPORTS_DIR / "prompt_feedback_monthly_latest.json"


def _month_key() -> str:
    today = date.today()
    return f"{today.year:04d}-{today.month:02d}"


def _surface_lines(summary: dict) -> list[str]:
    rows: list[str] = []
    for surface, stats in sorted(summary.get("by_surface", {}).items(), key=lambda kv: (-kv[1]["count"], kv[0])):
        rows.append(
            f"- `{surface}`: {stats['count']}件, PDCA {stats['pdca_rate']}%, "
            f"変化率 {stats['response_changed_rate']}%, 平均応答長 {stats['avg_response_len']}"
        )
    return rows


def _obsidian_body(summary: dict, source: Path) -> str:
    lines = [
        f"## 月次サマリー ({_month_key()})",
        "",
        f"- Source: `{source}`",
        f"- Total entries: {summary['total']}",
        f"- PDCA applied: {summary['pdca_count']} ({summary['pdca_rate']}%)",
        f"- Previous-response diffs: {summary['previous_diff_count']} ({summary['previous_diff_rate']}%)",
        f"- Avg response length: {summary['avg_response_len']}",
        f"- Avg prompt length: base {summary['avg_prompt_base_len']} -> final {summary['avg_prompt_final_len']}",
        "",
        "### Surfaces",
        *(_surface_lines(summary) or ["- No entries"]),
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a monthly prompt-feedback report.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH, help="Prompt feedback JSONL log path.")
    parser.add_argument("--output-md", type=Path, default=None, help="Markdown output path.")
    parser.add_argument("--output-json", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--obsidian", action="store_true", help="Also append to Obsidian monthly review note.")
    parser.add_argument("--stdout", action="store_true", help="Print markdown to stdout.")
    args = parser.parse_args()

    log_path = args.log.expanduser()
    rows = load_jsonl(log_path)
    summary = build_summary(rows)
    month = _month_key()

    md_path = args.output_md or REPORTS_DIR / f"prompt_feedback_monthly_{month}.md"
    json_path = args.output_json or REPORTS_DIR / f"prompt_feedback_monthly_{month}.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown(summary, log_path)
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DEFAULT_LATEST_MD.write_text(markdown, encoding="utf-8")
    DEFAULT_LATEST_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    obsidian_result = {"status": "skipped", "reason": "not requested"}
    if args.obsidian:
        try:
            from mobile_app.obsidian_bridge import append_monthly_review_note

            obsidian_result = append_monthly_review_note(
                title=f"Prompt Feedback Monthly Report {month}",
                body=_obsidian_body(summary, log_path),
            )
        except Exception as exc:
            obsidian_result = {"status": "error", "reason": str(exc)}

    if args.stdout:
        print(markdown, end="")
    else:
        print(f"saved: {md_path}")
        print(f"saved: {json_path}")
        print(f"latest: {DEFAULT_LATEST_MD}")
        print(f"latest: {DEFAULT_LATEST_JSON}")
        print(f"obsidian: {obsidian_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
