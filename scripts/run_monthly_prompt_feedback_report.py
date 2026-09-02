#!/usr/bin/env python3
"""Generate a monthly prompt-feedback report.

The report is written to `reports/` and, if the iCloud Obsidian Vault is
available, appended to `Projects/tune_lease_55/AI Chat/Monthly Review/`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompt_feedback_metrics import (
    DEFAULT_HUMAN_FEEDBACK_PATH,
    DEFAULT_LOG_PATH,
    build_human_feedback_summary,
    build_summary,
    filter_rows_by_month,
    load_jsonl,
    render_markdown,
)

REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_LATEST_MD = REPORTS_DIR / "prompt_feedback_monthly_latest.md"
DEFAULT_LATEST_JSON = REPORTS_DIR / "prompt_feedback_monthly_latest.json"


def _previous_month_key(today: date | None = None) -> str:
    current = today or date.today()
    previous = current.replace(day=1) - timedelta(days=1)
    return f"{previous.year:04d}-{previous.month:02d}"


def _validate_month(value: str) -> str:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        raise argparse.ArgumentTypeError("month must be YYYY-MM")
    return value


def _surface_lines(summary: dict) -> list[str]:
    rows: list[str] = []
    for surface, stats in sorted(summary.get("by_surface", {}).items(), key=lambda kv: (-kv[1]["count"], kv[0])):
        rows.append(
            f"- `{surface}`: {stats['count']}件, PDCA {stats['pdca_rate']}%, "
            f"変化率 {stats['response_changed_rate']}%, 平均応答長 {stats['avg_response_len']}"
        )
    return rows


def _obsidian_body(summary: dict, source: Path, month: str, human_feedback: dict, human_source: Path) -> str:
    lines = [
        f"## 月次サマリー ({month})",
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
        "",
        "### 回答ボタン評価",
        f"- Source: `{human_source}`",
        f"- 評価数: {human_feedback['usefulness_total']}件",
        f"- 効いた: {human_feedback['good_count']}件 ({human_feedback['useful_rate']}%)",
        f"- 微妙: {human_feedback['thin_count']}件",
        f"- 違う: {human_feedback['bad_count']}件",
        f"- 要改善率: {human_feedback['improvement_rate']}%",
    ]
    priorities = human_feedback.get("improvement_priorities") or []
    if priorities:
        lines.extend(["", "### 改善優先順位"])
        lines.extend(f"{index}. `{item['route']}` — {item['reason']}" for index, item in enumerate(priorities[:5], 1))
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a monthly prompt-feedback report.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH, help="Prompt feedback JSONL log path.")
    parser.add_argument("--human-feedback-log", type=Path, default=DEFAULT_HUMAN_FEEDBACK_PATH, help="Response button feedback JSONL log path.")
    parser.add_argument("--month", type=_validate_month, default=None, help="Target month (YYYY-MM). Defaults to the previous calendar month.")
    parser.add_argument("--output-md", type=Path, default=None, help="Markdown output path.")
    parser.add_argument("--output-json", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--obsidian", action="store_true", help="Also append to Obsidian monthly review note.")
    parser.add_argument("--stdout", action="store_true", help="Print markdown to stdout.")
    args = parser.parse_args()

    log_path = args.log.expanduser()
    month = args.month or _previous_month_key()
    rows = filter_rows_by_month(load_jsonl(log_path), month)
    summary = build_summary(rows)
    human_log_path = args.human_feedback_log.expanduser()
    human_rows = filter_rows_by_month(load_jsonl(human_log_path), month)
    human_summary = build_human_feedback_summary(human_rows)
    summary["report_month"] = month
    summary["human_response_feedback"] = human_summary

    md_path = args.output_md or REPORTS_DIR / f"prompt_feedback_monthly_{month}.md"
    json_path = args.output_json or REPORTS_DIR / f"prompt_feedback_monthly_{month}.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown(summary, log_path, human_summary, human_log_path)
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
                body=_obsidian_body(summary, log_path, month, human_summary, human_log_path),
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
