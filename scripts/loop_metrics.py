#!/usr/bin/env python3
"""Build a read-only loop engineering health report from existing artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompt_feedback_metrics import DEFAULT_LOG_PATH, build_summary as build_prompt_summary, load_jsonl

REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_LATEST_REPORT = REPORTS_DIR / "latest.json"
DEFAULT_RECURSIVE_REPORT = REPORTS_DIR / "recursive_self_improvement_latest.json"
DEFAULT_OUTPUT_JSON = REPORTS_DIR / "loop_engineering_latest.json"
DEFAULT_OUTPUT_MD = REPORTS_DIR / "loop_engineering_latest.md"


def _load_json(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.exists():
        return {}, False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, False
    return (data, True) if isinstance(data, dict) else ({}, False)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _percent(part: int | float, total: int | float) -> float:
    return round(float(part) / float(total) * 100, 1) if total else 0.0


def build_loop_metrics(
    *,
    latest_report_path: Path = DEFAULT_LATEST_REPORT,
    recursive_report_path: Path = DEFAULT_RECURSIVE_REPORT,
    prompt_log_path: Path = DEFAULT_LOG_PATH,
) -> dict[str, Any]:
    latest_report, latest_available = _load_json(latest_report_path)
    recursive_report, recursive_available = _load_json(recursive_report_path)
    prompt_rows = load_jsonl(prompt_log_path)
    prompt_available = prompt_log_path.exists()
    prompt_summary = build_prompt_summary(prompt_rows)

    applied_count = _safe_int(latest_report.get("applied_count"))
    needs_review_count = _safe_int(latest_report.get("needs_review_count"))
    failed_count = _safe_int(latest_report.get("failed_count"))
    improvement_total = applied_count + needs_review_count + failed_count

    recursive_measurement = recursive_report.get("measurement_summary") or {}
    canonical_count = _safe_int(recursive_report.get("canonical_candidate_count"))
    ranked_queue_count = _safe_int(recursive_report.get("ranked_queue_count"))
    suppressed_count = _safe_int(recursive_report.get("suppressed_count"))

    available_sources = sum([latest_available, recursive_available, prompt_available])
    status = "ok"
    recommendations: list[str] = []

    if not latest_available:
        status = "attention"
        recommendations.append("reports/latest.json を生成して改善候補ループの正本を確認する")
    if not recursive_available:
        if status == "ok":
            status = "warn"
        recommendations.append("日次改善パイプライン後に recursive_self_improvement_latest.json を生成する")
    if prompt_available and not prompt_rows:
        if status == "ok":
            status = "warn"
        recommendations.append("prompt feedback log は存在するが空のため、AI応答改善ループの観測が不足している")
    if needs_review_count >= 25:
        if status == "ok":
            status = "warn"
        recommendations.append("needs_review が多いため、低リスク候補と高リスク候補を分けて棚卸しする")
    if _safe_float(recursive_measurement.get("noise_rate")) >= 50.0:
        if status == "ok":
            status = "warn"
        recommendations.append("noise_rate が高いため、重複候補と抑制ルールを確認する")
    if not recommendations:
        recommendations.append("現状は読み取り専用の定点観測を継続する")

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "sources": {
            "latest_report": {"path": str(latest_report_path), "available": latest_available},
            "recursive_report": {"path": str(recursive_report_path), "available": recursive_available},
            "prompt_feedback_log": {
                "path": str(prompt_log_path),
                "available": prompt_available,
                "rows": len(prompt_rows),
            },
        },
        "health": {
            "source_coverage_rate": _percent(available_sources, 3),
            "review_pressure_rate": _percent(needs_review_count, improvement_total),
            "auto_application_rate": _percent(applied_count, applied_count + needs_review_count),
            "recursive_queue_rate": _percent(ranked_queue_count, canonical_count),
            "recursive_suppression_rate": _percent(suppressed_count, canonical_count),
        },
        "improvement_loop": {
            "applied_count": applied_count,
            "needs_review_count": needs_review_count,
            "failed_count": failed_count,
            "total": improvement_total,
        },
        "recursive_loop": {
            "canonical_candidate_count": canonical_count,
            "ranked_queue_count": ranked_queue_count,
            "suppressed_count": suppressed_count,
            "measurement_summary": {
                "pdca_rate": _safe_float(recursive_measurement.get("pdca_rate")),
                "response_changed_rate": _safe_float(recursive_measurement.get("response_changed_rate")),
                "repeat_issue_rate": _safe_float(recursive_measurement.get("repeat_issue_rate")),
                "reuse_rate": _safe_float(recursive_measurement.get("reuse_rate")),
                "noise_rate": _safe_float(recursive_measurement.get("noise_rate")),
            },
        },
        "prompt_feedback_loop": {
            "total": _safe_int(prompt_summary.get("total")),
            "pdca_count": _safe_int(prompt_summary.get("pdca_count")),
            "pdca_rate": _safe_float(prompt_summary.get("pdca_rate")),
            "previous_diff_count": _safe_int(prompt_summary.get("previous_diff_count")),
            "previous_diff_rate": _safe_float(prompt_summary.get("previous_diff_rate")),
            "surface_counts": prompt_summary.get("surface_counts") or {},
        },
        "recommendations": recommendations,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Loop Engineering Health")
    lines.append("")
    lines.append(f"- Generated at: `{report['generated_at']}`")
    lines.append(f"- Status: `{report['status']}`")
    lines.append(f"- Source coverage: {report['health']['source_coverage_rate']}%")
    lines.append("")
    lines.append("## Improvement Loop")
    improvement = report["improvement_loop"]
    lines.append(f"- Applied: {improvement['applied_count']}")
    lines.append(f"- Needs review: {improvement['needs_review_count']}")
    lines.append(f"- Failed: {improvement['failed_count']}")
    lines.append(f"- Review pressure: {report['health']['review_pressure_rate']}%")
    lines.append("")
    lines.append("## Recursive Loop")
    recursive = report["recursive_loop"]
    measurement = recursive["measurement_summary"]
    lines.append(f"- Canonical candidates: {recursive['canonical_candidate_count']}")
    lines.append(f"- Ranked queue: {recursive['ranked_queue_count']}")
    lines.append(f"- Suppressed: {recursive['suppressed_count']}")
    lines.append(f"- Repeat issue rate: {measurement['repeat_issue_rate']}%")
    lines.append(f"- Reuse rate: {measurement['reuse_rate']}%")
    lines.append(f"- Noise rate: {measurement['noise_rate']}%")
    lines.append("")
    lines.append("## Prompt Feedback Loop")
    prompt = report["prompt_feedback_loop"]
    lines.append(f"- Total: {prompt['total']}")
    lines.append(f"- PDCA applied: {prompt['pdca_count']} ({prompt['pdca_rate']}%)")
    lines.append(f"- Previous response diffs: {prompt['previous_diff_count']} ({prompt['previous_diff_rate']}%)")
    lines.append("")
    lines.append("## Recommendations")
    for item in report["recommendations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest-report", type=Path, default=DEFAULT_LATEST_REPORT)
    parser.add_argument("--recursive-report", type=Path, default=DEFAULT_RECURSIVE_REPORT)
    parser.add_argument("--prompt-log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = build_loop_metrics(
        latest_report_path=args.latest_report.expanduser(),
        recursive_report_path=args.recursive_report.expanduser(),
        prompt_log_path=args.prompt_log.expanduser(),
    )
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    write_outputs(
        report,
        output_json=args.output_json.expanduser(),
        output_md=args.output_md.expanduser(),
    )
    print(f"saved: {args.output_json.expanduser()}")
    print(f"saved: {args.output_md.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
