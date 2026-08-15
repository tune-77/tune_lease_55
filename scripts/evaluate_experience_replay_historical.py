#!/usr/bin/env python3
"""Score historical Shion answers against the experience replay eval set.

This is the no-network fallback for experience replay evaluation. It evaluates
responses already stored in local prompt feedback logs, so it does not send
workspace context or user queries to an external LLM.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_answer_quality import evaluate_answers

DEFAULT_EVAL_SET = REPO_ROOT / "api" / "knowledge" / "experience_replay_eval_set.json"
DEFAULT_PROMPT_FEEDBACK = REPO_ROOT / "data" / "prompt_feedback_log.jsonl"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports" / "experience_replay_historical_quality_latest.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "experience_replay_historical_quality_latest.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def build_historical_answers(
    cases: list[dict[str, Any]],
    prompt_feedback_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows_by_hash = {
        str(row.get("question_hash") or ""): row
        for row in prompt_feedback_rows
        if str(row.get("question_hash") or "")
    }
    rows_by_question: dict[str, dict[str, Any]] = {}
    for row in prompt_feedback_rows:
        question = str(row.get("question") or "").strip()
        if question and question not in rows_by_question:
            rows_by_question[question] = row

    answers: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case.get("id") or "")
        row = rows_by_hash.get(str(case.get("_source_id") or ""))
        if row is None:
            row = rows_by_question.get(str(case.get("query") or "").strip())
        answer = ""
        if row is not None:
            answer = str(row.get("response_text") or row.get("response_preview") or "")
        answers[case_id] = {
            "answer": answer,
            "source_paths": [],
            "matched_historical_log": bool(row),
        }
    return answers


def build_report(
    *,
    cases: list[dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    eval_set_path: Path,
    prompt_feedback_path: Path,
) -> dict[str, Any]:
    summary = evaluate_answers(cases, answers)
    missing = [
        {"id": case.get("id"), "query": case.get("query")}
        for case in cases
        if not answers.get(str(case.get("id") or ""), {}).get("matched_historical_log")
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "historical_local_no_network",
        "eval_set": str(eval_set_path),
        "prompt_feedback_log": str(prompt_feedback_path),
        "missing_historical_answers": missing,
        "final": summary,
    }


def render_markdown(report: dict[str, Any]) -> str:
    final = report["final"]
    lines = [
        "# Experience Replay Historical Quality",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Mode: `{report['mode']}`",
        f"- Passed: {final['passed']}/{final['total']}",
        f"- Pass rate: {final['pass_rate']}%",
        f"- Average score: {final['average_score']}",
        f"- Concept coverage: {final['concept_coverage']}%",
        f"- Forbidden cases: {final['forbidden_cases']}",
        f"- Uncertainty misses: {final['uncertainty_misses']}",
        f"- Missing historical answers: {len(report['missing_historical_answers'])}",
        "",
        "## Failed Cases",
    ]
    failed = [case for case in final["cases"] if not case["passed"]]
    if not failed:
        lines.append("- None")
    for case in failed:
        missing = [
            "/".join(result["aliases"])
            for result in case.get("concept_results") or []
            if not result.get("matched")
        ]
        lines.append(
            f"- `{case['id']}` score={case['score']} "
            f"missing={'; '.join(missing) or '-'} "
            f"uncertainty_present={case['uncertainty_present']} / {case['query']}"
        )
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--prompt-feedback", type=Path, default=DEFAULT_PROMPT_FEEDBACK)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    cases = json.loads(args.eval_set.expanduser().read_text(encoding="utf-8"))
    answers = build_historical_answers(cases, read_jsonl(args.prompt_feedback.expanduser()))
    report = build_report(
        cases=cases,
        answers=answers,
        eval_set_path=args.eval_set.expanduser(),
        prompt_feedback_path=args.prompt_feedback.expanduser(),
    )
    write_outputs(report, output_json=args.output_json.expanduser(), output_md=args.output_md.expanduser())
    final = report["final"]
    print(
        "[experience-replay-historical] "
        f"passed={final['passed']}/{final['total']} "
        f"avg={final['average_score']} "
        f"concept={final['concept_coverage']}% "
        f"forbidden={final['forbidden_cases']} "
        f"uncertainty_miss={final['uncertainty_misses']} "
        f"missing={len(report['missing_historical_answers'])}"
    )
    return 0 if final["passed"] == final["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
