#!/usr/bin/env python3
"""Run answer generation, local RAG improvement, and trusted Web fallback."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.auto_improve_answer_quality import (
    generate_retry_answer,
    research_failed_answer,
    retrieve_retry_hits,
    run_auto_improvement,
    run_web_fallback,
)
from scripts.evaluate_answer_quality import (
    DEFAULT_EVAL_SET,
    evaluate_answers,
    generate_answers,
)

DEFAULT_REPORT = REPO_ROOT / "reports" / "answer_quality_pipeline_latest.json"


def run_pipeline(
    *,
    cases: list[dict[str, Any]],
    generate_initial: Callable[[list[dict[str, Any]]], dict[str, dict[str, Any]]],
    improve_answer: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    web_improve_answer: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    max_iterations: int = 3,
    enable_web_fallback: bool = True,
) -> dict[str, Any]:
    initial_answers = generate_initial(cases)
    initial = evaluate_answers(cases, initial_answers)
    improvement = run_auto_improvement(
        cases=cases,
        initial_answers=initial_answers,
        improve_answer=improve_answer,
        max_iterations=max(1, max_iterations),
    )
    local_final = improvement["final"]
    web_used = False
    if (
        enable_web_fallback
        and local_final["passed"] < local_final["total"]
    ):
        improvement = run_web_fallback(
            cases=cases,
            report=improvement,
            web_improve_answer=web_improve_answer,
        )
        web_used = True

    final = improvement["final"]
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": (
            "passed"
            if final["passed"] == final["total"]
            else "needs_review"
        ),
        "web_fallback_enabled": enable_web_fallback,
        "web_fallback_used": web_used,
        "initial": initial,
        "improvement": improvement,
        "final": final,
    }


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-web-fallback", action="store_true")
    args = parser.parse_args()

    cases = json.loads(args.eval_set.read_text(encoding="utf-8"))

    from api.chat_memory import call_gemini_chat
    from api.knowledge.vector_store import get_store

    store = get_store()

    def generate_initial(
        target_cases: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        return generate_answers(
            target_cases,
            search_fn=lambda query, limit: store.search(query, top_k=limit),
            chat_fn=call_gemini_chat,
        )

    def improve_answer(
        case: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        hits = retrieve_retry_hits(
            case,
            result,
            lambda query, limit: store.search(query, top_k=limit),
            top_k=args.top_k,
        )
        return generate_retry_answer(case, result, hits, call_gemini_chat)

    report = run_pipeline(
        cases=cases,
        generate_initial=generate_initial,
        improve_answer=improve_answer,
        web_improve_answer=research_failed_answer,
        max_iterations=args.max_iterations,
        enable_web_fallback=not args.no_web_fallback,
    )
    report["eval_set"] = str(args.eval_set)
    _atomic_write_json(args.report, report)

    initial = report["initial"]
    final = report["final"]
    print(
        "[answer-quality-pipeline] "
        f"status={report['status']} "
        f"initial={initial['passed']}/{initial['total']} "
        f"final={final['passed']}/{final['total']} "
        f"avg={final['average_score']} "
        f"concept={final['concept_coverage']}% "
        f"forbidden={final['forbidden_cases']} "
        f"web={report['web_fallback_used']}"
    )
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
