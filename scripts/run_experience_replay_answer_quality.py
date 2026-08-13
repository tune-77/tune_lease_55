#!/usr/bin/env python3
"""Run the experience replay eval set with per-case timeout handling."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_answer_quality import evaluate_answers, generate_answers

DEFAULT_EVAL_SET = REPO_ROOT / "api" / "knowledge" / "experience_replay_eval_set.json"
DEFAULT_REPORT = REPO_ROOT / "reports" / "experience_replay_answer_quality_latest.json"


class CaseTimeoutError(TimeoutError):
    pass


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise CaseTimeoutError("case timed out")


def run_with_timeouts(
    *,
    cases: list[dict[str, Any]],
    per_case_timeout: int,
    top_k: int,
) -> dict[str, Any]:
    from api.chat_memory import call_gemini_chat
    from api.knowledge.vector_store import get_store

    store = get_store()
    answers: dict[str, dict[str, Any]] = {}
    case_errors: list[dict[str, Any]] = []

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        for case in cases:
            case_id = str(case.get("id") or "")
            signal.alarm(max(1, per_case_timeout))
            try:
                generated = generate_answers(
                    [case],
                    search_fn=lambda query, limit: store.search(query, top_k=min(limit, top_k)),
                    chat_fn=call_gemini_chat,
                )
                answers[case_id] = generated.get(case_id, {"answer": "", "source_paths": []})
            except CaseTimeoutError as exc:
                answers[case_id] = {"answer": "", "source_paths": []}
                case_errors.append(
                    {
                        "id": case_id,
                        "query": str(case.get("query") or ""),
                        "error_type": "timeout",
                        "error": str(exc),
                    }
                )
            except Exception as exc:  # pragma: no cover - depends on external API state
                answers[case_id] = {"answer": "", "source_paths": []}
                case_errors.append(
                    {
                        "id": case_id,
                        "query": str(case.get("query") or ""),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                )
            finally:
                signal.alarm(0)
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    summary = evaluate_answers(cases, answers)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "passed" if summary["passed"] == summary["total"] else "needs_review",
        "per_case_timeout": per_case_timeout,
        "top_k": top_k,
        "case_errors": case_errors,
        "answers": answers,
        "final": summary,
    }


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--per-case-timeout", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    cases = json.loads(args.eval_set.expanduser().read_text(encoding="utf-8"))
    if args.limit > 0:
        cases = cases[: args.limit]

    report = run_with_timeouts(
        cases=cases,
        per_case_timeout=args.per_case_timeout,
        top_k=args.top_k,
    )
    report["eval_set"] = str(args.eval_set.expanduser())
    _atomic_write_json(args.report.expanduser(), report)
    final = report["final"]
    print(
        "[experience-replay-answer-quality] "
        f"status={report['status']} "
        f"passed={final['passed']}/{final['total']} "
        f"avg={final['average_score']} "
        f"concept={final['concept_coverage']}% "
        f"forbidden={final['forbidden_cases']} "
        f"uncertainty_miss={final['uncertainty_misses']} "
        f"errors={len(report['case_errors'])}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
