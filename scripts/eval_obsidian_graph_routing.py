#!/usr/bin/env python3
"""Evaluate Obsidian retrieval-graph routing against a small golden set."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EVAL_SET = REPO_ROOT / "api" / "knowledge" / "obsidian_graph_routing_eval_set.json"


@dataclass
class EvalResult:
    case_id: str
    query: str
    passed: bool
    top_path: str
    source: str
    detail: str


def load_eval_cases(path: Path = DEFAULT_EVAL_SET) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [case for case in data if isinstance(case, dict)]


def _contains_any(value: str, needles: list[Any]) -> bool:
    return any(str(needle) in value for needle in needles if str(needle).strip())


def evaluate_case(case: dict[str, Any]) -> EvalResult:
    from mobile_app.obsidian_bridge import search_notes

    case_id = str(case.get("id") or "")
    query = str(case.get("query") or "")
    hits = search_notes(query, limit=3)
    top = hits[0] if hits else {}
    top_path = str(top.get("path") or "")
    source = str(top.get("source") or "")
    expected = list(case.get("expected_top_path_any") or [])
    forbidden = list(case.get("forbidden_top_path_any") or [])

    if not top_path:
        return EvalResult(case_id, query, False, "", source, "no hits")
    if expected and not _contains_any(top_path, expected):
        return EvalResult(
            case_id,
            query,
            False,
            top_path,
            source,
            f"top path did not match expected_top_path_any={expected}",
        )
    if forbidden and _contains_any(top_path, forbidden):
        return EvalResult(
            case_id,
            query,
            False,
            top_path,
            source,
            f"top path matched forbidden_top_path_any={forbidden}",
        )
    if "retrieval_graph" not in source:
        return EvalResult(case_id, query, False, top_path, source, "top hit did not use retrieval_graph")
    return EvalResult(case_id, query, True, top_path, source, "ok")


def run_eval(cases: list[dict[str, Any]] | None = None) -> list[EvalResult]:
    return [evaluate_case(case) for case in (cases if cases is not None else load_eval_cases())]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    results = run_eval(load_eval_cases(args.eval_set))
    failures = [result for result in results if not result.passed]
    if args.json:
        print(json.dumps([result.__dict__ for result in results], ensure_ascii=False, indent=2))
    else:
        print(f"cases={len(results)} passed={len(results) - len(failures)} failed={len(failures)}")
        for result in results:
            mark = "PASS" if result.passed else "FAIL"
            print(f"{mark} {result.case_id}: {result.top_path} [{result.source}]")
            if not result.passed:
                print(f"  query={result.query}")
                print(f"  detail={result.detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
