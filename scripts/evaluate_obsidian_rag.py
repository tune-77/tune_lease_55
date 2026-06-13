#!/usr/bin/env python3
"""Evaluate Obsidian RAG retrieval against a small hand-curated query set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EVAL_SET = REPO_ROOT / "api" / "knowledge" / "rag_eval_set.json"


def _display_path(hit: dict) -> str:
    raw = str(hit.get("file_path") or "").replace("\\", "/")
    if "/Obsidian Vault/" in raw:
        raw = raw.split("/Obsidian Vault/", 1)[1]
    return raw or str(hit.get("file_name") or hit.get("ref") or "")


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(pattern and pattern in path for pattern in patterns)


def evaluate_cases(cases: list[dict], search_fn, top_k: int) -> dict:
    total = len(cases)
    hit_at_1 = 0
    hit_at_k = 0
    mrr_total = 0.0
    forbidden_hits = 0
    case_results: list[dict] = []

    for case in cases:
        hits = search_fn(case["query"], top_k)
        paths = [_display_path(hit) for hit in hits]
        expected = list(case.get("expected_path_any") or [])
        forbidden = list(case.get("forbidden_path_any") or [])
        rank = next(
            (i for i, path in enumerate(paths, start=1) if _matches_any(path, expected)),
            0,
        )
        bad_paths = [path for path in paths if _matches_any(path, forbidden)]
        hit_at_1 += int(rank == 1)
        hit_at_k += int(bool(rank))
        mrr_total += 1.0 / rank if rank else 0.0
        forbidden_hits += int(bool(bad_paths))
        case_results.append(
            {
                "id": case["id"],
                "rank": rank,
                "paths": paths,
                "forbidden_paths": bad_paths,
                "passed": bool(rank) and not bad_paths,
            }
        )

    return {
        "total": total,
        "hit_at_1": hit_at_1,
        "hit_at_k": hit_at_k,
        "hit_at_1_rate": hit_at_1 / total if total else 0.0,
        "hit_at_k_rate": hit_at_k / total if total else 0.0,
        "mrr": mrr_total / total if total else 0.0,
        "forbidden_cases": forbidden_hits,
        "passed": hit_at_k == total and forbidden_hits == 0,
        "cases": case_results,
    }


def evaluate(eval_path: Path, top_k: int, json_output: Path | None = None) -> int:
    from api.knowledge.vector_store import get_store

    cases = json.loads(eval_path.read_text(encoding="utf-8"))
    store = get_store()
    summary = evaluate_cases(cases, lambda query, limit: store.search(query, top_k=limit), top_k)

    print(f"eval_set={eval_path}")
    print(f"collection_count={store.count()} top_k={top_k}")
    print("")

    result_by_id = {item["id"]: item for item in summary["cases"]}
    for case in cases:
        result = result_by_id[case["id"]]
        rank = result["rank"]
        bad_paths = result["forbidden_paths"]
        status = "PASS" if rank else "MISS"
        if bad_paths:
            status += "+NOISE"
        print(f"[{status}] {case['id']} rank={rank or '-'} query={case['query']}")
        for i, path in enumerate(result["paths"], start=1):
            print(f"  {i}. {path}")
        if bad_paths:
            print(f"  forbidden: {bad_paths}")
        print("")

    print("summary")
    print(f"  hit@1={summary['hit_at_1']}/{summary['total']} ({summary['hit_at_1_rate']:.1%})")
    print(f"  hit@{top_k}={summary['hit_at_k']}/{summary['total']} ({summary['hit_at_k_rate']:.1%})")
    print(f"  mrr={summary['mrr']:.3f}")
    print(f"  forbidden_cases={summary['forbidden_cases']}/{summary['total']}")
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["passed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default=os.fspath(DEFAULT_EVAL_SET), help="Path to eval JSON")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    raise SystemExit(evaluate(Path(args.eval_set), args.top_k, args.json_output))


if __name__ == "__main__":
    main()
