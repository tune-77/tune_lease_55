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


def evaluate(eval_path: Path, top_k: int) -> int:
    from api.knowledge.vector_store import get_store

    cases = json.loads(eval_path.read_text(encoding="utf-8"))
    store = get_store()
    total = len(cases)
    hit_at_1 = 0
    hit_at_k = 0
    mrr_total = 0.0
    forbidden_hits = 0

    print(f"eval_set={eval_path}")
    print(f"collection_count={store.count()} top_k={top_k}")
    print("")

    for case in cases:
        query = case["query"]
        expected = list(case.get("expected_path_any") or [])
        forbidden = list(case.get("forbidden_path_any") or [])
        hits = store.search(query, top_k=top_k)
        paths = [_display_path(hit) for hit in hits]

        rank = 0
        for i, path in enumerate(paths, start=1):
            if _matches_any(path, expected):
                rank = i
                break

        bad_paths = [path for path in paths if _matches_any(path, forbidden)]
        if bad_paths:
            forbidden_hits += 1

        if rank == 1:
            hit_at_1 += 1
        if rank:
            hit_at_k += 1
            mrr_total += 1.0 / rank

        status = "PASS" if rank else "MISS"
        if bad_paths:
            status += "+NOISE"
        print(f"[{status}] {case['id']} rank={rank or '-'} query={query}")
        for i, hit in enumerate(hits, start=1):
            path = paths[i - 1]
            source = hit.get("source")
            distance = hit.get("distance")
            priority = hit.get("priority_score")
            print(f"  {i}. {path} source={source} distance={distance} priority={priority}")
        if bad_paths:
            print(f"  forbidden: {bad_paths}")
        print("")

    print("summary")
    print(f"  hit@1={hit_at_1}/{total} ({hit_at_1 / total:.1%})")
    print(f"  hit@{top_k}={hit_at_k}/{total} ({hit_at_k / total:.1%})")
    print(f"  mrr={mrr_total / total:.3f}")
    print(f"  forbidden_cases={forbidden_hits}/{total}")
    return 0 if hit_at_k == total and forbidden_hits == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default=os.fspath(DEFAULT_EVAL_SET), help="Path to eval JSON")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    raise SystemExit(evaluate(Path(args.eval_set), args.top_k))


if __name__ == "__main__":
    main()
