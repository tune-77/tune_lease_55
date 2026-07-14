#!/usr/bin/env python3
"""Compare existing Obsidian RAG search with Shion-HyDE search.

This is an offline/debug tool. It does not change chat behavior or the RAG
index. Use it after the hackathon freeze to inspect whether HyDE improves
retrieval for ambiguous screening questions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.knowledge.shion_hyde import build_combined_search_query, build_shion_hyde_query
from api.knowledge.vector_store import confidence_for_hit, get_store


DEFAULT_QUERIES = [
    "この案件、なんか不安。どこを見ればいい？",
    "財務は悪くなさそうだけど成約できるか微妙",
    "この設備、売れるのかな",
    "新規先で情報が少ない。何を確認する？",
    "Q-riskが高い時は信用リスクとして見るべき？",
    "トラックの法定耐用年数とリース期間を確認したい",
]


def _summarize_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        confidence, level = confidence_for_hit(hit)
        rows.append(
            {
                "rank": rank,
                "ref": hit.get("ref") or hit.get("file_name") or hit.get("doc_id") or "",
                "file_path": hit.get("file_path") or (hit.get("metadata") or {}).get("file_path") or "",
                "section": hit.get("section") or "",
                "rank_score": hit.get("rank_score"),
                "confidence": confidence,
                "confidence_level": level,
                "source": hit.get("source") or "",
                "text_preview": str(hit.get("text") or "").replace("\n", " ")[:180],
            }
        )
    return rows


def evaluate_query(message: str, *, top_k: int) -> dict[str, Any]:
    store = get_store()
    baseline_hits = store.search(message, top_k=top_k, surface="shion_hyde_eval_baseline")
    hyde = build_shion_hyde_query(message)
    hyde_search_query = build_combined_search_query(hyde)
    hyde_hits = (
        store.search(hyde_search_query, top_k=top_k, surface="shion_hyde_eval")
        if hyde.should_search
        else []
    )
    return {
        "message": message,
        "hyde": hyde.to_dict(),
        "baseline_hits": _summarize_hits(baseline_hits),
        "hyde_hits": _summarize_hits(hyde_hits),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", nargs="*", help="Queries to evaluate. Defaults to built-in smoke queries.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a readable report.")
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES
    results = [evaluate_query(query, top_k=args.top_k) for query in queries]

    if args.json:
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
        return 0

    for result in results:
        print(f"\n## {result['message']}")
        print("\n[HyDE]")
        print(result["hyde"]["hyde_query"] or "(skipped)")
        print("\n[Baseline Top]")
        for hit in result["baseline_hits"][:3]:
            print(f"- {hit['confidence']:.3f} {hit['ref']} / {hit['section']}")
        print("\n[Shion-HyDE Top]")
        for hit in result["hyde_hits"][:3]:
            print(f"- {hit['confidence']:.3f} {hit['ref']} / {hit['section']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
