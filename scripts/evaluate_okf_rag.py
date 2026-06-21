#!/usr/bin/env python3
"""Evaluate the OKF-style knowledge pack in an isolated temporary RAG index."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.knowledge.obsidian_loader import _chunk_by_h2, _parse_frontmatter
from api.knowledge.vector_store import KnowledgeVectorStore
from scripts.evaluate_obsidian_rag import evaluate_cases

DEFAULT_KNOWLEDGE_DIR = REPO_ROOT / "knowledge_base" / "okf_lease_concepts"
DEFAULT_EVAL_SET = REPO_ROOT / "api" / "knowledge" / "okf_rag_eval_set.json"
DEFAULT_REPORT = REPO_ROOT / "reports" / "okf_rag_eval_latest.json"


def _load_okf_chunks(knowledge_dir: Path):
    chunks = []
    for path in sorted(knowledge_dir.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        rel = path.relative_to(REPO_ROOT).as_posix()
        chunks.extend(
            _chunk_by_h2(
                body,
                rel,
                path.name,
                meta,
                path.stat().st_mtime,
            )
        )
    return chunks


def _print_case_report(summary: dict, cases: list[dict], top_k: int) -> None:
    by_id = {item["id"]: item for item in summary["cases"]}
    for case in cases:
        result = by_id[case["id"]]
        status = "PASS" if result["passed"] else "MISS"
        print(f"[{status}] {case['id']} rank={result['rank'] or '-'} query={case['query']}")
        for idx, path in enumerate(result["paths"], start=1):
            print(f"  {idx}. {path}")
        print("")
    print("summary")
    print(f"  hit@1={summary['hit_at_1']}/{summary['total']} ({summary['hit_at_1_rate']:.1%})")
    print(f"  hit@{top_k}={summary['hit_at_k']}/{summary['total']} ({summary['hit_at_k_rate']:.1%})")
    print(f"  mrr={summary['mrr']:.3f}")
    print(f"  forbidden_cases={summary['forbidden_cases']}/{summary['total']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    knowledge_dir = args.knowledge_dir.expanduser()
    cases = json.loads(args.eval_set.expanduser().read_text(encoding="utf-8"))
    chunks = _load_okf_chunks(knowledge_dir)
    if not chunks:
        raise SystemExit(f"no markdown chunks found: {knowledge_dir}")

    with tempfile.TemporaryDirectory(prefix="okf-rag-") as temp_dir:
        store = KnowledgeVectorStore(chroma_dir=temp_dir)
        store.upsert_chunks(chunks)
        summary = evaluate_cases(cases, lambda query, limit: store.search(query, top_k=limit), args.top_k)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"eval_set={args.eval_set}")
    print(f"knowledge_dir={knowledge_dir}")
    print(f"chunks={len(chunks)} top_k={args.top_k}")
    print("")
    _print_case_report(summary, cases, args.top_k)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

