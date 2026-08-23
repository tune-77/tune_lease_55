#!/usr/bin/env python3
"""Compare a LlamaIndex sidecar retriever against the existing RAG eval set.

This script is intentionally isolated from production RAG. It writes reports only
and exits successfully with a skipped report when LlamaIndex is not installed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.knowledge.obsidian_loader import _chunk_by_h2, _parse_frontmatter
from obsidian_query import list_vault_md_files
from scripts.evaluate_obsidian_rag import evaluate_cases

DEFAULT_KNOWLEDGE_DIR = REPO_ROOT / "knowledge_base" / "okf_lease_concepts"
DEFAULT_EVAL_SET = REPO_ROOT / "api" / "knowledge" / "okf_rag_eval_set.json"
DEFAULT_REPORT_PREFIX = REPO_ROOT / "reports" / "llamaindex_rag_comparison_latest"


def _source_ref(path: Path, knowledge_dir: Path) -> str:
    for base in (REPO_ROOT, knowledge_dir):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    raw = path.as_posix()
    if "/Obsidian Vault/" in raw:
        return raw.split("/Obsidian Vault/", 1)[1]
    return raw


def _is_excluded(path_ref: str, exclude_paths: list[str]) -> bool:
    return any(pattern and pattern in path_ref for pattern in exclude_paths)


def _load_markdown_chunks(knowledge_dir: Path, exclude_paths: list[str] | None = None) -> list[Any]:
    exclude_paths = exclude_paths or []
    chunks = []
    for path in sorted(list_vault_md_files(knowledge_dir)):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        rel = _source_ref(path, knowledge_dir)
        if _is_excluded(rel, exclude_paths):
            continue
        chunks.extend(_chunk_by_h2(body, rel, path.name, meta, path.stat().st_mtime))
    return chunks


def _display_path_from_node(node: Any) -> str:
    metadata = getattr(node, "metadata", {}) or {}
    return str(metadata.get("file_path") or metadata.get("ref") or metadata.get("file_name") or "")


def _load_llamaindex() -> tuple[Any, Any, Any, Any] | None:
    try:
        from llama_index.core import Document, SimpleKeywordTableIndex, VectorStoreIndex
        from llama_index.core.embeddings import BaseEmbedding
        from llama_index.core.llms.mock import MockLLM
    except Exception:
        return None
    return Document, SimpleKeywordTableIndex, VectorStoreIndex, BaseEmbedding, MockLLM


def _char_ngrams(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    grams: list[str] = []
    for size in (2, 3):
        grams.extend(normalized[i : i + size] for i in range(max(0, len(normalized) - size + 1)))
    return grams


def _hashed_ngram_vector(text: str, dimensions: int) -> list[float]:
    values = [0.0] * dimensions
    for gram in _char_ngrams(text):
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[bucket] += sign
    norm = math.sqrt(sum(value * value for value in values))
    if not norm:
        return values
    return [value / norm for value in values]


def _make_char_ngram_embedding(BaseEmbedding: Any, dimensions: int = 768):
    class CharNgramEmbedding(BaseEmbedding):
        dimensions: int = 768

        def _get_text_embedding(self, text: str) -> list[float]:
            return _hashed_ngram_vector(text, self.dimensions)

        def _get_query_embedding(self, query: str) -> list[float]:
            return _hashed_ngram_vector(query, self.dimensions)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

    return CharNgramEmbedding(model_name=f"local-char-ngram-{dimensions}", dimensions=dimensions)


def _build_llamaindex_searcher(chunks: list[Any], top_k: int):
    loaded = _load_llamaindex()
    if loaded is None:
        return None

    Document, _SimpleKeywordTableIndex, VectorStoreIndex, BaseEmbedding, MockLLM = loaded
    documents = [
        Document(
            text=str(chunk.text),
            metadata={
                "file_path": str(chunk.file_path),
                "file_name": str(chunk.file_name),
                "section": str(chunk.section),
                "ref": str(chunk.file_path),
            },
        )
        for chunk in chunks
    ]
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=_make_char_ngram_embedding(BaseEmbedding),
        llm=MockLLM(),
    )
    retriever = index.as_retriever(similarity_top_k=max(top_k * 4, top_k))

    def search(query: str, limit: int) -> list[dict[str, str]]:
        hits: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        for result in retriever.retrieve(query):
            path = _display_path_from_node(result.node)
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            metadata = getattr(result.node, "metadata", {}) or {}
            hits.append(
                {
                    "file_path": path,
                    "file_name": str(metadata.get("file_name") or ""),
                    "ref": str(metadata.get("ref") or ""),
                }
            )
            if len(hits) >= limit:
                break
        return hits

    return search


def _skipped_payload(reason: str, *, knowledge_dir: Path, eval_set: Path, top_k: int) -> dict[str, Any]:
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "skipped",
        "reason": reason,
        "method": {
            "candidate": "llama_index.core.VectorStoreIndex + local char-ngram embedding",
            "top_k": top_k,
            "knowledge_dir": knowledge_dir.as_posix(),
            "eval_set": eval_set.as_posix(),
            "guardrail": "sidecar_only_no_rag_rank_change_no_prompt_change_no_scoring_no_obsidian_write",
        },
        "summary": {},
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# LlamaIndex RAG Comparison",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Status: {payload['status']}",
        f"- Method: {payload['method']['candidate']}",
        f"- Top K: {payload['method']['top_k']}",
        f"- Guardrail: {payload['method']['guardrail']}",
        "",
    ]
    if payload["status"] == "skipped":
        lines.extend([f"- Reason: {payload['reason']}", ""])
        return "\n".join(lines)

    summary = payload.get("summary") or {}
    lines.extend(
        [
            "## Summary",
            "",
            f"- hit@1: {summary.get('hit_at_1')}/{summary.get('total')} ({summary.get('hit_at_1_rate', 0):.1%})",
            f"- hit@{payload['method']['top_k']}: {summary.get('hit_at_k')}/{summary.get('total')} ({summary.get('hit_at_k_rate', 0):.1%})",
            f"- mrr: {summary.get('mrr', 0):.3f}",
            f"- forbidden_cases: {summary.get('forbidden_cases')}/{summary.get('total')}",
            f"- passed: {summary.get('passed')}",
            "",
            "## Cases",
            "",
        ]
    )
    for case in summary.get("cases", []):
        status = "PASS" if case.get("passed") else "MISS"
        lines.append(f"### {case.get('id')}")
        lines.append("")
        lines.append(f"- Status: {status}")
        lines.append(f"- Rank: {case.get('rank') or '-'}")
        if case.get("forbidden_paths"):
            lines.append(f"- Forbidden paths: {case['forbidden_paths']}")
        lines.append("")
        for idx, path in enumerate(case.get("paths") or [], start=1):
            lines.append(f"{idx}. {path}")
        lines.append("")
    return "\n".join(lines)


def _write_reports(payload: dict[str, Any], output_prefix: Path) -> tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return json_path, md_path


def _filter_cases(cases: list[dict[str, Any]], categories: list[str], case_ids: list[str], max_cases: int | None) -> list[dict[str, Any]]:
    selected = cases
    if categories:
        wanted = set(categories)
        selected = [case for case in selected if str(case.get("category") or "") in wanted]
    if case_ids:
        wanted_ids = set(case_ids)
        selected = [case for case in selected if str(case.get("id") or "") in wanted_ids]
    if max_cases is not None:
        selected = selected[:max_cases]
    return selected


def compare(
    knowledge_dir: Path,
    eval_set: Path,
    top_k: int,
    output_prefix: Path,
    categories: list[str] | None = None,
    case_ids: list[str] | None = None,
    max_cases: int | None = None,
    exclude_paths: list[str] | None = None,
) -> dict[str, Any]:
    categories = categories or []
    case_ids = case_ids or []
    exclude_paths = exclude_paths or []
    if _load_llamaindex() is None:
        payload = _skipped_payload(
            "LlamaIndex is not installed. Install it only in an optional pipeline environment before running the experiment.",
            knowledge_dir=knowledge_dir,
            eval_set=eval_set,
            top_k=top_k,
        )
        _write_reports(payload, output_prefix)
        return payload

    all_cases = json.loads(eval_set.read_text(encoding="utf-8"))
    cases = _filter_cases(all_cases, categories, case_ids, max_cases)
    if not cases:
        payload = _skipped_payload("no eval cases selected", knowledge_dir=knowledge_dir, eval_set=eval_set, top_k=top_k)
        payload["method"]["categories"] = categories
        payload["method"]["case_ids"] = case_ids
        payload["method"]["max_cases"] = max_cases
        _write_reports(payload, output_prefix)
        return payload
    chunks = _load_markdown_chunks(knowledge_dir, exclude_paths=exclude_paths)
    if not chunks:
        payload = _skipped_payload(f"no markdown chunks found: {knowledge_dir}", knowledge_dir=knowledge_dir, eval_set=eval_set, top_k=top_k)
        _write_reports(payload, output_prefix)
        return payload

    searcher = _build_llamaindex_searcher(chunks, top_k)
    if searcher is None:
        payload = _skipped_payload("LlamaIndex loader returned no searcher.", knowledge_dir=knowledge_dir, eval_set=eval_set, top_k=top_k)
        _write_reports(payload, output_prefix)
        return payload

    summary = evaluate_cases(cases, searcher, top_k)
    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "reason": "",
        "method": {
            "candidate": "llama_index.core.VectorStoreIndex + local char-ngram embedding",
            "top_k": top_k,
            "knowledge_dir": knowledge_dir.as_posix(),
            "eval_set": eval_set.as_posix(),
            "categories": categories,
            "case_ids": case_ids,
            "max_cases": max_cases,
            "exclude_paths": exclude_paths,
            "guardrail": "sidecar_only_no_rag_rank_change_no_prompt_change_no_scoring_no_obsidian_write",
        },
        "summary": summary,
    }
    _write_reports(payload, output_prefix)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_REPORT_PREFIX)
    parser.add_argument("--category", action="append", default=[], help="Filter eval cases by category. Can be repeated.")
    parser.add_argument("--case-id", action="append", default=[], help="Filter eval cases by id. Can be repeated.")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--exclude-path", action="append", default=[], help="Exclude markdown files whose source path contains this string.")
    args = parser.parse_args()

    payload = compare(
        knowledge_dir=args.knowledge_dir.expanduser(),
        eval_set=args.eval_set.expanduser(),
        top_k=args.top_k,
        output_prefix=args.output_prefix.expanduser(),
        categories=list(args.category),
        case_ids=list(args.case_id),
        max_cases=args.max_cases,
        exclude_paths=list(args.exclude_path),
    )
    print(f"status={payload['status']}")
    if payload["status"] == "skipped":
        print(f"reason={payload['reason']}")
    else:
        summary = payload["summary"]
        print(
            f"hit@1={summary['hit_at_1']}/{summary['total']} "
            f"hit@{args.top_k}={summary['hit_at_k']}/{summary['total']} "
            f"mrr={summary['mrr']:.3f} forbidden={summary['forbidden_cases']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
