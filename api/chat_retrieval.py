"""Retrieval assembly for `/api/chat`.

This module keeps local Obsidian RAG and Vertex AI Search/Answer wiring out of
`api.main` while preserving the existing response/debug payload shape.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


FallbackSearch = Callable[[str, int], list[dict[str, Any]]]


@dataclass
class ChatRetrievalResult:
    rag_context: str = ""
    rag_refs: list[str] = field(default_factory=list)
    rag_knowledge_refs: list[dict[str, Any]] = field(default_factory=list)
    vertex_agent_search: dict[str, Any] = field(
        default_factory=lambda: {"used": False, "status": "not_attempted", "refs": []}
    )
    vertex_answer_api: dict[str, Any] = field(
        default_factory=lambda: {"used": False, "status": "not_attempted", "refs": []}
    )


def extract_vertex_search_hint(message: str) -> str:
    """Use a compact embedded search hint for Vertex when a long prompt provides one."""
    text = str(message or "")
    marker = "【Vertex補助検索ヒント】"
    if marker not in text:
        return text
    after = text.split(marker, 1)[1]
    lines: list[str] = []
    for raw_line in after.splitlines():
        line = raw_line.strip()
        if not line:
            if lines:
                break
            continue
        if line.startswith("【") and line.endswith("】"):
            break
        lines.append(line.lstrip("・- ").strip())
        if len(" ".join(lines)) >= 220:
            break
    hint = " ".join(line for line in lines if line).strip()
    return hint or text


def _append_rag_hits(
    hits: list[dict[str, Any]],
    *,
    rag_refs: list[str],
    rag_knowledge_refs: list[dict[str, Any]],
) -> str:
    from api.knowledge.vector_store import confidence_for_hit

    all_docs = []
    for hit in hits:
        text = str(hit.get("text") or "").strip()
        ref = str(hit.get("ref") or hit.get("file_name") or "").strip()
        if not text:
            continue
        prefix = f"{ref}: " if ref else ""
        all_docs.append((prefix + text)[:600])
        if ref:
            rag_refs.append(ref)
        if hit.get("doc_id") or ref:
            confidence, confidence_level = confidence_for_hit(hit)
            rag_knowledge_refs.append(
                {
                    "doc_id": hit.get("doc_id", ""),
                    "obsidian_ref": ref,
                    "file_name": str(hit.get("file_name") or ""),
                    "rank_score": hit.get("rank_score"),
                    "confidence": confidence,
                    "confidence_level": confidence_level,
                }
            )
    if not all_docs:
        return ""
    return "\n\n【参照ナレッジ】\n" + "\n---\n".join(all_docs)


def build_chat_retrieval_context(
    message: str,
    *,
    rag_top_k: int,
    question_category: str,
    is_general_response_mode: bool,
    fallback_search: FallbackSearch,
) -> ChatRetrievalResult:
    """Build local RAG plus optional Vertex supplemental context for chat."""
    result = ChatRetrievalResult()
    if rag_top_k <= 0:
        return result

    try:
        from api.knowledge.vector_store import get_store

        hits = get_store().search(message, top_k=rag_top_k)
        result.rag_context = _append_rag_hits(
            hits,
            rag_refs=result.rag_refs,
            rag_knowledge_refs=result.rag_knowledge_refs,
        )
    except Exception as exc:
        print(f"[RAG] 検索エラー: {exc}")

    if not result.rag_context:
        fallback_hits = fallback_search(message, rag_top_k)
        if fallback_hits:
            result.rag_context = _append_rag_hits(
                fallback_hits,
                rag_refs=result.rag_refs,
                rag_knowledge_refs=result.rag_knowledge_refs,
            )

    if is_general_response_mode or question_category == "general":
        return result

    # Vertex AI Search is supplementary: local RAG remains primary, and any
    # auth/network failure degrades to the local path.
    try:
        from api.vertex_agent_search import answer_prompt_context, answer_vertex_agent, search_vertex_agent

        vertex_search_query = extract_vertex_search_hint(message)
        result.vertex_agent_search = search_vertex_agent(vertex_search_query)
        result.vertex_agent_search["query"] = vertex_search_query[:500]
        vertex_context = str(result.vertex_agent_search.get("prompt_context") or "").strip()
        if vertex_context:
            result.rag_context = (
                result.rag_context + "\n\n" + vertex_context
            ).strip() if result.rag_context else vertex_context

        for ref in list(result.vertex_agent_search.get("refs") or [])[:5]:
            ref_text = str(ref or "").strip()
            if not ref_text or ref_text in result.rag_refs:
                continue
            result.rag_refs.append(ref_text)
            result.rag_knowledge_refs.append(
                {
                    "doc_id": "",
                    "obsidian_ref": ref_text,
                    "file_name": Path(ref_text).name,
                    "rank_score": None,
                    "confidence": 0.72,
                    "confidence_level": "medium",
                    "source": "vertex_ai_search",
                }
            )

        if "【Vertex補助検索ヒント】" in message:
            result.vertex_answer_api = answer_vertex_agent(
                vertex_search_query,
                page_size=5,
                include_grounding_supports=True,
                grounding_filtering_level=os.environ.get("VERTEX_ANSWER_GROUNDING_FILTERING_LEVEL") or None,
            )
            answer_context = answer_prompt_context(result.vertex_answer_api)
            if answer_context:
                result.rag_context = (
                    result.rag_context + "\n\n" + answer_context
                ).strip() if result.rag_context else answer_context
    except Exception as exc:
        error = str(exc)[:240]
        result.vertex_agent_search = {"used": False, "status": "error", "error": error, "refs": []}
        result.vertex_answer_api = {"used": False, "status": "error", "error": error, "refs": []}

    return result
