"""Obsidian knowledge access for the persistent lease-intelligence self-model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

from obsidian_query import split_query_terms

# インデックス全件の列挙は対話のたびに行うには重いので、件数だけをTTL付きでキャッシュする
_COUNT_TTL_SECONDS = 600.0
_count_cache: dict[str, Any] = {"at": 0.0, "counts": None}


def _load_document_counts() -> dict[str, int]:
    from mobile_app.obsidian_bridge import iter_indexed_obsidian_documents

    documents = iter_indexed_obsidian_documents(include_chat_logs=True, max_chars=1)
    return {
        "indexed_notes": len(documents),
        "knowledge_notes": sum(1 for item in documents if item.get("source_type") == "knowledge"),
        "chat_log_notes": sum(1 for item in documents if item.get("source_type") == "chat_log"),
    }


def _document_counts(now: float | None = None) -> dict[str, int]:
    now = time.time() if now is None else now
    cached = _count_cache.get("counts")
    if cached is not None and now - float(_count_cache.get("at", 0.0)) < _COUNT_TTL_SECONDS:
        return cached
    counts = _load_document_counts()
    _count_cache["at"] = now
    _count_cache["counts"] = counts
    return counts


@dataclass(frozen=True)
class LeaseIntelligenceKnowledge:
    available: bool
    query: str = ""
    context_block: str = ""
    indexed_notes: int = 0
    knowledge_notes: int = 0
    chat_log_notes: int = 0
    source_paths: tuple[str, ...] = ()


def _interest_labels(interests: Iterable[Any]) -> list[str]:
    labels: list[str] = []
    for item in interests:
        value = item.get("label", "") if isinstance(item, dict) else item
        label = str(value or "").strip()
        if label:
            labels.append(label)
    return labels


def build_lease_intelligence_knowledge(
    *,
    theme: str = "",
    focus_lines: Iterable[str] = (),
    current_question: str = "",
    user_interests: Iterable[Any] = (),
    limit: int = 5,
) -> LeaseIntelligenceKnowledge:
    """Retrieve relevant knowledge from the entire shared Obsidian index."""
    raw_parts = [
        str(theme or "").strip(),
        *(str(line or "").strip() for line in focus_lines),
        str(current_question or "").strip(),
        *_interest_labels(user_interests),
    ]
    terms: list[str] = []
    for part in raw_parts:
        terms.extend(split_query_terms(part))
    query = " ".join(dict.fromkeys(term for term in terms if term))[:500]
    if not query:
        query = "リース 審査 判断 設備 与信"

    try:
        from obsidian_ai_context import collect_obsidian_ai_context

        counts = _document_counts()
        result = collect_obsidian_ai_context(
            query,
            limit=limit,
            max_chars=3200,
            heading="リース知性体が参照したObsidian知識",
        )
    except Exception:
        return LeaseIntelligenceKnowledge(available=False, query=query)

    source_paths = tuple(
        str(hit.get("path") or "").strip()
        for hit in result.get("hits", [])
        if str(hit.get("path") or "").strip()
    )
    return LeaseIntelligenceKnowledge(
        available=counts["indexed_notes"] > 0,
        query=query,
        context_block=str(result.get("block") or ""),
        indexed_notes=counts["indexed_notes"],
        knowledge_notes=counts["knowledge_notes"],
        chat_log_notes=counts["chat_log_notes"],
        source_paths=source_paths,
    )


__all__ = ["LeaseIntelligenceKnowledge", "build_lease_intelligence_knowledge"]
