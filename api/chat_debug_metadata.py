"""Stable debug/metadata payload helpers for chat responses."""

from __future__ import annotations

from typing import Any


def vertex_search_public_payload(vertex_ai_search: dict[str, Any], *, ref_limit: int = 5) -> dict[str, Any]:
    return {
        "used": bool(vertex_ai_search.get("used")),
        "status": vertex_ai_search.get("status"),
        "refs": list(vertex_ai_search.get("refs") or [])[:ref_limit],
    }


def vertex_answer_public_payload(vertex_answer_api: dict[str, Any], *, ref_limit: int = 5) -> dict[str, Any]:
    return {
        "used": bool(vertex_answer_api.get("used")),
        "status": vertex_answer_api.get("status"),
        "grounding_score": vertex_answer_api.get("grounding_score"),
        "grounding_score_source": vertex_answer_api.get("grounding_score_source"),
        "low_support_claim_count": vertex_answer_api.get("low_support_claim_count", 0),
        "support_count": vertex_answer_api.get("support_count", 0),
        "refs": list(vertex_answer_api.get("refs") or [])[:ref_limit],
    }


def user_personal_memory_debug_payload(user_personal_memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "used": bool(user_personal_memory.get("block")),
        "refs": user_personal_memory.get("refs", [])[:6],
        "line_count": user_personal_memory.get("line_count", 0),
    }


def append_retrieval_debug_payloads(
    memory_debug: dict[str, Any],
    *,
    vertex_ai_search: dict[str, Any],
    vertex_answer_api: dict[str, Any],
) -> dict[str, Any]:
    memory_debug["vertex_ai_search"] = {
        **vertex_search_public_payload(vertex_ai_search, ref_limit=8),
        "summary_preview": str(vertex_ai_search.get("summary") or "")[:500],
    }
    memory_debug["vertex_answer_api"] = {
        **vertex_answer_public_payload(vertex_answer_api, ref_limit=8),
        "answer_preview": str(vertex_answer_api.get("answer_text") or "")[:800],
        "grounding_supports": list(vertex_answer_api.get("grounding_supports") or [])[:5],
    }
    return memory_debug


def append_chat_debug_metadata(
    memory_debug: dict[str, Any],
    *,
    user_personal_memory: dict[str, Any],
    vertex_ai_search: dict[str, Any],
    vertex_answer_api: dict[str, Any],
) -> dict[str, Any]:
    memory_debug["user_personal_memory"] = user_personal_memory_debug_payload(user_personal_memory)
    return append_retrieval_debug_payloads(
        memory_debug,
        vertex_ai_search=vertex_ai_search,
        vertex_answer_api=vertex_answer_api,
    )
