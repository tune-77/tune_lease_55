"""Response payload helpers for `/api/chat`.

This is intentionally thin: endpoint code still owns business decisions, while
common payload assembly stays in one place.
"""

from __future__ import annotations

from typing import Any

from api.chat_debug_metadata import vertex_answer_public_payload, vertex_search_public_payload


def build_prompt_feedback_snapshot(system_prompt: str, history_for_gemini: list[dict[str, str]], user_message: str) -> str:
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history_for_gemini)
    return "\n\n".join([system_prompt, history_text, f"user: {user_message}"])


def obsidian_daily_intelligence_payload(
    *,
    used: bool,
    injected: dict[str, Any] | None = None,
    effect: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "used": bool(used),
        "injected": injected or {},
        "effect": effect or {},
    }


def vertex_retrieval_response_extra(
    *,
    vertex_ai_search: dict[str, Any],
    vertex_answer_api: dict[str, Any],
) -> dict[str, Any]:
    return {
        "vertex_ai_search": vertex_search_public_payload(vertex_ai_search),
        "vertex_answer_api": vertex_answer_public_payload(vertex_answer_api),
    }


def build_chat_response_payload(
    *,
    reply: str,
    total_messages: int,
    lease_news_focus: dict[str, Any],
    lease_news_brief: dict[str, Any],
    lease_news_actions: dict[str, Any],
    long_input_mode: bool,
    context_mode: str,
    response_mode: str,
    external_research: dict[str, Any],
    personal_memory_capture: dict[str, Any],
    language_material_capture: dict[str, Any],
    response_impact_prediction: dict[str, Any],
    judgment_asset_capture: dict[str, Any],
    obsidian_daily_intelligence: dict[str, Any],
    knowledge_refs: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "reply": reply,
        "total_messages": total_messages,
        "lease_news_focus": lease_news_focus,
        "lease_news_brief": lease_news_brief,
        "lease_news_actions": lease_news_actions,
        "long_input_mode": long_input_mode,
        "context_mode": context_mode,
        "personal_memory_capture": personal_memory_capture,
        "language_material_capture": language_material_capture,
        "response_impact_prediction": response_impact_prediction,
        "judgment_asset_capture": judgment_asset_capture,
        "response_mode": response_mode,
        "external_research": external_research,
        "obsidian_daily_intelligence": obsidian_daily_intelligence,
    }
    if knowledge_refs is not None:
        payload["knowledge_refs"] = knowledge_refs
    if extra:
        payload.update(extra)
    return payload
