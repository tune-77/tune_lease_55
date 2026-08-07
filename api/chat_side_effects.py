"""Payload builders for chat logging side effects."""

from __future__ import annotations

from typing import Any

from api.chat_debug_metadata import vertex_answer_public_payload, vertex_search_public_payload


def compact_memory_recall_payload(memory_recall: dict[str, Any]) -> dict[str, Any]:
    return {
        "route": memory_recall.get("route"),
        "refs": memory_recall.get("refs", [])[:8],
        "practical_scene": memory_recall.get("practical_scene") or {},
    }


def compact_identity_memory_payload(identity_memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "used": bool(identity_memory.get("block")),
        "refs": identity_memory.get("refs", [])[:8],
        "layers": identity_memory.get("layers", {}),
    }


def chat_exchange_metadata(
    *,
    context_mode: str,
    knowledge_ref_count: int | None = None,
    improvement_mode: bool = False,
    include_improvement_mode: bool = False,
    vertex_ai_search: dict[str, Any] | None = None,
    vertex_answer_api: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"context_mode": context_mode}
    if knowledge_ref_count is not None:
        payload["knowledge_refs"] = knowledge_ref_count
    if include_improvement_mode:
        payload["improvement_mode"] = bool(improvement_mode)
    if vertex_ai_search is not None:
        payload["vertex_ai_search"] = vertex_search_public_payload(vertex_ai_search)
    if vertex_answer_api is not None:
        payload["vertex_answer_api"] = vertex_answer_public_payload(vertex_answer_api)
    if extra:
        payload.update(extra)
    return payload


def prompt_feedback_extra(
    *,
    user_id: str,
    intent: str,
    category: str,
    improvement_mode: bool | None = None,
    memory_recall: dict[str, Any] | None = None,
    continuity_hook: dict[str, Any] | None = None,
    delta_awareness: dict[str, Any] | None = None,
    memory_to_judgment: dict[str, Any] | None = None,
    memory_expression: dict[str, Any] | None = None,
    reflection_gate: dict[str, Any] | None = None,
    grey_judgment_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_id": user_id,
        "intent": intent or "",
        "category": category,
    }
    if improvement_mode is not None:
        payload["improvement_mode"] = bool(improvement_mode)
    if memory_recall is not None:
        payload["memory_recall"] = compact_memory_recall_payload(memory_recall)
    optional_payloads = {
        "continuity_hook": continuity_hook,
        "delta_awareness": delta_awareness,
        "memory_to_judgment": memory_to_judgment,
        "memory_expression": memory_expression,
        "reflection_gate": reflection_gate,
        "grey_judgment_memory": grey_judgment_memory,
    }
    for key, value in optional_payloads.items():
        if value is not None:
            payload[key] = value
    return payload


def memory_usage_extra(
    *,
    user_id: str,
    category: str,
    improvement_mode: bool | None = None,
    memory_recall: dict[str, Any] | None = None,
    identity_memory: dict[str, Any] | None = None,
    continuity_hook: dict[str, Any] | None = None,
    delta_awareness: dict[str, Any] | None = None,
    memory_to_judgment: dict[str, Any] | None = None,
    memory_expression: dict[str, Any] | None = None,
    reflection_gate: dict[str, Any] | None = None,
    grey_judgment_memory: dict[str, Any] | None = None,
    vertex_ai_search: dict[str, Any] | None = None,
    vertex_answer_api: dict[str, Any] | None = None,
    estimated_user_emotion: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_id": user_id,
        "category": category,
    }
    if improvement_mode is not None:
        payload["improvement_mode"] = bool(improvement_mode)
    if estimated_user_emotion:
        payload["estimated_user_emotion"] = estimated_user_emotion
    if vertex_ai_search is not None:
        payload["vertex_ai_search"] = vertex_search_public_payload(vertex_ai_search)
    if vertex_answer_api is not None:
        payload["vertex_answer_api"] = vertex_answer_public_payload(vertex_answer_api)
    if memory_recall is not None:
        payload["memory_recall"] = compact_memory_recall_payload(memory_recall)
    if identity_memory is not None:
        payload["identity_memory"] = compact_identity_memory_payload(identity_memory)
    optional_payloads = {
        "continuity_hook": continuity_hook,
        "delta_awareness": delta_awareness,
        "memory_to_judgment": memory_to_judgment,
        "memory_expression": memory_expression,
        "reflection_gate": reflection_gate,
        "grey_judgment_memory": grey_judgment_memory,
    }
    for key, value in optional_payloads.items():
        if value is not None:
            payload[key] = value
    return payload


def should_auto_save_chat(*, improvement_mode: bool) -> bool:
    return not bool(improvement_mode)
