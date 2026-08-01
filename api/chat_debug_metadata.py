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


def relationship_loop_engineering_payload(
    *,
    continuity_hook: dict[str, Any] | None = None,
    delta_awareness: dict[str, Any] | None = None,
    memory_to_judgment: dict[str, Any] | None = None,
    reflection_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hook = continuity_hook if isinstance(continuity_hook, dict) else {}
    delta = delta_awareness if isinstance(delta_awareness, dict) else {}
    m2j = memory_to_judgment if isinstance(memory_to_judgment, dict) else {}
    reflection = reflection_gate if isinstance(reflection_gate, dict) else {}
    human_feedback = hook.get("human_response_feedback") if isinstance(hook.get("human_response_feedback"), dict) else {}
    return {
        "name": "Relationship Loop Engineering",
        "version": 1,
        "purpose": "Userの反応を観測し、次の返答冒頭・差分認識・判断変換へ戻す閉ループ",
        "loop": [
            {
                "step": "observe",
                "label": "Human Response Feedback",
                "evidence": {
                    "positive_count": int(human_feedback.get("positive_count") or 0),
                    "negative_count": int(human_feedback.get("negative_count") or 0),
                },
            },
            {
                "step": "classify",
                "label": "Route Classification",
                "evidence": {"route": str(hook.get("route") or "")},
            },
            {
                "step": "select",
                "label": "Continuity Hook",
                "evidence": {"hook": str(hook.get("hook") or "")},
            },
            {
                "step": "compare",
                "label": "Delta Awareness",
                "evidence": {"delta": str(delta.get("delta") or "")},
            },
            {
                "step": "convert",
                "label": "Memory-to-Judgment",
                "evidence": {"directive": str(m2j.get("directive") or "")},
            },
            {
                "step": "reflect",
                "label": "Reflection Gate",
                "evidence": {
                    "used": bool(reflection.get("used")),
                    "mode": str(reflection.get("mode") or "silent"),
                },
            },
            {
                "step": "return",
                "label": "Next Response",
                "evidence": {"next_feedback_endpoint": "/api/human-response-feedback"},
            },
        ],
        "closed_loop": True,
    }


def chat_memory_debug_payload(
    *,
    category: str,
    context_mode: str = "",
    knowledge_refs: list[str] | None = None,
    memory_recall: dict[str, Any] | None = None,
    pdca_block: str = "",
    judgment_learning_used: bool = False,
    rag_context: str = "",
    db_context: str = "",
    obsidian_daily_used: bool = False,
    identity_memory: dict[str, Any] | None = None,
    continuity_hook: dict[str, Any] | None = None,
    delta_awareness: dict[str, Any] | None = None,
    memory_to_judgment: dict[str, Any] | None = None,
    memory_expression: dict[str, Any] | None = None,
    relationship_loop_engineering: dict[str, Any] | None = None,
    reflection_gate: dict[str, Any] | None = None,
    experience_loop: dict[str, Any] | None = None,
    grey_judgment_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recall = memory_recall if isinstance(memory_recall, dict) else {}
    identity = identity_memory if isinstance(identity_memory, dict) else {}
    identity_layers = identity.get("layers") if isinstance(identity.get("layers"), dict) else {}
    hook = continuity_hook if isinstance(continuity_hook, dict) else {}
    delta = delta_awareness if isinstance(delta_awareness, dict) else {}
    m2j = memory_to_judgment if isinstance(memory_to_judgment, dict) else {}
    expression = memory_expression if isinstance(memory_expression, dict) else {}
    reflection = reflection_gate if isinstance(reflection_gate, dict) else {}
    experience = experience_loop if isinstance(experience_loop, dict) else {}
    grey_memory = grey_judgment_memory if isinstance(grey_judgment_memory, dict) else {}
    loop = relationship_loop_engineering if isinstance(
        relationship_loop_engineering, dict
    ) else relationship_loop_engineering_payload(
        continuity_hook=hook,
        delta_awareness=delta,
        memory_to_judgment=m2j,
        reflection_gate=reflection,
    )
    return {
        "category": category,
        "context_mode": context_mode,
        "relationship_loop_engineering": loop,
        "continuity_hook": {
            "used": bool(hook.get("used")),
            "route": str(hook.get("route") or ""),
            "hook": str(hook.get("hook") or ""),
            "reason": str(hook.get("reason") or ""),
            "human_response_feedback": hook.get("human_response_feedback") or {},
        },
        "delta_awareness": {
            "used": bool(delta.get("used")),
            "current_route": str(delta.get("current_route") or ""),
            "previous_route": str(delta.get("previous_route") or ""),
            "delta": str(delta.get("delta") or ""),
        },
        "memory_to_judgment": {
            "used": bool(m2j.get("used")),
            "route": str(m2j.get("route") or ""),
            "directive": str(m2j.get("directive") or ""),
            "memory_refs": list(m2j.get("memory_refs") or [])[:8],
            "knowledge_refs": list(m2j.get("knowledge_refs") or [])[:8],
        },
        "memory_expression": {
            "used": bool(expression.get("used")),
            "route": str(expression.get("route") or ""),
            "memory_refs": int(expression.get("memory_refs") or 0),
            "knowledge_refs": int(expression.get("knowledge_refs") or 0),
            "grey_refs": int(expression.get("grey_refs") or 0),
            "example": str(expression.get("example") or ""),
        },
        "reflection_gate": {
            "used": bool(reflection.get("used")),
            "mode": str(reflection.get("mode") or ""),
            "route": str(reflection.get("route") or ""),
            "checklist": list(reflection.get("checklist") or [])[:8],
        },
        "experience_loop": experience,
        "grey_judgment_memory": {
            "used": bool(grey_memory.get("used")),
            "reason": str(grey_memory.get("reason") or ""),
            "query_terms": list(grey_memory.get("query_terms") or [])[:12],
            "refs": list(grey_memory.get("refs") or [])[:8],
        },
        "knowledge_refs": list(knowledge_refs or [])[:12],
        "memory_recall": {
            "route": recall.get("route", ""),
            "refs": list(recall.get("refs") or [])[:12],
            "impact_hints": list(recall.get("impact_hints") or [])[:8],
            "practical_scene": recall.get("practical_scene") or {},
        },
        "identity_memory": {
            "used": bool(str(identity.get("block") or "").strip()),
            "refs": list(identity.get("refs") or [])[:8],
            "layers": {
                "identity": bool(identity_layers.get("identity")),
                "judgment": bool(identity_layers.get("judgment")),
                "recent": bool(identity_layers.get("recent")),
            },
        },
        "pdca_applied": bool(str(pdca_block or "").strip()),
        "judgment_learning_used": bool(judgment_learning_used),
        "rag_context_used": bool(str(rag_context or "").strip()),
        "db_context_used": bool(str(db_context or "").strip()),
        "obsidian_daily_used": bool(obsidian_daily_used),
    }


def dialogue_shared_memory_public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    identity = payload.get("identity_memory") if isinstance(payload.get("identity_memory"), dict) else {}
    experience = payload.get("experience_loop") if isinstance(payload.get("experience_loop"), dict) else {}
    recall = payload.get("memory_recall") if isinstance(payload.get("memory_recall"), dict) else {}
    hook = payload.get("continuity_hook") if isinstance(payload.get("continuity_hook"), dict) else {}
    delta = payload.get("delta_awareness") if isinstance(payload.get("delta_awareness"), dict) else {}
    m2j = payload.get("memory_to_judgment") if isinstance(payload.get("memory_to_judgment"), dict) else {}
    expression = payload.get("memory_expression") if isinstance(payload.get("memory_expression"), dict) else {}
    reflection = payload.get("reflection_gate") if isinstance(payload.get("reflection_gate"), dict) else {}
    grey = payload.get("grey_judgment_memory") if isinstance(payload.get("grey_judgment_memory"), dict) else {}
    layers = identity.get("layers") if isinstance(identity.get("layers"), dict) else {}
    return {
        "identity_memory": {
            "used": bool(str(identity.get("block") or "").strip()),
            "refs": list(identity.get("refs") or [])[:8],
            "layers": {
                "identity": bool(layers.get("identity")),
                "judgment": bool(layers.get("judgment")),
                "recent": bool(layers.get("recent")),
            },
        },
        "experience_loop": experience,
        "memory_recall": {
            "route": str(recall.get("route") or ""),
            "refs": list(recall.get("refs") or [])[:8],
            "practical_scene": recall.get("practical_scene") or {},
        },
        "continuity_hook": {
            "used": bool(hook.get("used")),
            "route": str(hook.get("route") or ""),
            "reason": str(hook.get("reason") or ""),
        },
        "delta_awareness": {
            "used": bool(delta.get("used")),
            "delta": str(delta.get("delta") or ""),
        },
        "memory_to_judgment": {
            "used": bool(m2j.get("used")),
            "route": str(m2j.get("route") or ""),
            "directive": str(m2j.get("directive") or ""),
        },
        "memory_expression": {
            "used": bool(expression.get("used")),
            "route": str(expression.get("route") or ""),
            "memory_refs": int(expression.get("memory_refs") or 0),
            "knowledge_refs": int(expression.get("knowledge_refs") or 0),
            "grey_refs": int(expression.get("grey_refs") or 0),
        },
        "reflection_gate": {
            "used": bool(reflection.get("used")),
            "mode": str(reflection.get("mode") or ""),
        },
        "grey_judgment_memory": {
            "used": bool(grey.get("used")),
            "refs": list(grey.get("refs") or [])[:8],
        },
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
