"""Payload builders for chat logging side effects."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from api.chat_debug_metadata import vertex_answer_public_payload, vertex_search_public_payload
from runtime_paths import get_data_path
from shion_verification_origin import is_verification_turn, origin_fields


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
    world_proxy: dict[str, Any] | None = None,
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
        "world_proxy": world_proxy,
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
    world_proxy: dict[str, Any] | None = None,
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
        "world_proxy": world_proxy,
    }
    for key, value in optional_payloads.items():
        if value is not None:
            payload[key] = value
    return payload


def should_auto_save_chat(*, improvement_mode: bool) -> bool:
    # REV-591: 検証の会話は Obsidian の自動保存の材料にしない
    return not bool(improvement_mode) and not is_verification_turn()


def redact_chat_log_text(value: str, limit: int = 1200) -> str:
    text = str(value or "")
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[email]", text)
    text = re.sub(r"\b0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{3,4}\b", "[phone]", text)
    text = re.sub(r"\b\d{6,}\b", "[number]", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def append_local_cloudrun_chat_log(
    *,
    surface: str,
    user_id: str,
    category: str,
    response_mode: str,
    user_message: str,
    assistant_reply: str,
    metadata: dict,
) -> None:
    """ローカル実行時、Private Reflectionが読むdata/cloudrun_chat_log.jsonlに直接1行追記する。

    scripts/sync_cloudrun_inputs_from_gcs.py の _chat_entry_from_event() と同じスキーマ
    （event_id/ts/surface/user_id/category/response_mode/user_message/assistant_reply/
    metadata/shion_hypothesis/source）に合わせている。GCS writebackはローカルでは無効な
    ため、この関数がローカル対話をPrivate Reflectionへ到達させる唯一の経路になる。
    """
    import datetime as _dt
    from uuid import uuid4

    row = {
        "event_id": str(uuid4()),
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "surface": surface or "unknown",
        "user_id": str(user_id or "default")[:80],
        "category": str(category or "")[:80],
        "response_mode": str(response_mode or "")[:40],
        "user_message": redact_chat_log_text(user_message, limit=1200),
        "assistant_reply": redact_chat_log_text(assistant_reply, limit=1800),
        "metadata": metadata if isinstance(metadata, dict) else {},
        "shion_hypothesis": {},
        "source": "local_direct",
        **origin_fields(),  # REV-591: 検証の会話は origin=verification（記憶・内省の材料から外す印）
    }
    path = Path(get_data_path("cloudrun_chat_log.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
