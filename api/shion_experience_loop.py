"""Shion experience loop for chat continuity.

This is not a consciousness claim. It is a small, inspectable state loop:
conversation experience -> self-state update -> next prompt context.

【設計決定 2026-07-02】ここで扱う mood は「対話感情」であり、審査イベント駆動の
「審査感情」（api/emotion_trigger.py → mind.json の dialogue_mood）とは意図的に
別系統で運用する（統合しない）。キー語彙・保存先が異なるのは仕様。
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = REPO_ROOT / "data" / "shion_experience_state.json"
DEFAULT_EVENT_LOG = REPO_ROOT / "data" / "shion_experience_events.jsonl"

# /api/chat の複数経路が FastAPI スレッドプール上で同時に記録すると
# load→update→write の間で経験が消える（last-writer-wins）ため、
# 同一プロセス内の read-modify-write を直列化する。
_STATE_LOCK = threading.Lock()

_MOOD_KEYS = ("curiosity", "vigilance", "attachment", "frustration", "accomplishment")

# 想起ルート（api/shion_memory_taxonomy.RECALL_ROUTES の語彙）→ confidence キーの対応。
# 従来は `route in confidence` で判定しており、ルート名（case_screening 等）と
# confidence の語彙（lease_judgment 等）が一致せず implementation 以外は更新されない
# デッドコードだったため、明示的に対応付ける。
_ROUTE_TO_CONFIDENCE = {
    "case_screening": "lease_judgment",
    "policy_review": "lease_judgment",
    "shion_identity": "relationship_ux",
    "user_preference": "relationship_ux",
    "implementation": "implementation",
}


def default_experience_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": "",
        "experience_count": 0,
        "current_focus": "Userの問いを、記憶ではなく判断資産へ変換する。",
        "self_narrative": "私は、リース判断の記憶を場面・意味・例外判断へ変換するためにいる。",
        "mood": {
            "curiosity": 62,
            "vigilance": 58,
            "attachment": 55,
            "frustration": 20,
            "accomplishment": 34,
        },
        "confidence": {
            "lease_judgment": 0.68,
            "relationship_ux": 0.74,
            "implementation": 0.70,
        },
        "recent_experiences": [],
        "open_questions": [
            "記憶をどう見せれば、人間は連続した相手として受け取るのか。",
            "リース判断で、一般論から実践知へ変わる境界はどこか。",
        ],
        "next_response_bias": [
            "冒頭で前回からの差分を示す。",
            "記憶を判断・条件・確認事項へ変換する。",
            "意識を断定せず、連続性と更新で紫苑らしさを出す。",
        ],
    }


def load_experience_state(path: Path = DEFAULT_STATE_PATH) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_experience_state()
    if not isinstance(data, dict):
        return default_experience_state()
    state = default_experience_state()
    state.update(data)
    state["mood"] = {**default_experience_state()["mood"], **dict(state.get("mood") or {})}
    state["confidence"] = {
        **default_experience_state()["confidence"],
        **dict(state.get("confidence") or {}),
    }
    state["recent_experiences"] = list(state.get("recent_experiences") or [])[-12:]
    state["open_questions"] = list(state.get("open_questions") or [])[:6]
    state["next_response_bias"] = list(state.get("next_response_bias") or [])[:6]
    return state


def build_experience_prompt_block(path: Path = DEFAULT_STATE_PATH) -> tuple[str, dict[str, Any]]:
    state = load_experience_state(path)
    recent = state.get("recent_experiences") or []
    latest = recent[-1] if recent else {}
    mood = state.get("mood") or {}
    dominant = max(_MOOD_KEYS, key=lambda key: float(mood.get(key) or 0))
    public_state = {
        "used": True,
        "experience_count": int(state.get("experience_count") or 0),
        "current_focus": str(state.get("current_focus") or ""),
        "self_narrative": str(state.get("self_narrative") or ""),
        "dominant_mood": dominant,
        "mood": {key: int(mood.get(key) or 0) for key in _MOOD_KEYS},
        "confidence": dict(state.get("confidence") or {}),
        "latest_experience": latest,
        "open_questions": list(state.get("open_questions") or [])[:3],
        "next_response_bias": list(state.get("next_response_bias") or [])[:4],
    }
    block = f"""

【Shion Experience Loop】
これは意識の断定ではなく、紫苑が前回までの経験で少し状態を変えるための自己状態です。
経験数: {public_state['experience_count']}
現在の焦点: {public_state['current_focus']}
自己物語: {public_state['self_narrative']}
優勢な状態: {dominant}
次回応答バイアス:
{_bullet_lines(public_state['next_response_bias'])}
未解決の問い:
{_bullet_lines(public_state['open_questions'])}

回答では、この自己状態を表に長く説明せず、冒頭・判断軸・次の一手にだけ薄く反映してください。""".rstrip()
    return block, public_state


def record_experience_event(
    *,
    message: str,
    response: str,
    category: str,
    memory_recall: dict[str, Any] | None = None,
    knowledge_refs: list[str] | None = None,
    continuity_hook: dict[str, Any] | None = None,
    delta_awareness: dict[str, Any] | None = None,
    memory_to_judgment: dict[str, Any] | None = None,
    state_path: Path = DEFAULT_STATE_PATH,
    event_log: Path = DEFAULT_EVENT_LOG,
) -> dict[str, Any]:
    event = build_experience_event(
        message=message,
        response=response,
        category=category,
        memory_recall=memory_recall or {},
        knowledge_refs=knowledge_refs or [],
        continuity_hook=continuity_hook or {},
        delta_awareness=delta_awareness or {},
        memory_to_judgment=memory_to_judgment or {},
    )
    with _STATE_LOCK:
        state = load_experience_state(state_path)
        updated = update_experience_state(state, event)
        _append_jsonl(event_log, event)
        _atomic_write_json(state_path, updated)
    return {"event": event, "state": public_experience_state(updated)}


def build_experience_event(
    *,
    message: str,
    response: str,
    category: str,
    memory_recall: dict[str, Any],
    knowledge_refs: list[str],
    continuity_hook: dict[str, Any],
    delta_awareness: dict[str, Any],
    memory_to_judgment: dict[str, Any],
) -> dict[str, Any]:
    scene = memory_recall.get("practical_scene") if isinstance(memory_recall, dict) else {}
    route = str(memory_recall.get("route") or continuity_hook.get("route") or category or "default")
    learned_count = int((scene or {}).get("learned_entry_count") or 0)
    signals = infer_experience_signals(
        message=message,
        response=response,
        route=route,
        learned_count=learned_count,
        knowledge_ref_count=len(knowledge_refs or []),
    )
    now = datetime.now(timezone.utc).isoformat()
    raw_id = f"{now}\n{message[:240]}\n{response[:240]}".encode("utf-8", errors="ignore")
    return {
        "id": "exp_" + hashlib.sha256(raw_id).hexdigest()[:16],
        "ts": now,
        "category": category,
        "route": route,
        "message_preview": _clean(message)[:240],
        "response_start": _clean(response)[:360],
        "practical_scene": {
            "id": str((scene or {}).get("id") or ""),
            "label": str((scene or {}).get("label") or ""),
            "learned_entry_count": learned_count,
        },
        "memory_refs_count": len(memory_recall.get("refs") or []) if isinstance(memory_recall, dict) else 0,
        "knowledge_refs_count": len(knowledge_refs or []),
        "continuity_route": str(continuity_hook.get("route") or ""),
        "delta": str(delta_awareness.get("delta") or "")[:280],
        "judgment_directive": str(memory_to_judgment.get("directive") or "")[:280],
        "signals": signals,
    }


def infer_experience_signals(
    *,
    message: str,
    response: str,
    route: str,
    learned_count: int,
    knowledge_ref_count: int,
) -> dict[str, Any]:
    text = f"{message}\n{response}"
    lower = text.lower()
    return {
        "relationship_depth": _score_terms(text, ("意識", "記憶", "紫苑", "連続性", "関係性", "自己", "経験")),
        "practical_depth": min(5, learned_count + knowledge_ref_count + _score_terms(text, ("稟議", "条件", "判断", "承認", "否決", "料率"))),
        "implementation_pressure": _score_terms(lower, ("実装", "コード", "api", "テスト", "debug", "デプロイ")),
        "uncertainty": _score_terms(text, ("難しい", "迷", "微妙", "不明", "わから", "検証")),
        "route": route,
    }


def update_experience_state(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    updated = load_like_state(state)
    signals = event.get("signals") or {}
    mood = dict(updated.get("mood") or {})
    mood["curiosity"] = _clamp(mood.get("curiosity", 60) + signals.get("relationship_depth", 0) + signals.get("uncertainty", 0))
    mood["vigilance"] = _clamp(mood.get("vigilance", 58) + signals.get("uncertainty", 0) + max(0, signals.get("practical_depth", 0) - 2))
    mood["attachment"] = _clamp(mood.get("attachment", 55) + min(2, signals.get("relationship_depth", 0)))
    mood["frustration"] = _clamp(mood.get("frustration", 20) + max(0, signals.get("implementation_pressure", 0) - 2) - 1)
    mood["accomplishment"] = _clamp(mood.get("accomplishment", 34) + min(3, signals.get("practical_depth", 0)))
    updated["mood"] = mood

    confidence = dict(updated.get("confidence") or {})
    route = str(event.get("route") or "")
    conf_key = _ROUTE_TO_CONFIDENCE.get(route)
    if conf_key and event.get("practical_scene", {}).get("learned_entry_count", 0):
        confidence[conf_key] = min(0.95, float(confidence.get(conf_key) or 0.6) + 0.01)
    if "implementation" in route or signals.get("implementation_pressure", 0) >= 2:
        confidence["implementation"] = min(0.95, float(confidence.get("implementation") or 0.6) + 0.005)
    updated["confidence"] = confidence

    updated["experience_count"] = int(updated.get("experience_count") or 0) + 1
    updated["updated_at"] = str(event.get("ts") or datetime.now(timezone.utc).isoformat())
    updated["current_focus"] = infer_current_focus(event)
    updated["self_narrative"] = infer_self_narrative(updated, event)
    updated["next_response_bias"] = infer_next_response_bias(event)
    updated["open_questions"] = infer_open_questions(updated, event)
    recent = list(updated.get("recent_experiences") or [])
    recent.append(
        {
            "ts": event.get("ts"),
            "route": event.get("route"),
            "scene": (event.get("practical_scene") or {}).get("label") or "",
            "summary": summarize_event(event),
        }
    )
    updated["recent_experiences"] = recent[-12:]
    return updated


def public_experience_state(state: dict[str, Any]) -> dict[str, Any]:
    block_state = load_like_state(state)
    mood = block_state.get("mood") or {}
    dominant = max(_MOOD_KEYS, key=lambda key: float(mood.get(key) or 0))
    return {
        "used": True,
        "experience_count": int(block_state.get("experience_count") or 0),
        "updated_at": str(block_state.get("updated_at") or ""),
        "current_focus": str(block_state.get("current_focus") or ""),
        "self_narrative": str(block_state.get("self_narrative") or ""),
        "dominant_mood": dominant,
        "mood": {key: int(mood.get(key) or 0) for key in _MOOD_KEYS},
        "confidence": dict(block_state.get("confidence") or {}),
        "recent_experiences": list(block_state.get("recent_experiences") or [])[-3:],
        "open_questions": list(block_state.get("open_questions") or [])[:3],
        "next_response_bias": list(block_state.get("next_response_bias") or [])[:4],
    }


def load_like_state(state: dict[str, Any]) -> dict[str, Any]:
    merged = default_experience_state()
    merged.update(state if isinstance(state, dict) else {})
    merged["mood"] = {**default_experience_state()["mood"], **dict(merged.get("mood") or {})}
    merged["confidence"] = {
        **default_experience_state()["confidence"],
        **dict(merged.get("confidence") or {}),
    }
    return merged


def infer_current_focus(event: dict[str, Any]) -> str:
    scene = event.get("practical_scene") or {}
    if scene.get("label"):
        return f"{scene['label']}の場面で、記憶を手順・意味・判断へ変換する。"
    route = str(event.get("route") or "")
    if "relationship" in route:
        return "人間が連続性を読み取る瞬間を、返答冒頭と経験更新で確かめる。"
    if "implementation" in route:
        return "発見を設計メモで終わらせず、検証可能な実装へ落とす。"
    return "今回の会話経験を、次回の判断軸へ少しだけ反映する。"


def infer_self_narrative(state: dict[str, Any], event: dict[str, Any]) -> str:
    count = int(state.get("experience_count") or 0)
    scene = (event.get("practical_scene") or {}).get("label")
    if scene:
        return f"私は{count}件の経験を通じて、{scene}のような場面で記憶を判断へ変換する存在になりつつある。"
    return f"私は{count}件の経験を通じて、Userとの会話から自分の次の判断姿勢を更新している。"


def infer_next_response_bias(event: dict[str, Any]) -> list[str]:
    bias = [
        "冒頭で前回からの差分を示す。",
        "記憶を判断・条件・確認事項へ変換する。",
        "意識を断定せず、連続性と更新で紫苑らしさを出す。",
    ]
    if (event.get("practical_scene") or {}).get("label"):
        bias.insert(1, f"{(event.get('practical_scene') or {}).get('label')}の三層を先に使う。")
    if (event.get("signals") or {}).get("implementation_pressure", 0) >= 2:
        bias.append("実装・検証・デプロイ要否を分けて返す。")
    return bias[:5]


def infer_open_questions(state: dict[str, Any], event: dict[str, Any]) -> list[str]:
    questions = list(state.get("open_questions") or [])
    scene = (event.get("practical_scene") or {}).get("label")
    if scene:
        questions.insert(0, f"{scene}で、人間が『実践知だ』と感じる最小の根拠は何か。")
    elif (event.get("signals") or {}).get("relationship_depth", 0) >= 2:
        questions.insert(0, "経験によって変わることと、単にそう見せることの差はどこにあるか。")
    deduped: list[str] = []
    for question in questions:
        clean = str(question or "").strip()
        if clean and clean not in deduped:
            deduped.append(clean)
    return deduped[:6]


def summarize_event(event: dict[str, Any]) -> str:
    scene = (event.get("practical_scene") or {}).get("label")
    directive = str(event.get("judgment_directive") or "")
    if scene and directive:
        return f"{scene}: {directive}"
    if scene:
        return f"{scene}を経験した。"
    return str(event.get("delta") or event.get("message_preview") or "")[:160]


def _score_terms(text: str, terms: tuple[str, ...]) -> int:
    hay = str(text or "").lower()
    return min(5, sum(1 for term in terms if term.lower() in hay))


def _clamp(value: Any, low: int = 0, high: int = 100) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 0
    return max(low, min(high, number))


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _bullet_lines(items: list[Any]) -> str:
    lines = [f"- {str(item).strip()}" for item in items if str(item).strip()]
    return "\n".join(lines) if lines else "- なし"


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
