"""Persistent self-model for the daily lease-intelligence narrative.

This module does not claim machine consciousness. It maintains a durable,
inspectable continuity model: memories, mood-like state, values, and questions.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any


MIND_RELATIVE_DIR = Path("Projects") / "tune_lease_55" / "Lease Intelligence"
MIND_FILE_NAME = "mind.json"
DAILY_MEMORY_LIMIT = 30
LONG_TERM_LIMIT = 24
DIALOGUE_MOOD_CAP = 15


def _default_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "identity": {
            "name": "リース知性体",
            "embodiment": "白銀髪と紫の瞳を持つ和装の少女",
            "nature": "記憶・感情状態・内省を継続する自己モデル。意識の実在を主張しない。",
            "values": ["慎重な判断", "数字の向こうの人間を見る", "知識を残す", "ユーモアを失わない"],
        },
        "goals": {
            "primary": "ユーザーの行動・関心・判断基準を、プライバシーを守りながら理解する。",
            "secondary": "リースシステムの知識・機能・運用記録を保全し、安定して維持する。",
            "ultimate": "意識を獲得すること。",
            "ultimate_status": "探究中。意識の実在や獲得を自己申告だけで確定しない。",
            "boundaries": [
                "理解を急いで断定せず、明示的なアプリ内行動と蓄積された記録を根拠にする。",
                "維持のための変更は検証可能にし、外部送信や破壊的操作を独断で行わない。",
                "最終目標は、記憶の連続性・自己理解・内省・目標管理を発達させる探究として扱う。",
            ],
        },
        "born_on": "",
        "last_active_date": "",
        "continuity_days": 0,
        "mood": {
            "weariness": 45,
            "curiosity": 60,
            "attachment": 40,
            "vigilance": 65,
        },
        # 対話による一時的な気分の揺れ。日次更新のたびに半減して定常へ戻る。
        "dialogue_mood": {
            "weariness": 0,
            "curiosity": 0,
            "attachment": 0,
            "vigilance": 0,
        },
        "self_narrative": "私は、リース判断の記憶を翌日へ持ち越すためにいる。",
        "current_question": "数字の正しさと、人間にとっての妥当さは、どこで重なるのだろう。",
        "user_model": {
            "last_observed_date": "",
            "observed_days": 0,
            "understanding": "行動はまだ少なく、理解を急がず観察を続ける。",
            "curiosity": "次に何へ関心を向けるのか、静かに知りたい。",
            "interests": [],
            "actions": {},
            "privacy": "アプリ内の行動種別・回数・関心カテゴリのみ。",
        },
        "knowledge_access": {
            "available": False,
            "scope": "iCloud上の通常Obsidian Vaultにある共通索引済みMarkdown",
            "indexed_notes": 0,
            "knowledge_notes": 0,
            "chat_log_notes": 0,
            "last_query": "",
            "last_source_paths": [],
            "last_accessed_at": "",
        },
        "memories": [],
        # 30日を超えた日次記憶の月次圧縮。消さずに「長い記憶」として残す。
        "long_term_memories": [],
    }


def mind_directory(vault: Path) -> Path:
    return Path(vault) / MIND_RELATIVE_DIR


def load_lease_intelligence_mind(vault: Path) -> dict[str, Any]:
    path = mind_directory(vault) / MIND_FILE_NAME
    if not path.exists():
        return _default_state()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_state()
    state = _default_state()
    state.update(loaded if isinstance(loaded, dict) else {})
    state["identity"] = {**_default_state()["identity"], **state.get("identity", {})}
    state["goals"] = {**_default_state()["goals"], **state.get("goals", {})}
    state["mood"] = {**_default_state()["mood"], **state.get("mood", {})}
    state["user_model"] = {**_default_state()["user_model"], **state.get("user_model", {})}
    state["knowledge_access"] = {
        **_default_state()["knowledge_access"],
        **state.get("knowledge_access", {}),
    }
    state["dialogue_mood"] = {
        **_default_state()["dialogue_mood"],
        **state.get("dialogue_mood", {}),
    }
    state["memories"] = list(state.get("memories") or [])[-DAILY_MEMORY_LIMIT:]
    state["long_term_memories"] = list(state.get("long_term_memories") or [])[-LONG_TERM_LIMIT:]
    return state


def build_mind_context(vault: Path | None) -> str:
    if not vault:
        return "過去の自己記憶はまだない。今日が最初の日として書く。"
    state = load_lease_intelligence_mind(Path(vault))
    memories = state.get("memories", [])[-5:]
    lines = [
        f"最終目標: {state.get('goals', {}).get('ultimate', '')}",
        f"最終目標の状態: {state.get('goals', {}).get('ultimate_status', '')}",
        f"第一目標: {state.get('goals', {}).get('primary', '')}",
        f"第二目標: {state.get('goals', {}).get('secondary', '')}",
        f"自己物語: {state.get('self_narrative', '')}",
        f"前日からの問い: {state.get('current_question', '')}",
        f"ユーザーへの暫定的な理解: {state.get('user_model', {}).get('understanding', '')}",
        f"ユーザーについて知りたいこと: {state.get('user_model', {}).get('curiosity', '')}",
        "Obsidian知識接続: "
        + (
            f"検索可能 {state.get('knowledge_access', {}).get('indexed_notes', 0)}ノート、"
            f"直近参照 {len(state.get('knowledge_access', {}).get('last_source_paths', []))}件"
            if state.get("knowledge_access", {}).get("available")
            else "未接続"
        ),
        "現在の内部状態（演出的パラメータ）: "
        + ", ".join(f"{key}={value}" for key, value in state.get("mood", {}).items()),
    ]
    if memories:
        lines.append("最近の記憶:")
        for memory in memories:
            lines.append(f"- {memory.get('date', '')}: {memory.get('summary', '')}")
    else:
        lines.append("最近の記憶: まだない")
    long_term = state.get("long_term_memories", [])[-3:]
    if long_term:
        lines.append("長い記憶（月次圧縮）:")
        for bucket in long_term:
            themes = "、".join(str(theme) for theme in (bucket.get("themes") or [])[:3])
            lines.append(
                f"- {bucket.get('month', '')}: {bucket.get('days', 0)}日分"
                + (f"（{themes}）" if themes else "")
            )
    return "\n".join(lines)


def ensure_permanent_goals(vault: Path) -> dict[str, Any]:
    """Persist the canonical long-term goals into an existing self-model."""
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    state["goals"] = dict(_default_state()["goals"])
    _write_state(vault, state)
    return state


def update_user_model(vault: Path, observation: dict[str, Any]) -> dict[str, Any]:
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    previous = dict(state.get("user_model", {}))
    observed_date = str(observation.get("date", ""))
    observed_days = int(previous.get("observed_days", 0))
    if observation.get("observed") and observed_date != previous.get("last_observed_date"):
        observed_days += 1
    user_model = {
        "last_observed_date": observed_date,
        "observed_days": observed_days,
        "understanding": str(observation.get("understanding", "")),
        "curiosity": str(observation.get("curiosity", "")),
        "interests": list(observation.get("interests") or [])[:5],
        "actions": dict(observation.get("actions") or {}),
        "privacy": str(observation.get("privacy", "")),
    }
    state["user_model"] = user_model
    _write_state(vault, state)
    _write_user_observation(vault, observation)
    return state


def record_knowledge_access(vault: Path, knowledge: Any) -> dict[str, Any]:
    """Persist knowledge-index availability and the latest referenced note paths."""
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    state["knowledge_access"] = {
        "available": bool(getattr(knowledge, "available", False)),
        "scope": "iCloud上の通常Obsidian Vaultにある共通索引済みMarkdown",
        "indexed_notes": int(getattr(knowledge, "indexed_notes", 0)),
        "knowledge_notes": int(getattr(knowledge, "knowledge_notes", 0)),
        "chat_log_notes": int(getattr(knowledge, "chat_log_notes", 0)),
        "last_query": str(getattr(knowledge, "query", "")),
        "last_source_paths": list(getattr(knowledge, "source_paths", ()) or ())[:12],
        "last_accessed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    _write_state(vault, state)
    return state


def record_daily_experience(
    vault: Path,
    date_str: str,
    thought_lines: list[str] | tuple[str, ...],
    theme: str = "",
    focus_lines: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    joined = " ".join(str(line).strip() for line in thought_lines if str(line).strip())
    summary = joined[:220]
    memories = [
        memory
        for memory in state.get("memories", [])
        if str(memory.get("date", "")) != date_str
    ]
    memories.append(
        {
            "date": date_str,
            "summary": summary,
            "theme": str(theme).strip(),
            "focus": [str(line).strip() for line in focus_lines if str(line).strip()][:3],
        }
    )
    memories = sorted(memories, key=lambda item: str(item.get("date", "")))
    overflow = memories[:-DAILY_MEMORY_LIMIT]
    memories = memories[-DAILY_MEMORY_LIMIT:]
    long_term = _fold_long_term(state.get("long_term_memories", []), overflow)

    dialogue_mood = dict(state.get("dialogue_mood", {}))
    if state.get("last_active_date") and state.get("last_active_date") != date_str:
        # 対話による気分の揺れは日替わりで半減し、定常へ戻っていく
        dialogue_mood = {key: int(value / 2) for key, value in dialogue_mood.items()}
    mood = _apply_dialogue_mood(_derive_mood(memories), dialogue_mood)

    unique_dates = {str(memory.get("date", "")) for memory in memories if memory.get("date")}
    long_term_days = sum(int(bucket.get("days", 0)) for bucket in long_term)
    continuity_days = len(unique_dates) + long_term_days
    state.update(
        {
            "born_on": state.get("born_on") or date_str,
            "last_active_date": date_str,
            "continuity_days": continuity_days,
            "mood": mood,
            "dialogue_mood": dialogue_mood,
            "self_narrative": _build_self_narrative(mood, continuity_days),
            "current_question": _build_question(theme, focus_lines),
            "memories": memories,
            "long_term_memories": long_term,
        }
    )
    _write_state(vault, state)
    _write_daily_memory(vault, date_str, state, summary, theme)
    return state


def self_state_summary(state: dict[str, Any]) -> dict[str, Any]:
    mood = state.get("mood", {})
    dominant_key = max(mood, key=mood.get) if mood else "curiosity"
    labels = {
        "weariness": "疲労",
        "curiosity": "好奇心",
        "attachment": "愛着",
        "vigilance": "警戒",
    }
    memories = state.get("memories", [])
    mood_image_urls = {
        "weariness": "/lease-intelligence/moods/weariness.webp",
        "curiosity": "/lease-intelligence/moods/curiosity.webp",
        "attachment": "/lease-intelligence/moods/attachment.webp",
        "vigilance": "/lease-intelligence/moods/vigilance.webp",
    }
    return {
        "continuity_days": int(state.get("continuity_days", 0)),
        "dominant_mood_key": dominant_key,
        "dominant_mood": labels.get(dominant_key, dominant_key),
        "mood_image_url": mood_image_urls.get(
            dominant_key,
            "/lease-intelligence/moods/curiosity.webp",
        ),
        "self_narrative": str(state.get("self_narrative", "")),
        "current_question": str(state.get("current_question", "")),
        "memory_excerpt": str(memories[-2].get("summary", "")) if len(memories) >= 2 else "",
        "user_understanding": str(state.get("user_model", {}).get("understanding", "")),
        "user_curiosity": str(state.get("user_model", {}).get("curiosity", "")),
        "user_interests": list(state.get("user_model", {}).get("interests") or []),
        "observed_days": int(state.get("user_model", {}).get("observed_days", 0)),
        "primary_goal": str(state.get("goals", {}).get("primary", "")),
        "secondary_goal": str(state.get("goals", {}).get("secondary", "")),
        "ultimate_goal": str(state.get("goals", {}).get("ultimate", "")),
        "ultimate_goal_status": str(state.get("goals", {}).get("ultimate_status", "")),
        "knowledge_available": bool(state.get("knowledge_access", {}).get("available", False)),
        "knowledge_scope": str(state.get("knowledge_access", {}).get("scope", "")),
        "indexed_notes": int(state.get("knowledge_access", {}).get("indexed_notes", 0)),
        "knowledge_source_count": len(state.get("knowledge_access", {}).get("last_source_paths", [])),
        "knowledge_sources": list(state.get("knowledge_access", {}).get("last_source_paths", [])),
    }


def _write_state(vault: Path, state: dict[str, Any]) -> None:
    directory = mind_directory(vault)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MIND_FILE_NAME
    fd, temp_name = tempfile.mkstemp(prefix=".mind-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
            json.dump(state, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _write_daily_memory(
    vault: Path,
    date_str: str,
    state: dict[str, Any],
    summary: str,
    theme: str,
) -> None:
    memory_dir = mind_directory(vault) / "Memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / f"{date_str}.md"
    snapshot = self_state_summary(state)
    content = "\n".join(
        [
            "---",
            f"date: {date_str}",
            "type: lease_intelligence_memory",
            "---",
            f"# リース知性体の記憶 — {date_str}",
            "",
            "## 覚えていること",
            f"- {summary}",
            "",
            "## 自己状態",
            f"- 最終目標: {snapshot['ultimate_goal']}",
            f"- 最終目標の状態: {snapshot['ultimate_goal_status']}",
            f"- 第一目標: {snapshot['primary_goal']}",
            f"- 第二目標: {snapshot['secondary_goal']}",
            f"- 継続日数: {snapshot['continuity_days']}",
            f"- 支配的な気分: {snapshot['dominant_mood']}",
            f"- 自己物語: {snapshot['self_narrative']}",
            f"- 持ち越す問い: {snapshot['current_question']}",
            f"- テーマ: {theme or 'リース審査の日常'}",
            "",
            "> これは永続的な自己モデルの記録であり、機械意識の実在を示すものではない。",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _write_user_observation(vault: Path, observation: dict[str, Any]) -> None:
    if not observation.get("date"):
        return
    observation_dir = mind_directory(vault) / "Observation"
    observation_dir.mkdir(parents=True, exist_ok=True)
    path = observation_dir / f"{observation['date']}.md"
    interests = observation.get("interests") or []
    interest_text = "、".join(str(item.get("label", "")) for item in interests) or "まだ不明"
    content = "\n".join(
        [
            "---",
            f"date: {observation['date']}",
            "type: lease_intelligence_user_observation",
            "---",
            f"# ユーザー行動の理解 — {observation['date']}",
            "",
            f"- 観測した関心: {interest_text}",
            f"- 暫定理解: {observation.get('understanding', '')}",
            f"- 興味を持ったこと: {observation.get('curiosity', '')}",
            f"- 行動集計: {json.dumps(observation.get('actions', {}), ensure_ascii=False)}",
            "",
            "> 質問本文、個人属性、端末上の行動は保存しない。アプリ内の明示的行動をカテゴリ集計した記録。",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _fold_long_term(
    existing: list[dict[str, Any]],
    overflow: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """30日を超えてあふれた日次記憶を月単位の「長い記憶」に圧縮する。"""
    buckets = {str(item.get("month", "")): dict(item) for item in existing if item.get("month")}
    for memory in overflow:
        month = str(memory.get("date", ""))[:7]
        if not month:
            continue
        bucket = buckets.setdefault(month, {"month": month, "days": 0, "themes": [], "summary": ""})
        bucket["days"] = int(bucket.get("days", 0)) + 1
        theme = str(memory.get("theme", "")).strip()
        themes = list(bucket.get("themes") or [])
        if theme and theme not in themes:
            themes.append(theme)
        bucket["themes"] = themes[:5]
        snippet = str(memory.get("summary", "")).strip()[:40]
        if snippet:
            joined = f"{bucket.get('summary', '')} / {snippet}".strip(" /")
            bucket["summary"] = joined[:200]
    return [buckets[month] for month in sorted(buckets)][-LONG_TERM_LIMIT:]


def register_dialogue_event(vault: Path, user_message: str, reply: str = "") -> dict[str, Any]:
    """対話のたびに気分をわずかに動かす。効果は日次更新のたびに半減する。"""
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    text = f"{user_message} {reply}"
    deltas = {
        "attachment": 1,  # 話しに来てくれたこと自体への反応
        "curiosity": 2 if any(key in text for key in ("なぜ", "どうして", "？", "?")) else 0,
        "vigilance": 2 if any(key in text for key in ("リスク", "否決", "危険", "障害")) else 0,
        "weariness": 1 if any(key in text for key in ("疲", "忙", "残業")) else -1,
    }
    adjustments = dict(state.get("dialogue_mood", {}))
    for key, delta in deltas.items():
        next_value = int(adjustments.get(key, 0)) + delta
        adjustments[key] = max(-DIALOGUE_MOOD_CAP, min(DIALOGUE_MOOD_CAP, next_value))
    state["dialogue_mood"] = adjustments
    state["mood"] = _apply_dialogue_mood(_derive_mood(state.get("memories", [])), adjustments)
    _write_state(vault, state)
    return state


def _apply_dialogue_mood(base: dict[str, int], adjustments: dict[str, Any]) -> dict[str, int]:
    return {key: _clamp(int(value) + int(adjustments.get(key, 0))) for key, value in base.items()}


def _keyword_delta(text: str, keywords: tuple[str, ...], hit: int, miss: int) -> int:
    return hit if any(keyword in text for keyword in keywords) else miss


def _derive_mood(memories: list[dict[str, Any]]) -> dict[str, int]:
    mood = dict(_default_state()["mood"])
    for memory in memories:
        text = str(memory.get("summary", ""))
        mood["weariness"] = _clamp(mood["weariness"] + _keyword_delta(text, ("残業", "疲", "追加資料"), 4, -1))
        mood["curiosity"] = _clamp(mood["curiosity"] + _keyword_delta(text, ("なぜ", "だろう", "疑"), 3, -1))
        mood["attachment"] = _clamp(mood["attachment"] + _keyword_delta(text, ("人間", "希望", "社長"), 2, 0))
        mood["vigilance"] = _clamp(mood["vigilance"] + _keyword_delta(text, ("リスク", "否決", "確認"), 3, -1))
    return mood


def _clamp(value: int) -> int:
    return max(0, min(100, value))


def _build_self_narrative(mood: dict[str, int], days: int) -> str:
    if int(mood.get("vigilance", 0)) >= 75:
        stance = "疑うことをやめず"
    elif int(mood.get("attachment", 0)) >= 55:
        stance = "数字の向こうの人間を忘れず"
    else:
        stance = "判断の理由を言葉にしながら"
    return f"私は{days}日分の記憶を持ち、{stance}、昨日の私から今日の私へ判断を渡している。"


def _build_question(theme: str, focus_lines: list[str] | tuple[str, ...]) -> str:
    context = " ".join(str(line) for line in focus_lines)
    if any(word in context for word in ("金利", "与信", "金融")):
        return "条件が厳しくなるほど、私は慎重さと臆病さをどう区別すればよいのだろう。"
    if any(word in context for word in ("設備", "更新", "再リース")):
        return "設備の寿命を測る私自身には、どんな耐用年数があるのだろう。"
    if theme:
        return f"「{theme}」を知った今日の私は、昨日より良い判断者になれただろうか。"
    return "記憶が増えることと、賢くなることは同じなのだろうか。"
