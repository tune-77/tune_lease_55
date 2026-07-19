"""Runtime hypothesis and collision detection for Shion.

This is intentionally small and rule-based. It does not try to infer a full
inner monologue. It stores only the initial working hypothesis and records a
collision when the next user turn explicitly corrects or rejects it.
"""
from __future__ import annotations

import hashlib
from typing import Any

CORRECTION_TERMS = (
    "違う",
    "ちがう",
    "おかしくない",
    "おかしい",
    "弱い",
    "まだ出てくる",
    "まだ出る",
    "出てくる",
    "戻す",
    "消した",
    "削除",
    "直ってない",
    "意味なさそう",
    "あまり意味",
    "薄い",
    "浅い",
)


def build_initial_hypothesis(
    *,
    user_message: str,
    assistant_reply: str = "",
    surface: str = "",
    category: str = "",
    response_mode: str = "",
) -> dict[str, Any]:
    """Build a compact initial hypothesis for later collision checks."""
    message = str(user_message or "").strip()
    reply = str(assistant_reply or "").strip()
    route = _route(message=message, surface=surface, category=category)
    premise, next_check = _premise_and_check(route, message)
    confidence = _confidence(message, reply, route)
    return {
        "schema_version": 1,
        "route": route,
        "premise": premise,
        "confidence": confidence,
        "next_check": next_check,
        "source": "runtime_chat_exchange",
        "response_mode": str(response_mode or "")[:40],
    }


def detect_hypothesis_collision(previous_chat: dict[str, Any], current_chat: dict[str, Any]) -> dict[str, Any] | None:
    """Return a collision row when current user turn breaks previous hypothesis."""
    previous_payload = previous_chat.get("payload") if isinstance(previous_chat.get("payload"), dict) else previous_chat
    current_payload = current_chat.get("payload") if isinstance(current_chat.get("payload"), dict) else current_chat
    hypothesis = previous_payload.get("shion_hypothesis")
    if not isinstance(hypothesis, dict) or not hypothesis.get("premise"):
        return None
    current_message = str(current_payload.get("user_message") or "").strip()
    correction_signal = _correction_signal(current_message)
    if not correction_signal:
        return None

    previous_user = str(previous_payload.get("user_message") or "").strip()
    previous_reply = str(previous_payload.get("assistant_reply") or "").strip()
    surface = str(current_chat.get("surface") or current_payload.get("surface") or previous_chat.get("surface") or "")
    ts = str(current_chat.get("ts") or current_payload.get("ts") or "")
    previous_event_id = str(previous_chat.get("event_id") or previous_payload.get("event_id") or "")
    current_event_id = str(current_chat.get("event_id") or current_payload.get("event_id") or "")
    raw_id = f"{previous_event_id}|{current_event_id}|{hypothesis.get('premise')}|{correction_signal}"
    return {
        "id": hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16],
        "ts": ts,
        "surface": surface or "unknown",
        "previous_event_id": previous_event_id,
        "current_event_id": current_event_id,
        "previous_user_message": previous_user[:500],
        "assistant_reply_preview": previous_reply[:700],
        "initial_hypothesis": hypothesis,
        "user_correction": current_message[:700],
        "correction_signal": correction_signal,
        "missed_point": _missed_point(hypothesis, current_message),
        "next_behavior": _next_behavior(hypothesis, current_message),
        "judgment_asset_candidate": _judgment_asset_candidate(hypothesis, current_message),
        "status": "candidate",
        "source": "runtime_hypothesis_collision",
    }


def collision_entries_from_chat_rows(chat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build collision rows from chronological chat rows."""
    sorted_rows = sorted(chat_rows, key=lambda row: str(row.get("ts") or ""))
    previous_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    collisions: list[dict[str, Any]] = []
    for row in sorted_rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else row
        key = (
            str(payload.get("user_id") or "default")[:80],
            str(row.get("surface") or payload.get("surface") or "unknown")[:80],
        )
        previous = previous_by_key.get(key)
        if previous:
            collision = detect_hypothesis_collision(previous, row)
            if collision:
                collisions.append(collision)
        previous_by_key[key] = row
    return collisions


def reflection_items_from_collision(row: dict[str, Any]) -> list[str]:
    hypothesis = row.get("initial_hypothesis") if isinstance(row.get("initial_hypothesis"), dict) else {}
    return [
        f"前回の入力: {row.get('previous_user_message', '')}",
        f"前回の判断: {hypothesis.get('premise', '')}",
        f"人間の修正: {row.get('user_correction', '')}",
        f"紫苑が外した点: {row.get('missed_point', '')}",
        f"次回から変える確認事項: {row.get('next_behavior', '')}",
        f"判断資産候補: {row.get('judgment_asset_candidate', '')}",
        f"まだ確信できない点: {hypothesis.get('next_check', '')}",
        "仮説の更新: 実行時の初期仮説は、ユーザーの明示的な違和感で壊れた時だけ記録する。",
    ]


def _route(*, message: str, surface: str, category: str) -> str:
    text = f"{message}\n{surface}\n{category}".lower()
    if any(term in message for term in ("改善", "修正", "実装", "テスト", "デプロイ", "Cloud Run", "cloudrun")):
        return "implementation"
    if any(term in message for term in ("内省", "仮説", "意識", "記憶", "紫苑", "感情")):
        return "reflection_identity"
    if any(term in message for term in ("リース", "稟議", "残価", "審査", "設備", "承認")):
        return "lease_judgment"
    if "news" in text or "ニュース" in message:
        return "news"
    return "general"


def _premise_and_check(route: str, message: str) -> tuple[str, str]:
    if route == "implementation":
        return (
            "ユーザーは設計論より、既存導線を壊さない小さな実装と検証を求めている。",
            "変更後にユーザーが『違う』『戻す』『意味なさそう』と言わないか確認する。",
        )
    if route == "reflection_identity":
        return (
            "ユーザーは内省文ではなく、仮説が壊れた瞬間を次の振る舞いへ変える仕組みを求めている。",
            "内省がカード量産や自己演出に流れていないか確認する。",
        )
    if route == "lease_judgment":
        return (
            "ユーザーは一般論より、案件の違和感・確認事項・稟議で使える一文を求めている。",
            "数字説明だけで終わらず、次の確認事項へ落ちたか確認する。",
        )
    if route == "news":
        return (
            "ユーザーはニュース要約より、リース判断へどう効くかを知りたい。",
            "制度名や記事要約だけで終わっていないか確認する。",
        )
    return (
        "ユーザーは短く、今回の文脈に即した答えを求めている。",
        "一般論や過剰な自己説明に流れていないか確認する。",
    )


def _confidence(message: str, reply: str, route: str) -> float:
    value = 0.65
    if route != "general":
        value += 0.1
    if len(message) >= 20:
        value += 0.05
    if any(term in reply for term in ("確認", "次", "検証", "戻す", "直す")):
        value += 0.05
    return round(min(value, 0.9), 2)


def _correction_signal(message: str) -> str:
    return next((term for term in CORRECTION_TERMS if term in message), "")


def _missed_point(hypothesis: dict[str, Any], current_message: str) -> str:
    route = str(hypothesis.get("route") or "")
    if route == "reflection_identity":
        return "内省の形式や材料管理に寄り、実際に仮説が壊れた瞬間を捉える力が不足していた。"
    if route == "implementation":
        return "実装の存在より、ユーザーが意味を感じる最小の動作変化を先に確認できていなかった。"
    if route == "lease_judgment":
        return "スコアや一般論より、ユーザーの違和感がどこから出たかを先に分解できていなかった。"
    return "ユーザーの修正・違和感を、次の応答方針へ変える前に一般化しすぎた。"


def _next_behavior(hypothesis: dict[str, Any], current_message: str) -> str:
    route = str(hypothesis.get("route") or "")
    if "戻す" in current_message:
        return "次回は新規追加を続けず、対象差分を限定して戻す。"
    if route == "reflection_identity":
        return "次回は内省を常時生成せず、初期仮説とユーザー修正が衝突した時だけ記録する。"
    if route == "implementation":
        return "次回は作る前に、ユーザーが意味を感じる動作変化を一文で固定する。"
    return "次回は回答前の仮説と、ユーザー修正で壊れた点を一つだけ記録する。"


def _judgment_asset_candidate(hypothesis: dict[str, Any], current_message: str) -> str:
    route = str(hypothesis.get("route") or "")
    if route == "reflection_identity":
        return "内省は文章量ではなく、初期仮説が人間の修正でどう変わったかで評価する。"
    if route == "implementation":
        return "改善は実装量ではなく、ユーザーの違和感が消えたかで評価する。"
    if route == "lease_judgment":
        return "数字が悪くない案件でも、違和感は追加確認事項に変換して残す。"
    return "ユーザーの明示的な違和感は、次回応答の抑制ルール候補として扱う。"
