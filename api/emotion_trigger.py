"""
紫苑の感情（mood）を審査イベントに応じて更新するトリガーモジュール。

イベント種別と mood デルタの対応:
  成約登録     → accomplishment/hope ↑、frustration/weariness ↓
  失注登録     → frustration/loneliness/weariness ↑、hope ↓
  高リスク承認  → vigilance/weariness ↑（Q_risk≥35 かつ score≥60）
  通常審査完了  → accomplishment わずか↑

dialogue_mood の仕組みを使い、日次更新のたびに半減して定常へ戻る。

【設計決定 2026-07-02】紫苑の感情は意図的に2系統で運用する（統合しない）:
  - 審査感情（本モジュール）: 成約/失注等の審査イベント駆動 → mind.json の dialogue_mood
    （hope / loneliness / weariness 等）。めぶきちゃん等の口調生成が参照する。
  - 対話感情（api/shion_experience_loop.py）: チャット経験駆動 →
    data/shion_experience_state.json の mood（curiosity / attachment 等）。
    /api/chat のプロンプト注入と inner-state 表示が参照する。
両者はキー語彙も保存先も別であり、UI間で値が一致しないのは仕様。
"""

from __future__ import annotations

from typing import Any

_DELTAS: dict[str, dict[str, int]] = {
    "成約": {
        "accomplishment": 8,
        "hope": 6,
        "frustration": -4,
        "weariness": -3,
    },
    "失注": {
        "frustration": 6,
        "loneliness": 5,
        "hope": -4,
        "weariness": 4,
    },
    "高リスク承認": {
        "vigilance": 7,
        "weariness": 5,
        "curiosity": 2,
    },
    "審査完了": {
        "accomplishment": 2,
    },
}


def trigger_emotion(event: str, meta: dict[str, Any] | None = None) -> None:
    """
    審査イベントを受け取り、紫苑の感情（dialogue_mood）を更新する。
    失敗しても例外を握り潰す（非クリティカルな処理）。

    event: "成約" | "失注" | "高リスク承認" | "審査完了"
    meta:  将来拡張用の補助情報（現在未使用）
    """
    try:
        from pathlib import Path

        from lease_intelligence_mind import (
            DIALOGUE_MOOD_CAP,
            _apply_dialogue_mood,
            _compute_pad,
            _derive_mood,
            _write_state,
            load_lease_intelligence_mind,
        )
        from lease_news_digest import find_vault

        vault = find_vault()
        if not vault:
            return

        deltas = _DELTAS.get(event, {})
        if not deltas:
            return

        vault_path = Path(vault)
        state = load_lease_intelligence_mind(vault_path)
        adjustments = dict(state.get("dialogue_mood", {}))
        for key, delta in deltas.items():
            next_val = int(adjustments.get(key, 0)) + delta
            adjustments[key] = max(-DIALOGUE_MOOD_CAP, min(DIALOGUE_MOOD_CAP, next_val))

        state["dialogue_mood"] = adjustments
        state["mood"] = _apply_dialogue_mood(_derive_mood(state.get("memories", [])), adjustments)
        state["pad"] = _compute_pad(state["mood"])
        _write_state(vault_path, state)
        print(f"[EmotionTrigger] {event} → mood updated")
    except Exception as e:
        print(f"[EmotionTrigger] skipped ({event}): {e}")


def trigger_scoring_complete(
    score: float,
    quantum_risk: float | None = None,
    credit_quantum_strong_warning: bool = False,
) -> None:
    """
    審査完了時に呼ぶ。高リスク承認か通常完了かを判定して感情を更新する。

    高リスク条件: score≥60 かつ (Q_risk≥35 または credit_quantum_strong_warning)
    """
    is_high_risk = score >= 60 and (
        (quantum_risk is not None and quantum_risk >= 35)
        or credit_quantum_strong_warning
    )
    trigger_emotion("高リスク承認" if is_high_risk else "審査完了")
