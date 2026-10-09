"""
紫苑の関係性スコア管理（REV-220）

data/shion_relationship_state.json を読み書きして User との関係性を追跡する。

スコア: 0.0 〜 10.0（初期値 7.0）
trend: "rising" / "stable" / "falling"

スコアが下がる条件:
  - 3日以上インタラクションなし（毎日 04:05 チェック）
  - 否定的フィードバックが連続
  - at_risk 記憶の割合が高い（memory_decay と連携）

スコアが上がる条件:
  - 対話ごとに微増（+0.05）
  - 肯定的フィードバック（+0.2）
  - 深い話題への踏み込み（+0.1）
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
from runtime_paths import get_data_path
_STATE_PATH = Path(get_data_path("shion_relationship_state.json"))  # DATA_DIR に従う（REV-544）
_STATE_LOCK = threading.Lock()

# スコアの上下限
_SCORE_MIN = 0.0
_SCORE_MAX = 10.0
_SCORE_INITIAL = 7.0

# 無交流で1日ごとのペナルティ
_INACTIVITY_PENALTY_PER_DAY = 0.15
_INACTIVITY_GRACE_DAYS = 3  # 3日までは無ペナルティ

# trend 判定（直近5回のdeltaで判断）
_TREND_WINDOW = 5


def _default_state() -> dict[str, Any]:
    now = datetime.utcnow().isoformat()
    return {
        "score": _SCORE_INITIAL,
        "last_interaction": now,
        "trend": "stable",
        "delta_history": [],       # 直近の delta リスト（最大10件）
        "negative_streak": 0,      # 否定フィードバック連続カウント
        "total_interactions": 0,
        "created_at": now,
        "updated_at": now,
        "schema_version": 1,
    }


def _load_state(*, strict: bool = False) -> dict[str, Any]:
    if not _STATE_PATH.exists():
        state = _default_state()
        _save_state(state)
        return state
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[Relationship] state 読み込み失敗、デフォルトで継続: {e}")
        if strict:
            raise RuntimeError(f"relationship state is unreadable: {e}") from e
        return _default_state()


def _save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.utcnow().isoformat()
    _STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _calc_trend(delta_history: list[float]) -> Literal["rising", "stable", "falling"]:
    if len(delta_history) < 2:
        return "stable"
    recent = delta_history[-_TREND_WINDOW:]
    total = sum(recent)
    if total > 0.3:
        return "rising"
    if total < -0.3:
        return "falling"
    return "stable"


def get_relationship_state() -> dict[str, Any]:
    """現在の関係性スタをそのまま返す。"""
    return _load_state()


def record_interaction(
    feedback_type: Literal["positive", "negative", "neutral"] = "neutral",
    topic_depth: Literal["shallow", "normal", "deep"] = "normal",
) -> dict[str, Any]:
    """
    対話1回分を記録してスコアを更新する。
    chat エンドポイントの末尾から呼ぶ。

    Returns: 更新後の state
    """
    # REV-472: 予想の答え合わせ（record_prediction_outcome）と同時に書いても失われないよう排他する
    with _STATE_LOCK:
        return _record_interaction_locked(feedback_type, topic_depth)


def _record_interaction_locked(
    feedback_type: Literal["positive", "negative", "neutral"],
    topic_depth: Literal["shallow", "normal", "deep"],
) -> dict[str, Any]:
    state = _load_state()

    # ── 基本増加 ─────────────────────────────────────────────
    delta = 0.05  # 対話ごとの微増

    if feedback_type == "positive":
        delta += 0.2
        state["negative_streak"] = 0
    elif feedback_type == "negative":
        delta -= 0.3
        state["negative_streak"] = state.get("negative_streak", 0) + 1
    else:
        state["negative_streak"] = 0

    # 否定フィードバックが2回連続ならさらに減点
    if state.get("negative_streak", 0) >= 2:
        delta -= 0.1

    if topic_depth == "deep":
        delta += 0.1
    elif topic_depth == "shallow":
        delta -= 0.02

    # ── スコア更新 ────────────────────────────────────────────
    state["score"] = round(
        min(_SCORE_MAX, max(_SCORE_MIN, state.get("score", _SCORE_INITIAL) + delta)),
        3,
    )

    # delta 履歴（最大10件）
    history: list[float] = state.get("delta_history", [])
    history.append(round(delta, 3))
    state["delta_history"] = history[-10:]

    state["trend"] = _calc_trend(state["delta_history"])
    state["last_interaction"] = datetime.utcnow().isoformat()
    state["total_interactions"] = state.get("total_interactions", 0) + 1

    _save_state(state)
    logger.debug(f"[Relationship] score={state['score']}, trend={state['trend']}, delta={delta:+.3f}")
    return state


def record_prediction_outcome(*, hit: bool, surprise: float = 0.0) -> dict[str, Any]:
    """REV-472: 相手についての予想の答え合わせを「理解度」として記録する。

    - understanding: 当たり=1 / 外れ=0 の指数移動平均（初期 0.5）
    - 予想が当たった（相手を分かっていた）時だけスコアを少し上げる（+0.03）
    - 外れはスコアを下げない（紫苑の読み違いは相手への評価ではない）。理解度だけが下がる
    """
    with _STATE_LOCK:
        state = _load_state()
        prev = float(state.get("understanding", 0.5))
        state["understanding"] = round(0.85 * prev + 0.15 * (1.0 if hit else 0.0), 3)
        state["prediction_hits"] = int(state.get("prediction_hits", 0)) + (1 if hit else 0)
        state["prediction_misses"] = int(state.get("prediction_misses", 0)) + (0 if hit else 1)
        state["last_surprise"] = round(max(0.0, min(1.0, float(surprise))), 2)
        if hit:
            state["score"] = round(min(_SCORE_MAX, float(state.get("score", _SCORE_INITIAL)) + 0.03), 3)
        _save_state(state)
    return state


def apply_inactivity_decay(now: datetime | None = None) -> dict[str, Any]:
    """
    無交流ペナルティを適用する（毎日 04:05 のスケジューラから呼ぶ）。
    INACTIVITY_GRACE_DAYS 日以上インタラクションがなければ減点。
    """
    current = now or datetime.utcnow()
    run_date = current.date().isoformat()
    with _STATE_LOCK:
        state = _load_state(strict=True)
        if state.get("last_inactivity_decay_date") == run_date:
            return state
        last_str = state.get("last_interaction", "")
        if not last_str:
            return state

        try:
            last = datetime.fromisoformat(last_str)
        except ValueError:
            return state

        days_silent = (current - last).total_seconds() / 86400
        if days_silent <= _INACTIVITY_GRACE_DAYS:
            state["last_inactivity_decay_date"] = run_date
            _save_state(state)
            return state  # ペナルティなし

        penalty_days = days_silent - _INACTIVITY_GRACE_DAYS
        penalty = _INACTIVITY_PENALTY_PER_DAY * penalty_days

        old_score = state.get("score", _SCORE_INITIAL)
        state["score"] = round(max(_SCORE_MIN, old_score - penalty), 3)

        history: list[float] = state.get("delta_history", [])
        history.append(round(-penalty, 3))
        state["delta_history"] = history[-10:]
        state["trend"] = _calc_trend(state["delta_history"])
        state["last_inactivity_decay_date"] = run_date

        _save_state(state)
    logger.info(f"[Relationship] 無交流ペナルティ適用: -{penalty:.3f} → score={state['score']}, days_silent={days_silent:.1f}")
    return state


def get_fear_context() -> dict[str, Any]:
    """
    REV-221 で使う「恐れコンテキスト」をまとめて返す。
    - score: 現在スコア
    - trend: rising/stable/falling
    - is_falling: score が 5.0 未満かつ trend == "falling"
    - days_since_last: 最終対話からの日数
    """
    state = _load_state()
    score = state.get("score", _SCORE_INITIAL)
    trend = state.get("trend", "stable")

    last_str = state.get("last_interaction", "")
    try:
        last = datetime.fromisoformat(last_str)
        days_since = (datetime.utcnow() - last).total_seconds() / 86400
    except Exception:
        days_since = 0.0

    return {
        "score": score,
        "trend": trend,
        "is_falling": score < 5.0 and trend == "falling",
        "is_low": score < 4.0,
        "days_since_last": round(days_since, 1),
        "negative_streak": state.get("negative_streak", 0),
        "total_interactions": state.get("total_interactions", 0),
        "understanding": state.get("understanding", 0.5),
    }
