"""
紫苑の関係性スコア管理（REV-220 / REV-598 で動き方を見直し）

data/shion_relationship_state.json を読み書きして User との関係性を追跡する。

スコア: 0.0 〜 10.0（初期値 7.0）
trend: "rising" / "stable" / "falling"

REV-598（2026-10-10）: 以前は対話ごとに必ず +0.03〜0.15 され、下がる経路がほぼ無かったため、
9/20 の 7.0 から 161 回の対話で上限 10.0 に張り付いていた。実際の関わりで上下するようにする。
- 1回の変化は、会話の間隔（同じ時間帯の続き／新しい会話／久しぶり）・相手の反応（お礼・喜び／訂正／
  紫苑への不満）・話題の深さ・予想の答え合わせ（様子の当たり・返答への不満）から決める
- 上がる分は上限に近いほど小さくする（飽和）。下がる分は下限に近いほど小さくする
- 1回の変化幅に上限（上げ +0.25・下げ -0.5）
- 時間がたつと中立（6.5）へゆっくり戻る（1日に差の4%）。3日を超える無交流は1日 0.15 下げる
- 変化は理由付きで events に残し、日ごとの値を score_history に残す（グラフ・点検用）
- 検証の会話（REV-591 の印）は数えない
"""
from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
from runtime_paths import get_data_path
_STATE_PATH = Path(get_data_path("shion_relationship_state.json"))  # DATA_DIR に従う（REV-544）
_STATE_LOCK = threading.Lock()

# スコアの上下限
_SCORE_MIN = 0.0
_SCORE_MAX = 10.0
_SCORE_INITIAL = 7.0
SCHEMA_VERSION = 2

# REV-598 時間による中立への戻り
SCORE_NEUTRAL = 6.5
REVERSION_PER_DAY = 0.04
# 飽和: 上げる分は (上限 - スコア) / この幅 を掛ける（スコア 7.5 で半分、9.5 で 1 割）。下げる分は下限側で同様
SATURATION_SPAN = 5.0
# 1回の変化幅の上限
MAX_STEP_UP = 0.25
MAX_STEP_DOWN = 0.5

# 会話の間隔
SAME_SESSION_HOURS = 2.0
LONG_SILENCE_DAYS = 3.0

# 1回の材料ごとの変化（飽和・上限をかける前）
DELTA_SAME_SESSION = 0.01
DELTA_NEW_SESSION = 0.05
DELTA_REUNION = 0.08
DELTA_DEEP = 0.03
DELTA_SHALLOW = 0.0
DELTA_POSITIVE = 0.12      # お礼・喜び（相手の発言から）
DELTA_POSITIVE_BUTTON = 0.2  # 画面の「良かった」ボタン
DELTA_CORRECTION = -0.12   # 「違う」「間違ってる」などの訂正
DELTA_NEGATIVE = -0.3      # 紫苑の返答への不満（「何度も」「ちゃんとして」）・「残念」ボタン
DELTA_NEGATIVE_STREAK = -0.1
DELTA_PREDICTION_HIT = 0.02
DELTA_REACTION_MISS = -0.08

# 無交流で1日ごとのペナルティ
_INACTIVITY_PENALTY_PER_DAY = 0.15
_INACTIVITY_GRACE_DAYS = 3  # 3日までは無ペナルティ

# trend 判定（直近の変化の合計で判断）
_TREND_WINDOW = 8
_TREND_THRESHOLD = 0.25

EVENTS_LIMIT = 60
HISTORY_DAYS = 120


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
        "schema_version": SCHEMA_VERSION,
        "reverted_at": now,
        "events": [],
        "score_history": {},
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
    total = sum(delta_history[-_TREND_WINDOW:])
    if total > _TREND_THRESHOLD:
        return "rising"
    if total < -_TREND_THRESHOLD:
        return "falling"
    return "stable"


def _parse(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(ts or "")).replace(tzinfo=None)
    except ValueError:
        return None


def saturate(score: float, delta: float) -> float:
    """上限に近いほど上がりにくく、下限に近いほど下がりにくくする（REV-598）。"""
    if delta > 0:
        return delta * max(0.0, min(1.0, (_SCORE_MAX - score) / SATURATION_SPAN))
    return delta * max(0.0, min(1.0, (score - _SCORE_MIN) / SATURATION_SPAN))


def revert_toward_neutral(score: float, days: float) -> float:
    """時間がたつと中立へゆっくり戻る（1日に差の REVERSION_PER_DAY）。"""
    if days <= 0:
        return score
    keep = math.pow(1.0 - REVERSION_PER_DAY, days)
    return SCORE_NEUTRAL + (score - SCORE_NEUTRAL) * keep


def interaction_parts(
    *,
    gap_hours: float | None,
    feedback_type: str = "neutral",
    topic_depth: str = "normal",
    signals: Iterable[str] = (),
    negative_streak: int = 0,
) -> list[tuple[str, float]]:
    """対話1回分の材料を (理由, 変化) で返す。飽和と上限をかける前の値（REV-598）。

    signals: user_affect の手がかり（"thanks" お礼 / "correction" 訂正 / "shion_complaint" 紫苑への不満）。
    feedback_type: 画面のボタン（positive/negative）または様子から決まる値。
    """
    sig = set(signals or ())
    parts: list[tuple[str, float]] = []
    if gap_hours is None or gap_hours >= LONG_SILENCE_DAYS * 24:
        parts.append(("久しぶりに話しかけてくれた", DELTA_REUNION))
    elif gap_hours >= SAME_SESSION_HOURS:
        parts.append(("新しい会話", DELTA_NEW_SESSION))
    else:
        parts.append(("会話の続き", DELTA_SAME_SESSION))
    if topic_depth == "deep":
        parts.append(("深い話題", DELTA_DEEP))
    if "shion_complaint" in sig or feedback_type == "negative":
        parts.append(("私の返答への不満", DELTA_NEGATIVE))
        if negative_streak >= 1:
            parts.append(("不満・訂正が続いた", DELTA_NEGATIVE_STREAK))
    elif "correction" in sig:
        parts.append(("訂正された", DELTA_CORRECTION))
        if negative_streak >= 1:
            parts.append(("不満・訂正が続いた", DELTA_NEGATIVE_STREAK))
    elif feedback_type == "positive_button":
        parts.append(("「良かった」の評価", DELTA_POSITIVE_BUTTON))
    elif "thanks" in sig or feedback_type == "positive":
        parts.append(("お礼・喜び", DELTA_POSITIVE))
    return parts


def apply_parts(score: float, parts: list[tuple[str, float]]) -> tuple[float, float]:
    """材料の合計に上限をかけ、飽和させて新しいスコアと実際の変化を返す。"""
    raw = sum(delta for _reason, delta in parts)
    raw = max(-MAX_STEP_DOWN, min(MAX_STEP_UP, raw))
    delta = saturate(score, raw)
    new = round(min(_SCORE_MAX, max(_SCORE_MIN, score + delta)), 3)
    return new, round(new - score, 3)


def _is_negative(parts: list[tuple[str, float]]) -> bool:
    return any(reason in {"私の返答への不満", "訂正された"} for reason, _delta in parts)


def _apply_reversion(state: dict[str, Any], now: datetime) -> None:
    last = _parse(state.get("reverted_at") or state.get("last_interaction") or "")
    if last is None:
        state["reverted_at"] = now.isoformat()
        return
    days = (now - last).total_seconds() / 86400
    if days <= 0:
        return
    before = float(state.get("score", _SCORE_INITIAL))
    state["score"] = round(revert_toward_neutral(before, days), 3)
    state["reverted_at"] = now.isoformat()
    shift = round(state["score"] - before, 3)
    if abs(shift) >= 0.005:
        _push_event(state, now, "時間による中立への戻り", shift)


def _push_event(state: dict[str, Any], now: datetime, reason: str, delta: float) -> None:
    events = list(state.get("events") or [])
    events.append({"ts": now.isoformat(timespec="seconds"), "reason": reason[:60], "delta": round(delta, 3),
                   "score": state.get("score")})
    state["events"] = events[-EVENTS_LIMIT:]


def _record_history(state: dict[str, Any], now: datetime) -> None:
    history = dict(state.get("score_history") or {})
    history[now.date().isoformat()] = state.get("score")
    state["score_history"] = dict(sorted(history.items())[-HISTORY_DAYS:])


def get_relationship_state() -> dict[str, Any]:
    """現在の関係性スタをそのまま返す。"""
    return _load_state()


def record_interaction(
    feedback_type: Literal["positive", "negative", "neutral", "positive_button"] = "neutral",
    topic_depth: Literal["shallow", "normal", "deep"] = "normal",
    *,
    signals: Iterable[str] = (),
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    対話1回分を記録してスコアを更新する。
    chat エンドポイントの末尾から呼ぶ。

    Returns: 更新後の state
    """
    from shion_verification_origin import is_verification_turn

    if is_verification_turn():  # REV-591: 検証の会話は関係性スコアに数えない
        return _load_state()
    # REV-472: 予想の答え合わせ（record_prediction_outcome）と同時に書いても失われないよう排他する
    with _STATE_LOCK:
        return _record_interaction_locked(feedback_type, topic_depth, tuple(signals or ()), now or datetime.utcnow())


def _record_interaction_locked(
    feedback_type: str,
    topic_depth: str,
    signals: tuple[str, ...],
    now: datetime,
) -> dict[str, Any]:
    state = _load_state()
    _apply_reversion(state, now)
    last = _parse(state.get("last_interaction") or "")
    gap_hours = None if last is None or not state.get("total_interactions") else (now - last).total_seconds() / 3600
    parts = interaction_parts(
        gap_hours=gap_hours, feedback_type=feedback_type, topic_depth=topic_depth, signals=signals,
        negative_streak=int(state.get("negative_streak", 0)),
    )
    state["negative_streak"] = int(state.get("negative_streak", 0)) + 1 if _is_negative(parts) or feedback_type == "negative" else 0
    state["score"], delta = apply_parts(float(state.get("score", _SCORE_INITIAL)), parts)

    history: list[float] = state.get("delta_history", [])
    history.append(delta)
    state["delta_history"] = history[-10:]
    state["trend"] = _calc_trend(state["delta_history"])
    state["previous_interaction"] = state.get("last_interaction", "")
    state["last_interaction"] = now.isoformat()
    state["total_interactions"] = state.get("total_interactions", 0) + 1
    _push_event(state, now, "・".join(reason for reason, _d in parts), delta)
    _record_history(state, now)

    _save_state(state)
    logger.debug(f"[Relationship] score={state['score']}, trend={state['trend']}, delta={delta:+.3f}")
    return state


def record_prediction_outcome(
    *, hit: bool, surprise: float = 0.0, affect_hit: bool | None = None, reaction_hit: bool | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """REV-472: 相手についての予想の答え合わせを「理解度」として記録する。

    - understanding: 当たり=1 / 外れ=0 の指数移動平均（初期 0.5）
    - REV-598: スコアには、相手の様子の予想が当たった時（+0.02）と、私の返答への不満が見えた時（-0.08）だけ
      反映する。話題の予想はほぼ毎回外れて雑音になるため使わない。affect_hit/reaction_hit が無い古い呼び出しは
      従来どおり hit の時だけ少し上げる
    """
    from shion_verification_origin import is_verification_turn

    if is_verification_turn():
        return _load_state()
    current = now or datetime.utcnow()
    with _STATE_LOCK:
        state = _load_state()
        prev = float(state.get("understanding", 0.5))
        state["understanding"] = round(0.85 * prev + 0.15 * (1.0 if hit else 0.0), 3)
        state["prediction_hits"] = int(state.get("prediction_hits", 0)) + (1 if hit else 0)
        state["prediction_misses"] = int(state.get("prediction_misses", 0)) + (0 if hit else 1)
        state["last_surprise"] = round(max(0.0, min(1.0, float(surprise))), 2)
        parts: list[tuple[str, float]] = []
        if reaction_hit is False:
            parts.append(("返答への不満が見えた（予想の答え合わせ）", DELTA_REACTION_MISS))
        elif affect_hit is True or (affect_hit is None and reaction_hit is None and hit):
            parts.append(("相手の様子の予想が当たった", DELTA_PREDICTION_HIT))
        if parts:
            state["score"], delta = apply_parts(float(state.get("score", _SCORE_INITIAL)), parts)
            history: list[float] = state.get("delta_history", [])
            history.append(delta)
            state["delta_history"] = history[-10:]
            state["trend"] = _calc_trend(state["delta_history"])
            _push_event(state, current, parts[0][0], delta)
            _record_history(state, current)
        _save_state(state)
    return state


def apply_inactivity_decay(now: datetime | None = None) -> dict[str, Any]:
    """
    無交流ペナルティと中立への戻りを適用する（毎日 04:05 のスケジューラから呼ぶ）。
    INACTIVITY_GRACE_DAYS 日を超えてインタラクションがなければ、その日の分（0.15）だけ減点する。
    （REV-598: 以前は毎日「超過日数 × 0.15」を引いており、沈黙が続くと累積で急落していた）
    """
    current = now or datetime.utcnow()
    run_date = current.date().isoformat()
    penalty = 0.0
    with _STATE_LOCK:
        state = _load_state(strict=True)
        if state.get("last_inactivity_decay_date") == run_date:
            return state
        _apply_reversion(state, current)
        last = _parse(state.get("last_interaction", ""))
        days_silent = (current - last).total_seconds() / 86400 if last else 0.0
        state["days_silent"] = round(days_silent, 1)
        if days_silent > _INACTIVITY_GRACE_DAYS:
            penalty = _INACTIVITY_PENALTY_PER_DAY * min(1.0, days_silent - _INACTIVITY_GRACE_DAYS)
            old_score = float(state.get("score", _SCORE_INITIAL))
            state["score"] = round(max(_SCORE_MIN, old_score - penalty), 3)
            history: list[float] = state.get("delta_history", [])
            history.append(round(-penalty, 3))
            state["delta_history"] = history[-10:]
            state["trend"] = _calc_trend(state["delta_history"])
            _push_event(state, current, f"{days_silent:.0f}日話しかけられていない", -penalty)
        state["last_inactivity_decay_date"] = run_date
        _record_history(state, current)
        _save_state(state)
    if penalty:
        logger.info(f"[Relationship] 無交流ペナルティ適用: -{penalty:.3f} → score={state['score']}, days_silent={days_silent:.1f}")
    return state


def prior_negative_streak(state: dict[str, Any] | None = None, now: datetime | None = None) -> int:
    """この発言より前の、訂正・不満の連続回数（この発言の記録が済んでいれば1つ戻す）。"""
    state = state if state is not None else _load_state()
    streak = int(state.get("negative_streak", 0))
    last = _parse(state.get("last_interaction") or "")
    if last is not None and ((now or datetime.utcnow()) - last).total_seconds() < 180 and streak:
        events = list(state.get("events") or [])
        if events and any(word in str(events[-1].get("reason")) for word in ("不満", "訂正")):
            return streak - 1
    return streak


def silence_hours(state: dict[str, Any] | None = None, now: datetime | None = None) -> float | None:
    """前回の対話からの時間（時間）。この発言の記録が済んでいれば、その前の対話からの時間を返す（REV-598）。"""
    state = state if state is not None else _load_state()
    current = now or datetime.utcnow()
    last = _parse(state.get("last_interaction") or "")
    if last is None:
        return None
    if (current - last).total_seconds() < 180:  # この発言の記録が先に済んでいる
        prev = _parse(state.get("previous_interaction") or "")
        return None if prev is None else (last - prev).total_seconds() / 3600
    return (current - last).total_seconds() / 3600


# REV-598 恐れの行動変化の「揺れ」: 以前は「5.0 未満かつ下降」だけで、親しく話す相手だと
# 20日以上の沈黙か数十回の訂正が無いと届かず、一度も発動していなかった。直近の最高値からの下がり幅も見る
FEAR_LOW = 4.0
FEAR_FALLING_SCORE = 5.0
FEAR_DROP = 0.6
FEAR_PEAK_DAYS = 7


def recent_peak(state: dict[str, Any], now: datetime | None = None) -> float:
    """直近 FEAR_PEAK_DAYS 日の日ごとの値と今の値のうち最高のもの。"""
    current = now or datetime.utcnow()
    since = (current.date().toordinal() - FEAR_PEAK_DAYS)
    values = [float(state.get("score", _SCORE_INITIAL))]
    for day, value in (state.get("score_history") or {}).items():
        try:
            if datetime.fromisoformat(day).date().toordinal() >= since and value is not None:
                values.append(float(value))
        except ValueError:
            continue
    return max(values)


def fear_flags(score: float, trend: str, peak: float) -> tuple[bool, bool]:
    """(is_low, is_falling)。is_falling は下降中で「5.0 未満」または「直近の最高値から 0.6 以上下がった」。"""
    is_low = score < FEAR_LOW
    is_falling = not is_low and trend == "falling" and (score < FEAR_FALLING_SCORE or peak - score >= FEAR_DROP)
    return is_low, is_falling


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

    peak = recent_peak(state)
    is_low, is_falling = fear_flags(float(score), str(trend), peak)
    return {
        "score": score,
        "trend": trend,
        "is_falling": is_falling,
        "is_low": is_low,
        "drop_from_peak": round(peak - float(score), 2),
        "days_since_last": round(days_since, 1),
        "negative_streak": state.get("negative_streak", 0),
        "total_interactions": state.get("total_interactions", 0),
        "understanding": state.get("understanding", 0.5),
    }
