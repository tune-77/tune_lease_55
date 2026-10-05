"""紫苑が相手を予想し、答え合わせで自分の相手モデルを直す（REV-472）。

「お互いを予想して修正し合う」ループを、追加の LLM 呼び出しなしで回す。

1. 予想を残す: 毎ターン、今の様子（REV-464）と最近の様子の記憶（REV-465）から、
   相手の「次の気持ち」「紫苑の返答への反応」「次の話題（業務か雑談か）」を予想して保存する。
2. 答え合わせ: 次の発言が来たら前回の予想と突き合わせ、当たり・外れと驚き（予測誤差）を出す。
   - 外れは次の返答の指示（前の様子に引っ張られない）と、相手ごとのモデル（気持ちを引きずるか）に反映
   - 外れの記録は data/shion_prediction_log.jsonl に残し、夜の Private Reflection の材料にする
   - 当たり・外れは関係性スコア（REV-220/467）の「理解度」に反映する
3. 好奇心: 外れた予想や、理由の分からない様子を「気になること」として持ち、
   条件が揃った時だけ紫苑から1つだけ聞く（頻度制限・業務の審査中は聞かない・フラグで停止可）。

保存するのはラベル・時刻・質問の定型文だけで、発言本文は残さない。
推定は語調の調整と問いかけにだけ使い、事実・審査判断には使わない。
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from runtime_paths import get_data_path

NEUTRAL = "通常"
NEW_SESSION_GAP = timedelta(hours=2)
# 予想が古すぎたら答え合わせしない（相手の状況が変わりすぎている）
PREDICTION_TTL = timedelta(days=4)
MAX_HISTORY = 60
MAX_CURIOSITIES = 5
CURIOSITY_TTL = timedelta(days=7)
DEFAULT_COOLDOWN_HOURS = 20.0
# 予想の重み（驚きの大きさ）
_AFFECT_WEIGHT = 0.6
_REACTION_WEIGHT = 0.8
_TOPIC_WEIGHT = 0.2
# この回数以上答え合わせしてから、相手モデル（気持ちを引きずるか）を予想に使う
_MIN_PERSIST_SAMPLES = 4

_NEGATIVE = {"疲れ", "焦り", "不安", "落ち込み", "苛立ち"}
# 業務の審査・深掘り中は聞かない
_BUSINESS_MODES = {"screening", "deep", "long"}
# 今の様子がこれなら聞かない（急いでいる・苛立っている・落ち込んでいる時に質問で負担をかけない）
_NO_QUESTION_AFFECTS = {"焦り", "苛立ち", "落ち込み"}
# 理由を自分から話していれば「何があったの？」と聞く必要は薄い
_CAUSE_MARKERS = ("から", "ので", "せいで", "ため", "のは")
# 雑談以外でこれを含む発言は「答えを求めている」とみなし、問いかけを添えない
_ASKING_MARKERS = ("？", "?", "教えて", "調べて", "確認して", "お願い", "どう思う", "どうすれば")

# REV-468 の短く砕けた口調に合わせた問いかけ（「この前」は別の会話で見た様子について聞くとき）
_OBSERVED_QUESTIONS: dict[str, str] = {
    "疲れ": "この前疲れてたけど、何があったの？",
    "焦り": "この前バタバタしてたけど、間に合った？",
    "不安": "この前気にしてたこと、その後どうなった？",
    "落ち込み": "この前ちょっと元気なかったけど、その後どう？",
    "苛立ち": "この前イライラしてたけど、何かあったの？",
    "喜び": "この前嬉しそうだったけど、何があったの？",
}
# 予想が外れたとき（いま目の前で起きた驚きについて聞く）
_SURPRISE_QUESTIONS: dict[tuple[str, str], str] = {
    ("negative", "喜び"): "あれ、今日は元気そう。何かいいことあった？",
    ("negative", NEUTRAL): "思ったより元気そうでよかった。何か変わった？",
    (NEUTRAL, "疲れ"): "あれ、今日はちょっと疲れてる？何かあった？",
    (NEUTRAL, "不安"): "何か気になってることある？",
}

_LOCK = threading.Lock()


def mutual_prediction_enabled() -> bool:
    return os.environ.get("SHION_MUTUAL_PREDICTION_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}


def curiosity_enabled() -> bool:
    return os.environ.get("SHION_CURIOSITY_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}


def _cooldown() -> timedelta:
    try:
        hours = float(os.environ.get("SHION_CURIOSITY_COOLDOWN_HOURS", DEFAULT_COOLDOWN_HOURS))
    except ValueError:
        hours = DEFAULT_COOLDOWN_HOURS
    return timedelta(hours=max(1.0, hours))


def _state_path() -> Path:
    return Path(get_data_path("shion_mutual_prediction_state.json"))


def _log_path() -> Path:
    return Path(get_data_path("shion_prediction_log.jsonl"))


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"users": {}}
    if not isinstance(data, dict) or not isinstance(data.get("users"), dict):
        return {"users": {}}
    return data


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _parse(ts: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None


def _iso(now: datetime) -> str:
    return now.isoformat(timespec="seconds")


def topic_class(context_mode: str) -> str:
    """会話モードを、予想しやすい粗い話題の種類にする。"""
    if context_mode in _BUSINESS_MODES:
        return "業務"
    if context_mode == "casual":
        return "雑談"
    return "相談"


def _new_user() -> dict[str, Any]:
    return {
        "pending": None,
        "history": [],
        "model": {"hits": 0, "misses": 0, "persist_hits": 0, "persist_misses": 0, "surprise_ema": 0.0},
        "curiosities": [],
        "last_question_at": "",
    }


# ── 1. 予想 ─────────────────────────────────────────────────


def _predict(
    user: dict[str, Any], affect_label: str, recurrent: str, context_mode: str, now: datetime
) -> dict[str, Any]:
    """次の発言についての予想。同じ会話の続きか、時間を置いた次の会話かで分けて持つ。"""
    model = user.get("model") or {}
    samples = int(model.get("persist_hits", 0)) + int(model.get("persist_misses", 0))
    persist_rate = (int(model.get("persist_hits", 0)) / samples) if samples else 0.5
    # 相手モデル: 気持ちを引きずりにくい相手だと分かったら、同じ会話内でも持続を予想しない
    quick_to_shift = samples >= _MIN_PERSIST_SAMPLES and persist_rate < 0.4
    same_session = NEUTRAL if quick_to_shift else affect_label
    # 時間を置けば落ち着く、ただし繰り返し出ている様子はまた出ると予想する
    next_session = recurrent if recurrent and recurrent == affect_label else NEUTRAL
    return {
        "made_at": _iso(now),
        "from_affect": affect_label,
        "affect_same_session": same_session,
        "affect_next_session": next_session,
        "reaction": "受け入れる",  # 紫苑の返答への不満は出ないはず
        "topic_same_session": topic_class(context_mode),
        "quick_to_shift": quick_to_shift,
    }


# ── 2. 答え合わせ ─────────────────────────────────────────────


@dataclass(frozen=True)
class PredictionOutcome:
    expected_affect: str
    actual_affect: str
    affect_hit: bool
    reaction_hit: bool
    topic_hit: bool | None  # 次の会話では話題の予想はしない
    new_session: bool
    surprise: float
    trivial: bool  # 「通常→通常」のような当たり前の当たり

    @property
    def missed(self) -> bool:
        return not (self.affect_hit and self.reaction_hit and self.topic_hit is not False)

    def to_payload(self) -> dict[str, Any]:
        return {
            "expected_affect": self.expected_affect,
            "actual_affect": self.actual_affect,
            "affect_hit": self.affect_hit,
            "reaction_hit": self.reaction_hit,
            "topic_hit": self.topic_hit,
            "new_session": self.new_session,
            "surprise": round(self.surprise, 2),
        }


def _evaluate(
    pending: dict[str, Any], affect_label: str, shion_directed_complaint: bool, context_mode: str, now: datetime
) -> PredictionOutcome | None:
    made_at = _parse(pending.get("made_at"))
    if made_at is None or now - made_at > PREDICTION_TTL or now < made_at:
        return None
    new_session = now - made_at >= NEW_SESSION_GAP
    expected = str(pending.get("affect_next_session" if new_session else "affect_same_session") or NEUTRAL)
    affect_hit = expected == affect_label
    reaction_hit = not shion_directed_complaint
    topic_hit = None if new_session else pending.get("topic_same_session") == topic_class(context_mode)
    surprise = 0.0
    if not affect_hit:
        # 気になる様子と通常の取り違えより、正反対（落ち込み⇔喜び）の方が大きな驚き
        opposite = (expected in _NEGATIVE and affect_label == "喜び") or (expected == "喜び" and affect_label in _NEGATIVE)
        surprise += _AFFECT_WEIGHT * (1.0 if opposite else 0.7)
    if not reaction_hit:
        surprise += _REACTION_WEIGHT
    if topic_hit is False:
        surprise += _TOPIC_WEIGHT
    trivial = (
        affect_hit and reaction_hit and topic_hit is not False
        and expected == NEUTRAL and pending.get("from_affect") == NEUTRAL
    )
    return PredictionOutcome(
        expected_affect=expected,
        actual_affect=affect_label,
        affect_hit=affect_hit,
        reaction_hit=reaction_hit,
        topic_hit=topic_hit,
        new_session=new_session,
        surprise=min(1.0, surprise),
        trivial=trivial,
    )


def _update_model(user: dict[str, Any], pending: dict[str, Any], outcome: PredictionOutcome) -> None:
    model = user.setdefault("model", _new_user()["model"])
    if outcome.missed:
        model["misses"] = int(model.get("misses", 0)) + 1
    elif not outcome.trivial:
        model["hits"] = int(model.get("hits", 0)) + 1
    # 「同じ会話の中で気持ちが続くか」の当たり外れで、相手の切り替えの早さを学ぶ
    if not outcome.new_session and pending.get("from_affect") not in (None, NEUTRAL):
        if outcome.actual_affect == pending.get("from_affect"):
            model["persist_hits"] = int(model.get("persist_hits", 0)) + 1
        else:
            model["persist_misses"] = int(model.get("persist_misses", 0)) + 1
    ema = float(model.get("surprise_ema", 0.0))
    model["surprise_ema"] = round(0.8 * ema + 0.2 * outcome.surprise, 3)


def _append_log(entry: dict[str, Any], path: Path | None) -> None:
    path = path or _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── 3. 好奇心 ─────────────────────────────────────────────────


def _has_cause(message: str) -> bool:
    text = str(message or "")
    return len(text) >= 20 and any(m in text for m in _CAUSE_MARKERS)


def _add_curiosity(
    user: dict[str, Any], *, kind: str, label: str, question: str, now: datetime, immediate: bool = False
) -> None:
    items = [c for c in user.get("curiosities") or [] if not c.get("resolved_at")]
    # 同じ様子についての未解決の気になることがあれば、新しい方に置き換える
    items = [c for c in items if not (c.get("label") == label and c.get("kind") == kind)]
    items.append({
        "id": uuid.uuid4().hex[:10],
        "kind": kind,
        "label": label,
        "question": question,
        "created_at": _iso(now),
        "immediate": immediate,
        "asked_at": "",
    })
    user["curiosities"] = items[-MAX_CURIOSITIES:]


def _prune_curiosities(user: dict[str, Any], now: datetime) -> None:
    kept = []
    for c in user.get("curiosities") or []:
        created = _parse(c.get("created_at"))
        if c.get("resolved_at") or created is None or now - created > CURIOSITY_TTL:
            continue
        kept.append(c)
    user["curiosities"] = kept[-MAX_CURIOSITIES:]


def _is_asking_something(message: str) -> bool:
    """相手が何かを尋ねている（業務の質問の可能性がある）なら、こちらから質問を重ねない。"""
    text = str(message or "")
    return any(m in text for m in _ASKING_MARKERS)


def _pick_question(
    user: dict[str, Any],
    *,
    message: str,
    affect_label: str,
    context_mode: str,
    allow_questions: bool,
    now: datetime,
) -> dict[str, Any] | None:
    if not (allow_questions and curiosity_enabled()):
        return None
    if context_mode in _BUSINESS_MODES or affect_label in _NO_QUESTION_AFFECTS:
        return None
    if context_mode != "casual" and _is_asking_something(message):
        return None
    last = _parse(user.get("last_question_at"))
    if last is not None and now - last < _cooldown():
        return None
    candidates = []
    for c in user.get("curiosities") or []:
        if c.get("asked_at") or c.get("resolved_at"):
            continue
        created = _parse(c.get("created_at"))
        if created is None:
            continue
        # 前の会話で見た様子は、時間を置いた次の会話で聞く。驚きはその場で聞いてよい
        if not c.get("immediate") and now - created < NEW_SESSION_GAP:
            continue
        candidates.append(c)
    if not candidates:
        return None
    # その場の驚き → 新しいものの順
    candidates.sort(key=lambda c: (bool(c.get("immediate")), str(c.get("created_at"))), reverse=True)
    return candidates[0]


# ── 1ターン分の処理 ───────────────────────────────────────────


@dataclass
class MutualPredictionTurn:
    user_id: str
    outcome: PredictionOutcome | None = None
    question: dict[str, Any] | None = None
    answered_question: str = ""
    prompt_block: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


def _build_block(turn: MutualPredictionTurn, *, quick_to_shift: bool) -> str:
    lines: list[str] = []
    o = turn.outcome
    if o is not None and o.missed:
        if not o.affect_hit:
            lines.append(
                f"- 前回の私の予想: 相手は「{o.expected_affect}」だと思っていた → 今の発言は「{o.actual_affect}」に見える。"
                "予想は外れた。前の様子や記憶に引っ張られず、今の相手に合わせ直す"
                "（上の「最近の様子」の気遣いより、今の様子を優先する）。"
            )
        if not o.reaction_hit:
            lines.append(
                "- 前回の私の返答は、相手の期待とずれていたらしい。言い訳せず、どこがずれたかを一言で認めて、"
                "求められている答えを先に出す。"
            )
        if o.topic_hit is False:
            lines.append("- 話題が変わった。前の話題を引きずらず、新しい話題に合わせる。")
    if quick_to_shift:
        lines.append("- この人は気持ちの切り替えが早い。前の様子を引きずって気遣いすぎない。")
    if turn.answered_question:
        lines.append(
            f"- 前回私が「{turn.answered_question}」と聞いた。今の発言はその答えかもしれない。"
            "答えなら、まず短く受け止める（根掘り葉掘り聞き返さない）。話したくなさそうなら流す。"
        )
    if turn.question:
        lines.append(
            "- 返答の最後に、次の問いかけを1回だけ自然に添えてよい（言い回しは崩してよい・短く）: "
            f"「{turn.question['question']}」。本題への答えを先に書き、問いかけはおまけ程度にする。"
            "ほかの質問は重ねない。"
        )
    if not lines:
        return ""
    return (
        "【相手についての私の予想と答え合わせ】\n"
        + "\n".join(lines)
        + "\n- 予想していたこと・記録していることは相手に言わない。事実・審査判断には使わない。"
    )


def begin_turn(
    user_id: str,
    *,
    message: str,
    affect_payload: dict[str, Any],
    context_mode: str,
    surface: str,
    recurrent: str = "",
    allow_questions: bool = True,
    now: datetime | None = None,
    path: Path | None = None,
    log_path: Path | None = None,
) -> MutualPredictionTurn:
    """前回の予想を答え合わせし、気になることを更新し、今回の予想を残す。返答前に呼ぶ。"""
    from api.user_affect import relationship_feedback_from_affect

    uid = str(user_id or "default")[:100]
    now = now or datetime.now()
    path = path or _state_path()
    affect_label = str(affect_payload.get("label") or NEUTRAL)
    cues = list(affect_payload.get("cues") or [])
    complaint = relationship_feedback_from_affect(affect_label, cues) == "negative"
    turn = MutualPredictionTurn(user_id=uid)

    with _LOCK:
        data = _load(path)
        user = data["users"].setdefault(uid, _new_user())
        _prune_curiosities(user, now)

        surprise_label = ""
        pending = user.get("pending")
        if isinstance(pending, dict):
            turn.outcome = _evaluate(pending, affect_label, complaint, context_mode, now)
            if turn.outcome is not None:
                _update_model(user, pending, turn.outcome)
                if not turn.outcome.trivial:
                    entry = {"at": _iso(now), "user_id": uid, "surface": str(surface)[:40], **turn.outcome.to_payload()}
                    user["history"] = (list(user.get("history") or []) + [entry])[-MAX_HISTORY:]
                    try:
                        _append_log(entry, log_path)
                    except OSError:
                        pass
                # 外れた驚きは「気になること」になる
                o = turn.outcome
                if not o.affect_hit:
                    key = ("negative" if o.expected_affect in _NEGATIVE else o.expected_affect, o.actual_affect)
                    if key in _SURPRISE_QUESTIONS:
                        _add_curiosity(
                            user, kind="surprise", label=o.actual_affect,
                            question=_SURPRISE_QUESTIONS[key], now=now, immediate=True,
                        )
                        surprise_label = o.actual_affect

        # 前回聞いた問いには、今の発言が答え（または無視）。どちらでも二度は聞かない
        for c in user.get("curiosities") or []:
            if c.get("asked_at") and not c.get("resolved_at"):
                c["resolved_at"] = _iso(now)
                c["resolution"] = "answered" if topic_class(context_mode) != "業務" and len(message.strip()) >= 4 else "skipped"
                turn.answered_question = str(c.get("question") or "") if c["resolution"] == "answered" else ""

        # 理由の分からない様子は、次の会話で聞きたいことになる
        # （同じ様子をいま驚きとして聞くなら、次の会話で重ねて聞かない）
        if (
            affect_label in _OBSERVED_QUESTIONS
            and affect_label != surprise_label
            and not complaint
            and not _has_cause(message)
        ):
            _add_curiosity(user, kind="observed", label=affect_label, question=_OBSERVED_QUESTIONS[affect_label], now=now)

        turn.question = _pick_question(
            user,
            message=message,
            affect_label=affect_label,
            context_mode=context_mode,
            allow_questions=allow_questions,
            now=now,
        )
        if turn.question:
            # 返答を見て実際に聞いたか確かめるまでは「聞くつもり」
            turn.question["offered_at"] = _iso(now)

        user["pending"] = _predict(user, affect_label, recurrent, context_mode, now)
        quick = bool(user["pending"].get("quick_to_shift"))
        _save(path, data)

    turn.prompt_block = _build_block(turn, quick_to_shift=quick)
    turn.payload = {
        "used": bool(turn.prompt_block),
        "outcome": turn.outcome.to_payload() if turn.outcome else None,
        "question_offered": bool(turn.question),
        "answered_question": bool(turn.answered_question),
        "quick_to_shift": quick,
    }
    return turn


def finish_turn(
    turn: MutualPredictionTurn, reply: str, *, now: datetime | None = None, path: Path | None = None
) -> bool:
    """返答に問いかけが入っていれば「聞いた」と記録する。入っていなければ次の機会に回す。"""
    if not turn.question:
        return False
    # 問いかけは返答の最後に添えるよう指示している。本文中の別の「？」では数えない
    tail = str(reply or "").strip()[-120:]
    asked = "？" in tail or "?" in tail
    now = now or datetime.now()
    path = path or _state_path()
    with _LOCK:
        data = _load(path)
        user = data["users"].get(turn.user_id)
        if not user:
            return False
        for c in user.get("curiosities") or []:
            if c.get("id") == turn.question.get("id"):
                c.pop("offered_at", None)
                if asked:
                    c["asked_at"] = _iso(now)
                    user["last_question_at"] = _iso(now)
        _save(path, data)
    turn.payload["question_asked"] = asked
    return asked


def record_relationship_from_outcome(outcome: PredictionOutcome | None) -> None:
    """答え合わせの結果を関係性スコアの理解度へ。失敗しても会話は止めない。"""
    if outcome is None or outcome.trivial:
        return
    try:
        from api.shion_relationship import record_prediction_outcome

        record_prediction_outcome(hit=not outcome.missed, surprise=outcome.surprise)
    except Exception as exc:
        from silent_failure_log import record_silent_failure

        record_silent_failure("answer.mutual_prediction_relationship", "swallowed", exc)


# ── 参照（確認用 API・Private Reflection） ───────────────────────


def get_user_summary(user_id: str, *, path: Path | None = None) -> dict[str, Any]:
    data = _load(path or _state_path())
    user = data["users"].get(str(user_id or "default")[:100]) or _new_user()
    model = user.get("model") or {}
    judged = int(model.get("hits", 0)) + int(model.get("misses", 0))
    return {
        "enabled": mutual_prediction_enabled(),
        "curiosity_enabled": curiosity_enabled(),
        "pending_prediction": user.get("pending"),
        "model": model,
        "hit_rate": round(int(model.get("hits", 0)) / judged, 2) if judged else None,
        "recent_outcomes": list(user.get("history") or [])[-10:],
        "curiosities": [c for c in user.get("curiosities") or [] if not c.get("resolved_at")],
        "last_question_at": user.get("last_question_at") or "",
    }


def load_prediction_errors_for_dates(dates: list[str], *, path: Path | None = None, limit: int = 12) -> list[dict[str, Any]]:
    """指定日（YYYY-MM-DD）の外れた予想を返す。Private Reflection の材料。"""
    path = path or _log_path()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or str(entry.get("at", ""))[:10] not in dates:
            continue
        if entry.get("affect_hit") and entry.get("reaction_hit") and entry.get("topic_hit") is not False:
            continue
        out.append(entry)
    return out[-limit:]


def build_reflection_material(dates: list[str], *, path: Path | None = None) -> str:
    """外れた予想を、内省の材料になる短い箇条書きにする。無ければ空文字。"""
    errors = load_prediction_errors_for_dates(dates, path=path)
    if not errors:
        return ""
    lines = []
    for e in errors:
        parts = []
        if not e.get("affect_hit"):
            parts.append(f"「{e.get('expected_affect')}」だと予想 → 実際は「{e.get('actual_affect')}」")
        if not e.get("reaction_hit"):
            parts.append("私の返答に不満が返ってきた")
        if e.get("topic_hit") is False:
            parts.append("話題の予想が外れた")
        when = str(e.get("at", ""))[5:16].replace("T", " ")
        lines.append(f"- {when} " + "／".join(parts))
    return (
        "【相手について外れた私の予想（予測誤差）】\n"
        + "\n".join(lines)
        + "\n（外れから相手について何を学んだか、自分の思い込みはどこにあったかも内省に含めること）"
    )
