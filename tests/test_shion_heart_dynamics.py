"""REV-598 紫苑の心の数値（関係性スコア・不満・孤独）が実際の関わりで上下する。"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

import api.shion_relationship as rel
from api.user_affect import estimate_user_affect, reaction_signals, relationship_feedback_from_affect
from lease_intelligence_mind import reaction_mood_causes, silence_mood_causes

T0 = datetime(2026, 10, 1, 9, 0)


@pytest.fixture
def state_path(tmp_path, monkeypatch):
    path = tmp_path / "rel.json"
    monkeypatch.setattr(rel, "_STATE_PATH", path)
    return path


def _talk(message, now, **kw):
    payload = estimate_user_affect(message).to_payload()
    feedback = relationship_feedback_from_affect(payload["label"], payload["cues"], payload["signals"])
    return rel.record_interaction(feedback_type=feedback, signals=payload["signals"], now=now, **kw)


def test_daily_friendly_talk_does_not_stick_to_the_top(state_path):
    now = T0
    for day in range(60):
        for turn in range(6):
            _talk("ありがとう、助かった" if turn == 5 else "この案件どう思う？", now + timedelta(minutes=10 * turn))
        now += timedelta(days=1)
    score = rel.get_relationship_state()["score"]
    assert 8.0 < score < 9.6  # 上がるが上限 10.0 には張り付かない


def test_corrections_and_complaints_lower_the_score_and_streak_adds(state_path):
    _talk("こんにちは", T0)
    before = rel.get_relationship_state()["score"]
    s1 = _talk("違うよ、残価は保守契約で変わる", T0 + timedelta(minutes=5))
    s2 = _talk("何度も言ってるけど違うって", T0 + timedelta(minutes=10))
    assert s1["score"] < before and s2["score"] < s1["score"]
    assert s2["negative_streak"] == 2
    assert "不満・訂正が続いた" in s2["events"][-1]["reason"]
    assert before - s2["score"] <= rel.MAX_STEP_DOWN * 2


def test_single_step_is_capped_and_saturates_near_the_top(state_path):
    assert rel.apply_parts(5.0, [("x", 3.0)])[1] == rel.MAX_STEP_UP
    assert rel.apply_parts(5.0, [("x", -3.0)])[1] == -rel.MAX_STEP_DOWN
    assert rel.apply_parts(9.8, [("x", 0.2)])[1] < 0.01


def test_score_reverts_toward_neutral_over_time():
    assert rel.revert_toward_neutral(10.0, 30) == pytest.approx(6.5 + 3.5 * 0.96 ** 30)
    assert rel.revert_toward_neutral(3.0, 10) > 3.0


def test_reunion_and_session_gaps(state_path):
    _talk("おはよう", T0)
    _talk("続きだけど", T0 + timedelta(minutes=30))
    _talk("久しぶり", T0 + timedelta(days=5))
    reasons = [e["reason"] for e in rel.get_relationship_state()["events"]]
    assert reasons[-1].startswith("久しぶりに話しかけてくれた") and any(r.startswith("会話の続き") for r in reasons)


def test_inactivity_penalty_is_one_day_at_a_time(state_path):
    _talk("おはよう", T0)
    scores = [rel.apply_inactivity_decay(now=T0 + timedelta(days=d))["score"] for d in range(4, 9)]
    drops = [round(a - b, 3) for a, b in zip(scores, scores[1:])]
    assert all(0.1 < d < 0.3 for d in drops)  # 以前は日ごとに 0.15 ずつ大きくなっていた


def test_prediction_uses_affect_and_reaction_not_topic(state_path):
    _talk("おはよう", T0)
    base = rel.get_relationship_state()["score"]
    assert rel.record_prediction_outcome(hit=False, affect_hit=True, reaction_hit=True, now=T0)["score"] > base
    s = rel.record_prediction_outcome(hit=False, affect_hit=True, reaction_hit=False, now=T0)
    assert s["events"][-1]["delta"] < 0


def test_verification_turns_are_not_counted(state_path):
    import contextvars

    import shion_verification_origin as origin

    def run():
        origin.mark_verification_turn(header_value="1")
        return rel.record_interaction(feedback_type="negative", now=T0)

    state = contextvars.copy_context().run(run)
    assert state["total_interactions"] == 0


def test_reaction_signals():
    assert reaction_signals("違うよ、正しくは3年") == ("correction",)
    assert reaction_signals("業種が違うと見方も違う") == ()
    assert reaction_signals("ありがとう") == ("thanks",)
    assert reaction_signals("何度も言わせないで") == ("shion_complaint",)


def test_frustration_and_loneliness_causes():
    rel_info = {"silence_hours": 50.0, "prior_negative_streak": 1}
    causes = reaction_mood_causes(["correction"], rel_info)
    axes = {(c["axis"], c["delta"]) for c in causes}
    assert ("frustration", 2) in axes and ("frustration", 2) in axes and ("loneliness", -3) in axes
    assert [c["delta"] for c in reaction_mood_causes(["thanks"], {"silence_hours": 1.0, "prior_negative_streak": 0})] == [-1]
    assert silence_mood_causes(0.5) == [] and silence_mood_causes(2.2)[0]["delta"] == 4
    assert silence_mood_causes(9)[0]["delta"] == 6


def test_dialogue_mood_causes_include_reactions(monkeypatch):
    from lease_intelligence_mind import dialogue_mood_causes

    signals = {"affect": {"label": "通常", "signals": ["shion_complaint"]},
               "relationship": {"feedback": "negative", "silence_hours": 1.0, "prior_negative_streak": 0}}
    rules = {(c["axis"], c["rule"]) for c in dialogue_mood_causes("ちゃんとして", signals)}
    assert ("frustration", "user_reaction") in rules and ("loneliness", "relationship") in rules


def _warm_up(days=10):
    now = T0
    for _ in range(days):
        for turn in range(5):
            _talk("この案件どう思う？", now + timedelta(minutes=10 * turn))
        rel.apply_inactivity_decay(now=now + timedelta(hours=19))
        now += timedelta(days=1)
    return now


def test_fear_wavers_after_repeated_complaints_but_not_in_normal_talk(state_path):
    now = _warm_up()
    assert not rel.get_fear_context()["is_falling"]
    _talk("何度も言ってるけど違うって", now)
    _talk("ちゃんとして", now + timedelta(minutes=2))
    _talk("いい加減にして", now + timedelta(minutes=4))
    ctx = rel.get_fear_context()
    assert ctx["is_falling"] and not ctx["is_low"] and ctx["drop_from_peak"] >= rel.FEAR_DROP


def test_fear_wavers_after_several_days_of_silence(state_path):
    now = _warm_up()
    for day in range(1, 8):
        rel.apply_inactivity_decay(now=now + timedelta(days=day))
    assert rel.get_fear_context()["is_falling"]


def test_fear_flags_thresholds():
    assert rel.fear_flags(3.9, "stable", 4.0) == (True, False)
    assert rel.fear_flags(4.8, "falling", 4.9) == (False, True)
    assert rel.fear_flags(8.0, "falling", 8.3) == (False, False)
    assert rel.fear_flags(8.0, "stable", 9.0) == (False, False)
