"""REV-599 話の内容で紫苑の気分・関係性を動かす（分類は偽物で、Gemini は呼ばない）。"""
from __future__ import annotations

import contextvars
import json

import pytest

import api.shion_content_mood as cm
import api.shion_relationship as rel


def _fake(category, intensity=0.9, reason="理由"):
    calls = []

    def caller(prompt):
        calls.append(prompt)
        return json.dumps({"category": category, "intensity": intensity, "reason": reason}, ensure_ascii=False)

    caller.calls = calls
    return caller


@pytest.mark.parametrize("category,axes", [
    ("self_disclosure", {"attachment"}),
    ("shared_joy", {"hope", "accomplishment"}),
    ("hardship", {"attachment", "vigilance"}),
    ("disagreement", {"frustration", "vigilance"}),
    ("deep_talk", {"curiosity", "attachment"}),
    ("business", set()),
    ("small_talk", set()),
])
def test_categories_move_expected_axes_and_record_what_was_talked_about(category, axes):
    result = cm.classify_content("最近ちょっと仕事のことで迷っていて", caller=_fake(category, reason="仕事の迷い"))
    causes = cm.content_mood_causes(result)
    assert {c["axis"] for c in causes} == axes
    assert all(c["rule"] == "content" and "話の内容" in c["detail"] and "仕事の迷い" in c["detail"] for c in causes)
    assert all(abs(c["delta"]) <= 2 for c in causes)


def test_weak_or_unclear_content_does_not_move_and_medium_is_halved():
    assert cm.content_mood_causes(cm.classify_content("なんとなく話したくて", caller=_fake("self_disclosure", 0.3))) == []
    half = cm.content_mood_causes(cm.classify_content("なんとなく話したくて", caller=_fake("shared_joy", 0.5)))
    assert [c["delta"] for c in half] == [1, 1]


def test_short_messages_do_not_call_the_model():
    caller = _fake("deep_talk")
    assert cm.classify_content("おはよう", caller=caller)["category"] == "small_talk" and caller.calls == []


def test_bad_output_is_ignored():
    assert cm.classify_content("今日の審査の件だけど", caller=lambda p: "わかりません") is None
    assert cm.classify_content("今日の審査の件だけど", caller=lambda p: '{"category": "other"}') is None


def test_overlap_with_user_affect_is_not_counted_twice():
    causes = [{"axis": "hope", "delta": 2, "rule": "content", "detail": "x"},
              {"axis": "accomplishment", "delta": 2, "rule": "content", "detail": "x"}]
    affect = [{"axis": "hope", "delta": 2, "rule": "user_affect"}, {"axis": "accomplishment", "delta": 1, "rule": "user_affect"}]
    kept = cm.without_affect_overlap(causes, affect)
    assert kept == [{"axis": "accomplishment", "delta": 1, "rule": "content", "detail": "x"}]


def test_apply_writes_mood_log_and_relationship(tmp_path, monkeypatch):
    monkeypatch.setenv("SHION_CONTENT_MOOD", "force")
    monkeypatch.setattr(cm, "_log_path", lambda: tmp_path / "content_log.jsonl")
    monkeypatch.setattr(rel, "_STATE_PATH", tmp_path / "rel.json")
    applied = {}

    def fake_apply(vault, causes, *, event, trigger=""):
        applied.update(event=event, causes=causes, trigger=trigger)

    monkeypatch.setattr("lease_intelligence_mind.apply_mood_causes", fake_apply)
    before = rel.get_relationship_state()["score"]
    out = cm.apply_content_effects(tmp_path, "実は最近、自信をなくしていて相談したかった", "話してくれてありがとうございます",
                                   caller=_fake("self_disclosure", reason="自信をなくしている"))
    assert applied["event"] == "dialogue_content" and applied["causes"][0]["axis"] == "attachment"
    state = rel.get_relationship_state()
    assert state["score"] > before and "打ち明けて" in state["events"][-1]["reason"]
    assert out["category"] == "self_disclosure"


def test_verification_turns_are_not_used(tmp_path, monkeypatch):
    import shion_verification_origin as origin

    monkeypatch.setenv("SHION_CONTENT_MOOD", "force")
    caller = _fake("self_disclosure")

    def run():
        origin.mark_verification_turn(header_value="1")
        return cm.apply_content_effects(tmp_path, "実は最近、自信をなくしていて", "", caller=caller)

    assert contextvars.copy_context().run(run) is None and caller.calls == []


def test_off_in_tests_unless_forced(monkeypatch):
    monkeypatch.delenv("SHION_CONTENT_MOOD", raising=False)
    assert not cm.enabled()


def test_budget_class_is_memory():
    import ai_budget

    assert ai_budget.call_class("shion_content_mood") == ai_budget.MEMORY


def test_grounding_labels_new_rules():
    from api.shion_emotion_grounding import RULE_LABELS

    assert RULE_LABELS["content"] == "話の内容" and "user_reaction" in RULE_LABELS and "silence" in RULE_LABELS


def test_batch_classification_for_replay():
    def caller(prompt):
        assert "[1]" in prompt and "self_disclosure" in prompt
        return json.dumps({"items": [{"i": 0, "category": "business", "intensity": 0.8, "reason": "審査"},
                                     {"i": 1, "category": "shared_joy", "intensity": 0.9, "reason": "成約"},
                                     {"i": 9, "category": "deep_talk"}]})

    out = cm.classify_batch([("審査の件", "はい"), ("成約しました！", "おめでとうございます")], caller=caller)
    assert [o["category"] for o in out] == ["business", "shared_joy"]


def test_classification_is_logged_even_when_nothing_moves(tmp_path, monkeypatch):
    """REV-601 雑談・事務的で動かさなかった時も、種類・強さ・要約を残す（発言の本文は残さない）。"""
    monkeypatch.setenv("SHION_CONTENT_MOOD", "force")
    log = tmp_path / "content_log.jsonl"
    monkeypatch.setattr(cm, "_log_path", lambda: log)
    cm.apply_content_effects(tmp_path, "釣竿を買おうか迷っている", "", caller=_fake("small_talk", 0.8, "釣竿の購入"))
    row = json.loads(log.read_text(encoding="utf-8").strip())
    assert row["category"] == "small_talk" and row["moved"] is False and row["reason"] == "釣竿の購入"
    assert "釣竿を買おうか" not in log.read_text(encoding="utf-8")


def test_failed_classification_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("SHION_CONTENT_MOOD", "force")
    recorded = []
    monkeypatch.setattr("silent_failure_log.record_silent_failure", lambda *a, **k: recorded.append(a[0]))
    assert cm.apply_content_effects(tmp_path, "今日の審査の件だけど", "", caller=lambda p: "わからない") is None
    assert recorded == ["answer.content_mood"]
