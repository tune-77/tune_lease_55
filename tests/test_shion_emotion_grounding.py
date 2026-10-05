"""REV-481: 気分の変化記録・感情の自己報告の接地・照合ログ。"""

import json

import lease_intelligence_mind as mind
from api import shion_emotion_grounding as grounding
from api.shion_experience_loop import update_experience_state
from lease_intelligence_mind import (
    DIALOGUE_MOOD_CAP,
    MOOD_STEP_LIMIT,
    apply_mood_causes,
    load_lease_intelligence_mind,
    register_dialogue_event,
)


def _rules(entry):
    return {cause["rule"] for cause in entry["causes"]}


def test_dialogue_event_records_changes_with_rules_and_signals(tmp_path):
    state = register_dialogue_event(
        tmp_path,
        "このリスクはなぜ見落としたの？",
        "返答",
        signals={
            "affect": {"label": "不安", "intensity": 0.6},
            "prediction": {"expected_affect": "通常", "actual_affect": "不安", "affect_hit": False, "reaction_hit": True},
            "relationship": {"score": 6.5, "trend": "falling", "last_delta": -0.2},
        },
    )
    entry = state["mood_change_log"][-1]
    assert entry["event"] == "dialogue"
    assert entry["trigger"].startswith("このリスク")
    assert {"dialogue_visit", "content_keyword", "user_affect", "prediction_error", "relationship"} <= _rules(entry)
    vigilance = next(change for change in entry["changes"] if change["axis"] == "vigilance")
    assert vigilance["after"] > vigilance["before"]
    assert any(cause["rule"] == "user_affect" for cause in vigilance["causes"])


def test_reply_text_does_not_move_mood(tmp_path):
    state = register_dialogue_event(tmp_path, "了解", "リスク 否決 危険 失敗")
    assert "content_keyword" not in _rules(state["mood_change_log"][-1])


def test_each_update_moves_each_axis_at_most_step_limit(tmp_path):
    before = load_lease_intelligence_mind(tmp_path)["mood"]
    state = register_dialogue_event(
        tmp_path,
        "リスク 否決 危険 失敗 孤独 寂しい",
        signals={"affect": {"label": "焦り", "intensity": 0.9}},
    )
    for axis, value in state["mood"].items():
        assert abs(value - before[axis]) <= MOOD_STEP_LIMIT


def test_repeated_dialogue_does_not_pin_to_cap(tmp_path):
    for _ in range(60):
        state = register_dialogue_event(tmp_path, "なぜ？")
    assert 0 < state["dialogue_mood"]["curiosity"] < DIALOGUE_MOOD_CAP
    assert 0 < state["dialogue_mood"]["attachment"] < DIALOGUE_MOOD_CAP
    assert len(state["mood_change_log"]) == mind.MOOD_CHANGE_LOG_LIMIT


def test_neutral_memories_no_longer_drain_vigilance_to_floor():
    memories = [{"summary": "雑談をした"} for _ in range(30)]
    mood = mind._derive_mood(memories)
    defaults = mind._default_state()["mood"]
    assert mood["vigilance"] == defaults["vigilance"]
    assert mood["frustration"] == defaults["frustration"]


def test_screening_event_is_logged(tmp_path):
    state = apply_mood_causes(
        tmp_path,
        [{"axis": "accomplishment", "delta": 2, "rule": "screening_event", "detail": "審査の出来事: 成約"}],
        event="screening:成約",
        trigger="成約",
    )
    entry = state["mood_change_log"][-1]
    assert entry["event"] == "screening:成約"
    assert _rules(entry) == {"screening_event"}


def test_self_emotion_question_detection():
    assert grounding.is_self_emotion_question("君にも感情はあるのか？")
    assert grounding.is_self_emotion_question("今の気分はどう？")
    assert grounding.is_self_emotion_question("しおんは寂しくないの")
    assert not grounding.is_self_emotion_question("お客さんの気持ちを考えると難しい")
    assert not grounding.is_self_emotion_question("この案件のリスクは？")


def test_grounding_block_lists_log_and_evidence_hides_user_text(tmp_path):
    register_dialogue_event(tmp_path, "株式会社テストの件はリスクが高い", signals={"affect": {"label": "不安", "intensity": 0.6}})
    state = load_lease_intelligence_mind(tmp_path)
    block, evidence = grounding.build_grounding_block(state)
    assert "【感情の自己報告: 事実と解釈の言い分け（REV-481）】" in block
    assert "記録にない項目名を事実として作らない" in block
    assert "株式会社テスト" in block  # 紫苑には何の発言で動いたかを見せる
    assert "株式会社テスト" not in evidence  # Jev へは発言本文を送らない
    assert "警戒" in evidence and "→" in evidence
    assert len(block) <= grounding.MAX_BLOCK_CHARS


def test_grounding_block_stays_within_budget(tmp_path):
    for i in range(40):
        register_dialogue_event(tmp_path, f"なぜリスクが{i}？" + "長い発言" * 30)
    block, _ = grounding.build_grounding_block(load_lease_intelligence_mind(tmp_path))
    assert len(block) <= grounding.MAX_BLOCK_CHARS
    assert block.startswith("【感情の自己報告")


def test_extract_state_claims_picks_state_sentences():
    reply = "今の私には慎重な愛着が強く出ています。あなたの言葉で警戒心が和らぎました。今日はどの案件から見ますか？"
    claims = grounding.extract_state_claims(reply)
    assert claims[:2] == ["今の私には慎重な愛着が強く出ています。", "あなたの言葉で警戒心が和らぎました。"]
    assert all("案件から" not in claim for claim in claims)


def test_verify_reply_flags_unsupported_claims():
    reply = "慎重な愛着が強く出ています。納得感のパラメータが上昇しました。"

    def fake_request(payload):
        assert set(payload["questions"]) == {"c0_verdict", "c1_verdict"}
        return {
            "model": "jev-test",
            "answers": {
                "c0_verdict": {"choice": "verified", "confidence": 0.9},
                "c1_verdict": {"choice": "unsupported_fact", "confidence": 0.95},
            },
        }

    result = grounding.verify_reply(reply, "evidence", request_fn=fake_request)
    assert result["status"] == "applied"
    assert [item["claim"] for item in result["mismatches"]] == ["納得感のパラメータが上昇しました。"]
    assert result["counts"]["verified"] == 1


def test_verify_and_log_writes_skip_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("SHION_EMOTION_VERIFY", "off")
    path = tmp_path / "log.jsonl"
    grounding.verify_and_log("君にも感情はある？", "返答", "evidence", surface="test", path=path)
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["status"] == "skipped"
    assert record["surface"] == "test"


def test_experience_loop_mood_no_longer_saturates():
    state = {"mood": {"curiosity": 100, "vigilance": 100, "attachment": 100, "frustration": 0, "accomplishment": 100}}
    event = {"signals": {"relationship_depth": 1, "uncertainty": 0, "practical_depth": 2, "implementation_pressure": 0}}
    for _ in range(40):
        state = update_experience_state(state, event)
    assert all(value < 100 for value in state["mood"].values())


def test_main_helpers_build_block_only_for_emotion_questions(tmp_path, monkeypatch):
    import api.main as main
    import api.shion_relationship as relationship

    register_dialogue_event(tmp_path, "なぜ？")
    block, evidence, kind = main._build_emotion_grounding(tmp_path, "君にも感情はあるのか？")
    assert kind == "serious_emotion" and "事実と解釈の言い分け" in block and evidence
    block, evidence, kind = main._build_emotion_grounding(tmp_path, "この案件のスコアは？", "screening")
    assert kind == "screening" and block.startswith("【審査回答の根拠") and evidence == ""

    monkeypatch.setattr(
        relationship, "get_relationship_state", lambda: {"score": 7.7, "trend": "stable", "delta_history": [0.03, -0.1]}
    )
    signals = main._dialogue_mood_signals(
        {"user_affect": {"label": "喜び", "intensity": 0.7}, "mutual_prediction": {"outcome": {"affect_hit": True}}}
    )
    assert signals["affect"]["label"] == "喜び"
    assert signals["prediction"] == {"affect_hit": True}
    assert signals["relationship"] == {"score": 7.7, "trend": "stable", "last_delta": -0.1}


def test_classify_turn_by_scene():
    assert grounding.classify_turn("君にも感情はあるのか？") == "serious_emotion"
    assert grounding.classify_turn("さっき人間関係の話をした時、君の中で何か変わった？") == "serious_emotion"
    assert grounding.classify_turn("君にも感情はある？", "screening") == "serious_emotion"
    assert grounding.classify_turn("この案件は承認でいい？") == "screening"
    assert grounding.classify_turn("競合の件を整理して", "screening") == "screening"
    assert grounding.classify_turn("しおん、今日は楽しかった？", "casual") == "casual_emotion"
    assert grounding.classify_turn("たこ焼き作ったよ", "casual") == "casual"


def test_report_mode_switch(monkeypatch):
    monkeypatch.delenv("SHION_EMOTION_REPORT_MODE", raising=False)
    assert grounding.report_mode() == "distinguish"
    monkeypatch.setenv("SHION_EMOTION_REPORT_MODE", "recorder")
    assert grounding.report_mode() == "recorder"


def test_turn_blocks_per_scene_and_mode(tmp_path):
    register_dialogue_event(tmp_path, "なぜ？")
    state = load_lease_intelligence_mind(tmp_path)
    serious, evidence = grounding.build_turn_block(state, "serious_emotion", "distinguish")
    assert "「記録上は〜」" in serious and "「私の解釈では〜」" in serious and evidence
    recorder, _ = grounding.build_turn_block(state, "serious_emotion", "recorder")
    assert "記録係モード" in recorder and "推測は述べない" in recorder
    casual, casual_evidence = grounding.build_turn_block(state, "casual_emotion", "distinguish")
    assert "たぶん〜かな" in casual and "物語っぽくても歓迎" in casual and casual_evidence == ""
    screening, _ = grounding.build_turn_block(state, "screening", "distinguish")
    assert "解釈、後付けの理由" in screening
    chat, _ = grounding.build_turn_block({}, "casual", "distinguish")
    assert chat.startswith("【自分の気持ちに触れる時")
    assert grounding.should_verify("serious_emotion") and grounding.should_verify("screening")
    assert not grounding.should_verify("casual_emotion") and not grounding.should_verify("casual")


def _fake(answers):
    return lambda payload: {"model": "jev-test", "answers": answers}


def test_marked_interpretation_is_not_a_mismatch_unless_recorder():
    reply = "記録上は愛着が51から52に上がった。たぶん人間関係の話が嬉しかったのかも、愛着が動いた気がする。"
    answers = {
        "c0_verdict": {"choice": "verified", "confidence": 0.9},
        "c1_verdict": {"choice": "marked_interpretation", "confidence": 0.9},
    }
    distinguish = grounding.verify_reply(reply, "evidence", mode="distinguish", request_fn=_fake(answers))
    assert distinguish["mismatches"] == []
    assert distinguish["counts"]["marked_interpretation"] == 1
    recorder = grounding.verify_reply(reply, "evidence", mode="recorder", request_fn=_fake(answers))
    assert [item["reason"] for item in recorder["mismatches"]] == ["記録係モードで解釈を述べた"]


def test_unsupported_fact_is_a_mismatch():
    reply = "あなたが人間関係の話をした時、警戒が和らいで納得感が上がった。"
    result = grounding.verify_reply(
        reply, "evidence", mode="distinguish", request_fn=_fake({"c0_verdict": {"choice": "unsupported_fact", "confidence": 0.9}})
    )
    assert [item["reason"] for item in result["mismatches"]] == ["記録にないのに事実として語った"]


def test_screening_interpretation_is_flagged():
    reply = "スコアは68点で条件付きです。社長の誠実さが伝わってきたので、私は通したい気持ちです。"
    result = grounding.verify_screening_reply(
        reply,
        request_fn=_fake(
            {
                "c0_verdict": {"choice": "data_basis", "confidence": 0.95},
                "c1_verdict": {"choice": "interpretation_as_basis", "confidence": 0.9},
            }
        ),
    )
    assert [item["reason"] for item in result["mismatches"]] == ["審査の根拠に解釈・後付けを使った"]


def test_verify_and_log_records_kind_and_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("SHION_EMOTION_VERIFY", "off")
    path = tmp_path / "log.jsonl"
    grounding.verify_and_log("承認でいい？", "返答", "", surface="test", kind="screening", mode="recorder", path=path)
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert (record["kind"], record["mode"]) == ("screening", "recorder")
