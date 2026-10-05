from datetime import datetime, timedelta

import pytest

from api.chat_prompt_budget import SHAPE, SPECS
from api.shion_mutual_prediction import (
    begin_turn,
    build_reflection_material,
    finish_turn,
    get_user_summary,
)

T0 = datetime(2026, 10, 5, 21, 0)


def _affect(label="通常", cues=()):
    return {"label": label, "intensity": 0.8 if label != "通常" else 0.0, "cues": list(cues)}


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.delenv("SHION_CURIOSITY_ENABLED", raising=False)
    monkeypatch.delenv("SHION_CURIOSITY_COOLDOWN_HOURS", raising=False)
    return {"path": tmp_path / "state.json", "log_path": tmp_path / "log.jsonl"}


def _turn(paths, message, label="通常", *, mode="casual", now=T0, cues=(), recurrent=""):
    return begin_turn(
        "alice",
        message=message,
        affect_payload=_affect(label, cues),
        context_mode=mode,
        surface="test",
        recurrent=recurrent,
        now=now,
        **paths,
    )


def test_first_turn_only_records_a_prediction(paths):
    turn = _turn(paths, "疲れた…", "疲れ")
    assert turn.outcome is None
    pending = get_user_summary("alice", path=paths["path"])["pending_prediction"]
    assert pending["affect_same_session"] == "疲れ"
    assert pending["affect_next_session"] == "通常"


def test_miss_is_fed_back_into_next_prompt_and_reflection_log(paths):
    _turn(paths, "疲れた…", "疲れ")
    turn = _turn(paths, "やった、通った！", "喜び", now=T0 + timedelta(minutes=5))
    assert turn.outcome is not None and not turn.outcome.affect_hit
    assert turn.outcome.surprise >= 0.6  # 正反対は大きな驚き
    assert "予想は外れた" in turn.prompt_block and "「疲れ」" in turn.prompt_block
    material = build_reflection_material(["2026-10-05"], path=paths["log_path"])
    assert "「疲れ」だと予想" in material and "「喜び」" in material


def test_hit_on_recovery_in_next_session(paths):
    _turn(paths, "疲れた…", "疲れ")
    turn = _turn(paths, "おはよう", "通常", now=T0 + timedelta(hours=12))
    assert turn.outcome.new_session and turn.outcome.affect_hit
    assert "予想は外れた" not in turn.prompt_block


def test_reaction_miss_on_complaint_to_shion(paths):
    _turn(paths, "金利の目安は？", "通常", mode="normal")
    turn = _turn(paths, "違うって、何度も言ってる", "苛立ち", mode="normal", cues=("違うって", "何度も"),
                 now=T0 + timedelta(minutes=2))
    assert not turn.outcome.reaction_hit
    assert "期待とずれていた" in turn.prompt_block


def test_curiosity_is_asked_once_in_next_casual_session(paths):
    _turn(paths, "疲れた…", "疲れ")
    turn = _turn(paths, "おはよう", "通常", now=T0 + timedelta(hours=12))
    assert turn.question and turn.question["question"] == "この前疲れてたけど、何があったの？"
    assert "1回だけ" in turn.prompt_block
    assert finish_turn(turn, "おはよ。この前疲れてたけど、何があったの？", now=T0 + timedelta(hours=12), path=paths["path"])

    # 答えを受け止める指示が出て、同じことは二度聞かない
    answer = _turn(paths, "決算資料の締めが重なってて", "通常", now=T0 + timedelta(hours=12, minutes=1))
    assert "その答えかもしれない" in answer.prompt_block
    assert answer.question is None
    assert get_user_summary("alice", path=paths["path"])["curiosities"] == []


def test_question_not_counted_when_reply_has_no_question(paths):
    _turn(paths, "疲れた…", "疲れ")
    turn = _turn(paths, "おはよう", "通常", now=T0 + timedelta(hours=12))
    assert not finish_turn(turn, "おはよ。今日もよろしくね。", path=paths["path"])
    retry = _turn(paths, "今日は晴れだね", "通常", now=T0 + timedelta(hours=12, minutes=3))
    assert retry.question is not None


def test_no_question_during_screening_or_when_rushed(paths):
    _turn(paths, "疲れた…", "疲れ")
    screening = _turn(paths, "この案件のスコアを見て", "通常", mode="screening", now=T0 + timedelta(hours=12))
    assert screening.question is None
    rushed = _turn(paths, "至急！今日中に出したい", "焦り", now=T0 + timedelta(hours=12, minutes=5))
    assert rushed.question is None


def test_no_question_on_business_question_in_normal_mode(paths):
    _turn(paths, "疲れた…", "疲れ")
    turn = _turn(paths, "リース料率の決め方を教えて", "通常", mode="normal", now=T0 + timedelta(hours=12))
    assert turn.question is None


def test_question_mark_in_body_only_is_not_counted(paths):
    _turn(paths, "疲れた…", "疲れ")
    turn = _turn(paths, "おはよう", "通常", now=T0 + timedelta(hours=12))
    reply = "おはよ。昨日の件、覚えてる？" + "今日はゆっくりいこうね。" * 15
    assert not finish_turn(turn, reply, path=paths["path"])


def test_cooldown_limits_questions(paths):
    _turn(paths, "疲れた…", "疲れ")
    first = _turn(paths, "おはよう", "通常", now=T0 + timedelta(hours=12))
    finish_turn(first, "何があったの？", now=T0 + timedelta(hours=12), path=paths["path"])
    _turn(paths, "まあね", "通常", now=T0 + timedelta(hours=12, minutes=1))
    _turn(paths, "不安なんだよね", "不安", now=T0 + timedelta(hours=13))
    later = _turn(paths, "こんばんは", "通常", now=T0 + timedelta(hours=16))
    assert later.question is None  # 20時間のクールダウン中
    next_day = _turn(paths, "おはよう", "通常", now=T0 + timedelta(hours=34))
    assert next_day.question is not None


def test_flag_disables_questions(paths, monkeypatch):
    monkeypatch.setenv("SHION_CURIOSITY_ENABLED", "0")
    _turn(paths, "疲れた…", "疲れ")
    turn = _turn(paths, "おはよう", "通常", now=T0 + timedelta(hours=12))
    assert turn.question is None


def test_known_cause_does_not_create_curiosity(paths):
    _turn(paths, "決算資料の締め切りが重なったから、さすがに疲れた", "疲れ")
    assert get_user_summary("alice", path=paths["path"])["curiosities"] == []


def test_model_learns_quick_mood_shifts(paths):
    now = T0
    for _ in range(4):
        _turn(paths, "疲れた…", "疲れ", now=now)
        now += timedelta(minutes=2)
        _turn(paths, "まあいいや", "通常", now=now)
        now += timedelta(minutes=2)
    turn = _turn(paths, "疲れた…", "疲れ", now=now)
    assert turn.payload["quick_to_shift"]
    assert get_user_summary("alice", path=paths["path"])["pending_prediction"]["affect_same_session"] == "通常"
    assert "切り替えが早い" in turn.prompt_block


def test_budget_registers_block():
    assert SPECS["mutual_prediction_context"].tier == SHAPE


def test_relationship_understanding(tmp_path, monkeypatch):
    import api.shion_relationship as rel

    monkeypatch.setattr(rel, "_STATE_PATH", tmp_path / "rel.json")
    before = rel.get_relationship_state()["score"]
    state = rel.record_prediction_outcome(hit=True, surprise=0.0)
    assert state["score"] > before and state["understanding"] > 0.5
    state = rel.record_prediction_outcome(hit=False, surprise=0.8)
    assert state["score"] == pytest.approx(before + 0.03)  # 外れでは下げない
    assert rel.get_fear_context()["understanding"] == state["understanding"]
