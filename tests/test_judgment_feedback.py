from __future__ import annotations

from decision_state_ledger import load_events
from judgment_feedback import (
    build_judgment_learning_prompt_block,
    get_judgment_feedback_summary,
    load_judgment_training_candidates,
    record_judgment_feedback,
    review_judgment_feedback,
)


def test_changed_judgment_becomes_review_candidate(tmp_path):
    db_path = str(tmp_path / "lease.db")
    result = record_judgment_feedback(
        case_id="case-1",
        model_decision="条件付承認",
        human_decision="否決",
        reason="受注急減の影響を重く見た",
        source="test",
        score=52,
        input_snapshot={"company_name": "秘密会社", "nenshu": 100},
        db_path=db_path,
    )

    assert result["success"] is True
    assert result["review_status"] == "candidate"
    assert get_judgment_feedback_summary(db_path)["candidates"] == 1

    candidates = load_judgment_training_candidates(approved_only=False, db_path=db_path)
    assert candidates[0]["target_label"] == 0
    assert candidates[0]["input_snapshot"]["company_name"] == "[REDACTED]"
    events = load_events(tmp_path / "judgment_state_events.jsonl")
    assert len(events) == 1
    assert events[0]["event_type"] == "decision_changed"
    assert events[0]["transition"]["from"]["stance"] == "conditional"
    assert events[0]["transition"]["to"]["stance"] == "reject"
    assert result["state_event_recorded"] is True


def test_unchanged_judgment_is_not_recorded(tmp_path):
    db_path = str(tmp_path / "lease.db")
    result = record_judgment_feedback(
        case_id="case-1",
        model_decision="承認",
        human_decision="承認",
        reason="変更なしのため保存しない",
        source="test",
        db_path=db_path,
    )

    assert result["success"] is False
    assert "not changed" in result["error"]


def test_only_approved_rows_feed_training_export(tmp_path):
    db_path = str(tmp_path / "lease.db")
    saved = record_judgment_feedback(
        case_id="case-2",
        model_decision="承認",
        human_decision="条件付",
        reason="追加担保を条件にする",
        source="test",
        db_path=db_path,
    )
    reviewed = review_judgment_feedback(saved["record_id"], "approved", db_path)
    assert reviewed["success"] is True

    candidates = load_judgment_training_candidates(db_path=db_path)
    assert len(candidates) == 1
    assert candidates[0]["target_label"] == 1

    block = build_judgment_learning_prompt_block(db_path=db_path)
    assert "承認→条件付" in block
    assert "追加担保を条件にする" in block


def test_unapproved_feedback_is_not_in_learning_prompt(tmp_path):
    db_path = str(tmp_path / "lease.db")
    record_judgment_feedback(
        case_id="case-3",
        model_decision="承認",
        human_decision="否決",
        reason="直近の受注急減を重く見た",
        source="test",
        db_path=db_path,
    )

    assert build_judgment_learning_prompt_block(db_path=db_path) == ""


def test_registered_business_result_is_an_outcome_not_a_human_decision(tmp_path):
    db_path = str(tmp_path / "lease.db")
    result = record_judgment_feedback(
        case_id="case-outcome",
        model_decision="条件付",
        human_decision="承認",
        reason="案件登録トリガー: 成約",
        source="register_trigger",
        db_path=db_path,
    )

    events = load_events(tmp_path / "judgment_state_events.jsonl")
    assert result["success"] is True
    assert events[0]["event_type"] == "outcome_recorded"
    assert events[0]["transition"]["to"]["status"] == "approve"


def test_ledger_failure_does_not_rollback_feedback(tmp_path):
    db_path = str(tmp_path / "lease.db")
    result = record_judgment_feedback(
        case_id="case-ledger-failure",
        model_decision="承認",
        human_decision="否決",
        reason="判断理由はDBへ残す",
        source="test",
        db_path=db_path,
        ledger_path=tmp_path,
    )

    assert result["success"] is True
    assert result["state_event_recorded"] is False
    assert result["state_event_error"]
    assert len(load_judgment_training_candidates(approved_only=False, db_path=db_path)) == 1
