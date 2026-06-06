from __future__ import annotations

from judgment_feedback import (
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
