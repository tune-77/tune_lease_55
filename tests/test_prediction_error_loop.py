from __future__ import annotations

import json

from api.prediction_error_loop import (
    build_case_result_prediction_error_payload,
    classify_prediction_error,
    load_update_candidates,
    record_update_candidate_review,
    record_case_result_prediction_error,
    record_prediction_error_event,
    summarize_prediction_errors,
)


def test_records_risk_overestimate_as_reviewable_candidate_after_repeats(tmp_path):
    events_path = tmp_path / "events.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    prediction = {
        "risk_level": "high",
        "main_concern": "資金繰り",
        "recommended_action": "見送り",
        "confidence": 0.74,
        "used_judgment_assets": ["JA-001"],
    }
    outcome = {
        "contracted": True,
        "delinquency": False,
        "user_feedback": "revised",
        "feedback_note": "補助金入金前の借入増加で、危険視しすぎた",
    }

    first = record_prediction_error_event(
        case_id="case-1",
        prediction=prediction,
        outcome=outcome,
        events_path=events_path,
        candidates_path=candidates_path,
        review_threshold=3,
    )
    record_prediction_error_event(
        case_id="case-2",
        prediction=prediction,
        outcome=outcome,
        events_path=events_path,
        candidates_path=candidates_path,
        review_threshold=3,
    )
    third = record_prediction_error_event(
        case_id="case-3",
        prediction=prediction,
        outcome=outcome,
        events_path=events_path,
        candidates_path=candidates_path,
        review_threshold=3,
    )

    assert first["event"]["prediction_error"]["type"] == "risk_overestimated"
    assert first["candidate"]["status"] == "candidate"
    assert third["candidate"]["target_belief"] == "JA-001"
    assert third["candidate"]["repeat_count"] == 3
    assert third["candidate"]["status"] == "ready_for_review"
    assert third["candidate"]["next_action"]["requires_human_review"] is True
    assert third["candidate"]["next_action"]["no_automatic_scoring_or_rag_change"] is True


def test_classifies_sales_context_separately_from_credit_risk():
    error = classify_prediction_error(
        {"risk_level": "medium", "main_concern": "信用リスク", "recommended_action": "追加確認"},
        {
            "contracted": False,
            "delinquency": False,
            "user_feedback": "revised",
            "feedback_note": "信用ではなく競合と価格条件で失注した",
        },
    )

    assert error["type"] == "sales_context_miss"


def test_records_condition_setting_miss_without_promoting_to_scoring_change(tmp_path):
    result = record_prediction_error_event(
        case_id="case-condition",
        prediction={
            "risk_level": "medium",
            "main_concern": "設備妥当性",
            "recommended_action": "条件付き承認",
        },
        outcome={
            "contracted": True,
            "condition_changed": True,
            "user_feedback": "revised",
            "feedback_note": "保証条件とリース期間の条件が弱かった",
        },
        events_path=tmp_path / "events.jsonl",
        candidates_path=tmp_path / "candidates.jsonl",
    )

    assert result["event"]["prediction_error"]["type"] == "condition_setting_miss"
    assert result["candidate"]["next_action"]["kind"] == "revise_condition_template"
    assert result["candidate"]["next_action"]["no_automatic_scoring_or_rag_change"] is True


def test_candidate_loader_and_summary_are_path_scoped(tmp_path):
    events_path = tmp_path / "events.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    record_prediction_error_event(
        case_id="case-a",
        prediction={
            "risk_level": "low",
            "main_concern": "返済原資",
            "recommended_action": "承認",
            "used_judgment_assets": ["JA-LOW"],
        },
        outcome={
            "contracted": True,
            "delinquency": True,
            "user_feedback": "missed",
            "feedback_note": "受注残と入金予定の確認が不足していた",
        },
        events_path=events_path,
        candidates_path=candidates_path,
    )

    candidates = load_update_candidates(candidates_path=candidates_path)
    summary = summarize_prediction_errors(events_path=events_path, candidates_path=candidates_path)

    assert candidates[0]["error_type"] == "risk_underestimated"
    assert summary["total_events"] == 1
    assert summary["by_error_type"]["risk_underestimated"] == 1
    assert summary["guardrail"] == "records_candidates_only_no_scoring_no_rag_no_prompt_change"
    assert "case-a" in json.dumps(candidates, ensure_ascii=False)


def test_builds_prediction_error_payload_from_registered_case_result():
    payload = build_case_result_prediction_error_payload(
        case_data={
            "score": 82,
            "result": {
                "hantei": "承認",
                "diagnostic_recommendations": ["返済原資は受注残で確認"],
                "used_judgment_assets": ["JA-CASHFLOW"],
            },
        },
        status="失注",
        patches={
            "lost_reason": "他社競合の価格条件で失注",
            "competitor_rate": 1.8,
        },
    )

    assert payload["prediction"]["risk_level"] == "low"
    assert payload["prediction"]["recommended_action"] == "承認"
    assert payload["prediction"]["main_concern"] == "返済原資は受注残で確認"
    assert payload["prediction"]["used_judgment_assets"] == ["JA-CASHFLOW"]
    assert payload["outcome"]["contracted"] is False
    assert payload["outcome"]["user_feedback"] == "revised"
    assert "競合" in payload["outcome"]["feedback_note"]


def test_case_result_registration_records_prediction_error_event(tmp_path):
    result = record_case_result_prediction_error(
        case_id="case-registration",
        case_data={
            "score": 45,
            "result": {"hantei": "否決", "main_concern": "設備妥当性"},
        },
        status="成約",
        patches={"final_note": "保証追加後に成約した"},
        events_path=tmp_path / "events.jsonl",
        candidates_path=tmp_path / "candidates.jsonl",
    )

    assert result["status"] == "recorded"
    assert result["event"]["outcome"]["contracted"] is True
    assert result["candidate"]["next_action"]["requires_human_review"] is True


def test_case_result_prediction_error_skips_non_final_status(tmp_path):
    result = record_case_result_prediction_error(
        case_id="case-pending",
        case_data={"score": 60, "result": {"hantei": "条件付き承認"}},
        status="稟議中",
        events_path=tmp_path / "events.jsonl",
        candidates_path=tmp_path / "candidates.jsonl",
    )

    assert result == {"status": "skipped", "reason": "status_not_observable_for_prediction_error"}


def test_review_candidate_overlays_latest_human_decision(tmp_path):
    events_path = tmp_path / "events.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"
    recorded = record_case_result_prediction_error(
        case_id="case-review",
        case_data={"score": 82, "result": {"hantei": "承認", "main_concern": "価格条件"}},
        status="失注",
        patches={"lost_reason": "競合金利で負けた"},
        events_path=events_path,
        candidates_path=candidates_path,
    )
    candidate_id = recorded["candidate"]["candidate_id"]

    review = record_update_candidate_review(
        candidate_id=candidate_id,
        decision="revised",
        note="信用ではなく競合リスクとして扱う",
        edited_update="競合条件を初期確認に入れる",
        candidates_path=candidates_path,
        reviews_path=reviews_path,
    )
    candidates = load_update_candidates(candidates_path=candidates_path, reviews_path=reviews_path)
    summary = summarize_prediction_errors(
        events_path=events_path,
        candidates_path=candidates_path,
        reviews_path=reviews_path,
    )

    assert review["status"] == "recorded"
    assert candidates[0]["status"] == "revised"
    assert candidates[0]["review_decision"] == "revised"
    assert candidates[0]["review"]["edited_update"] == "競合条件を初期確認に入れる"
    assert summary["review_counts"]["revised"] == 1


def test_review_candidate_rejects_unknown_candidate(tmp_path):
    try:
        record_update_candidate_review(
            candidate_id="missing",
            decision="adopted",
            candidates_path=tmp_path / "candidates.jsonl",
            reviews_path=tmp_path / "reviews.jsonl",
        )
    except ValueError as exc:
        assert "candidate not found" in str(exc)
    else:
        raise AssertionError("expected ValueError")
