from api.screening_followup import (
    answer_followup_session,
    build_followup_questions,
    build_updated_view,
    create_followup_session,
    list_followup_sessions,
    record_followup_outcome,
)


def test_followup_questions_are_limited_and_prioritize_missing_high_impact_items():
    questions = build_followup_questions(
        {
            "asset_purpose": "",
            "asset_name": "大型加工機",
            "asset_evidence_level": "未確認",
            "customer_type": "新規先",
            "competitor": "競合あり",
            "acquisition_cost": 80,
            "nenshu": 400,
        },
        {"score": 52, "hantei": "条件付き承認", "quantum_risk": 62},
    )

    assert len(questions) == 3
    assert questions[0]["id"] == "asset-purpose"
    assert any(question["id"] == "numeric-consistency" for question in questions)
    assert all(len(question["answer_options"]) == 3 for question in questions)


def test_judgment_asset_confirmation_question_can_enter_followup_selection():
    questions = build_followup_questions(
        {"asset_purpose": "増産", "asset_name": "設備", "asset_evidence_level": "確認済"},
        {"score": 82, "hantei": "承認"},
        [{
            "id": "ja-1",
            "candidate_type": "confirmation_question",
            "claim": "既存設備の稼働率と今回増設の必要性は整合していますか？",
        }],
    )

    assert any(question["source_asset_id"] == "ja-1" for question in questions)


def test_partial_answers_become_conditions_without_changing_score():
    questions = build_followup_questions(
        {"asset_purpose": "増産", "asset_name": "設備", "asset_evidence_level": "未確認"},
        {"score": 55, "hantei": "条件付き承認"},
    )
    answers = [
        {"question_id": question["id"], "status": "partial", "note": "資料は営業確認中"}
        for question in questions
    ]

    updated = build_updated_view("条件付き承認", questions, answers)

    assert updated["updated_decision"] == "条件付きで進行可"
    assert updated["approval_conditions"]
    assert updated["score_changed"] is False
    assert len(updated["verification_targets"]) == len(questions)


def test_concern_answer_keeps_stop_line():
    questions = build_followup_questions({}, {"score": 30, "hantei": "否決"})
    answers = [
        {"question_id": question["id"], "status": "concern", "note": "裏付けなし"}
        for question in questions
    ]

    updated = build_updated_view("否決", questions, answers)

    assert updated["updated_decision"] == "追加確認を継続"
    assert "停止線" in updated["change_reason"]


def test_followup_session_persists_answers_and_links_outcome(tmp_path, monkeypatch):
    import runtime_paths

    db_path = tmp_path / "followup.db"
    monkeypatch.setattr(runtime_paths, "get_db_path", lambda: str(db_path))
    monkeypatch.setattr(runtime_paths, "ensure_cloudrun_demo_db_seeded", lambda: None)

    created = create_followup_session(
        case_id="CASE-001",
        review_id=12,
        form={"asset_purpose": "増産", "asset_name": "設備", "asset_evidence_level": "未確認"},
        result={"score": 55, "hantei": "条件付き承認"},
    )
    answered = answer_followup_session(
        created["followup_id"],
        [
            {"question_id": question["id"], "status": "confirmed", "note": "営業確認済み"}
            for question in created["questions"]
        ],
    )
    linked = record_followup_outcome("CASE-001", "成約", "条件履行済み")
    saved = list_followup_sessions("CASE-001", limit=1)[0]

    assert answered["status"] == "answered"
    assert linked["linked_count"] == 1
    assert saved["status"] == "outcome_linked"
    assert saved["outcome_status"] == "成約"


def test_followup_rejects_partial_payload_and_post_outcome_rewrite(tmp_path, monkeypatch):
    import pytest
    import runtime_paths

    db_path = tmp_path / "followup-locked.db"
    monkeypatch.setattr(runtime_paths, "get_db_path", lambda: str(db_path))
    monkeypatch.setattr(runtime_paths, "ensure_cloudrun_demo_db_seeded", lambda: None)
    created = create_followup_session(
        case_id="CASE-LOCK",
        review_id=None,
        form={},
        result={"score": 45},
    )
    first = created["questions"][0]
    with pytest.raises(ValueError, match="every followup question"):
        answer_followup_session(
            created["followup_id"],
            [{"question_id": first["id"], "status": "confirmed", "note": ""}],
        )

    full_answers = [
        {"question_id": question["id"], "status": "confirmed", "note": ""}
        for question in created["questions"]
    ]
    answer_followup_session(created["followup_id"], full_answers)
    record_followup_outcome("CASE-LOCK", "失注")
    with pytest.raises(ValueError, match="outcome-linked"):
        answer_followup_session(created["followup_id"], full_answers)
