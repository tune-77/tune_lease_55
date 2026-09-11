from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from api.routers import feedback_loop
from judgment_asset_bandit import build_bandit_signals, read_feedback_rows, select_current_feedback_rows
from scripts.judgment_asset_growth_report import summarize_field_feedback


CANDIDATE_ID = "cr-b259411afb954d6d"
CANONICAL_RULE_ID = "b259411afb954d6d"
EVENT_1 = "11111111-1111-4111-8111-111111111111"
EVENT_2 = "22222222-2222-4222-8222-222222222222"
EVENT_3 = "33333333-3333-4333-8333-333333333333"


@pytest.fixture
def feedback_store(tmp_path, monkeypatch):
    state_path = tmp_path / "candidate_state.json"
    feedback_path = tmp_path / "feedback.jsonl"
    lock_path = tmp_path / "feedback.lock"
    candidate = {
        "id": CANDIDATE_ID,
        "claim": "受注根拠と返済原資を確認する。",
        "candidate_type": "application_rule",
    }
    monkeypatch.setattr(feedback_loop, "_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON", state_path)
    monkeypatch.setattr(feedback_loop, "_JUDGMENT_ASSET_USAGE_FEEDBACK_LOG", feedback_path)
    monkeypatch.setattr(feedback_loop, "_JUDGMENT_ASSET_CANDIDATE_FEEDBACK_LOCK", lock_path)
    monkeypatch.setattr(feedback_loop, "_load_canonical_judgment_asset_candidates", lambda *a, **k: [candidate])
    monkeypatch.setattr(feedback_loop, "_load_autoresearch_judgment_asset_candidates", lambda *a, **k: [])
    monkeypatch.setattr(feedback_loop, "_load_news_judgment_signals", lambda *a, **k: [])
    return state_path, feedback_path


def _request(
    feedback: str,
    event_id: str,
    *,
    supersedes_event_id: str = "",
) -> feedback_loop.JudgmentAssetCandidateFeedbackRequest:
    return feedback_loop.JudgmentAssetCandidateFeedbackRequest(
        feedback=feedback,
        case_id="case-1",
        review_id=7,
        event_id=event_id,
        supersedes_event_id=supersedes_event_id,
        recorded_at="2026-09-11T00:00:00Z",
        source="real_case",
    )


def test_same_event_id_is_idempotent(feedback_store):
    state_path, feedback_path = feedback_store
    first = feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, _request("useful", EVENT_1)
    )
    duplicate = feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, _request("useful", EVENT_1)
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))[CANDIDATE_ID]
    assert state["use_count"] == 1
    assert state["useful_count"] == 1
    assert len(read_feedback_rows(feedback_path)) == 1
    assert first["duplicate"] is False
    assert duplicate["duplicate"] is True


def test_cloudrun_writeback_preserves_comment_and_edited_claim(feedback_store):
    captured: dict = {}

    class BackgroundTasksCapture:
        def add_task(self, _function, **kwargs):
            captured.update(kwargs)

    request = _request("neutral", EVENT_1)
    request.comment = "案件実績に合わせた補足"
    request.edited_claim = "受注実績と稼働率を合わせて確認する。"

    feedback_loop.post_judgment_asset_candidate_feedback(
        CANDIDATE_ID,
        request,
        BackgroundTasksCapture(),
    )

    assert captured["event_type"] == "judgment_asset_candidate_feedback"
    assert captured["payload"]["comment"] == "案件実績に合わせた補足"
    assert captured["payload"]["edited_claim"] == "受注実績と稼働率を合わせて確認する。"
    assert captured["payload"]["recorded_at"] == "2026-09-11T00:00:00Z"


def test_correction_supersedes_old_effectiveness(feedback_store):
    state_path, feedback_path = feedback_store
    feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, _request("useful", EVENT_1)
    )
    corrected = feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, _request("neutral", EVENT_2, supersedes_event_id=EVENT_1)
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))[CANDIDATE_ID]
    assert state["use_count"] == 1
    assert state["useful_count"] == 0
    assert state["neutral_count"] == 1
    current = select_current_feedback_rows(read_feedback_rows(feedback_path))
    assert [row["event_id"] for row in current] == [EVENT_2]
    assert corrected["feedback_event"]["disposition"] == "challenged"
    signal = build_bandit_signals(read_feedback_rows(feedback_path))[CANDIDATE_ID]
    assert signal.total_feedback == 1
    assert signal.counts["challenged"] == 1


def test_not_applied_is_audited_but_not_counted(feedback_store):
    state_path, feedback_path = feedback_store
    feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, _request("useful", EVENT_1)
    )
    feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, _request("not_applied", EVENT_2, supersedes_event_id=EVENT_1)
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))[CANDIDATE_ID]
    assert state["use_count"] == 0
    assert state["useful_count"] == 0
    current = select_current_feedback_rows(read_feedback_rows(feedback_path))
    assert len(current) == 1
    assert current[0]["outcome"] == "not_applied"
    assert build_bandit_signals(read_feedback_rows(feedback_path)) == {}
    summary = summarize_field_feedback(
        read_feedback_rows(feedback_path),
        [{"id": CANONICAL_RULE_ID, "status": "active", "concept": "test"}],
    )
    assert summary["totals"]["used"] == 0
    assert summary["totals"]["helped"] == 0


def test_canonical_candidate_prefix_maps_to_active_rule(feedback_store):
    _state_path, feedback_path = feedback_store
    feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, _request("useful", EVENT_1)
    )

    summary = summarize_field_feedback(
        read_feedback_rows(feedback_path),
        [{"id": CANONICAL_RULE_ID, "status": "active", "concept": "test"}],
    )

    assert summary["totals"]["used"] == 1
    assert summary["totals"]["helped"] == 1
    assert summary["totals"]["unknown_rule"] == 0
    assert summary["rules"][0]["rule_id"] == CANONICAL_RULE_ID


def test_stale_parallel_correction_returns_conflict(feedback_store):
    _state_path, _feedback_path = feedback_store
    feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, _request("useful", EVENT_1)
    )
    feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, _request("neutral", EVENT_2, supersedes_event_id=EVENT_1)
    )

    with pytest.raises(HTTPException) as exc_info:
        feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
            CANDIDATE_ID, _request("rejected", EVENT_3, supersedes_event_id=EVENT_1)
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["current_event_id"] == EVENT_2


def test_second_root_requires_explicit_supersedes(feedback_store):
    _state_path, _feedback_path = feedback_store
    feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, _request("useful", EVENT_1)
    )

    with pytest.raises(HTTPException) as exc_info:
        feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
            CANDIDATE_ID, _request("rejected", EVENT_2)
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["current_event_id"] == EVENT_1


def test_current_head_can_be_restored_after_page_reload(feedback_store):
    feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, _request("useful", EVENT_1)
    )
    feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, _request("neutral", EVENT_2, supersedes_event_id=EVENT_1)
    )

    heads = feedback_loop._candidate_feedback_heads("case-1", 7)

    assert heads[CANDIDATE_ID]["event_id"] == EVENT_2
    assert heads[CANDIDATE_ID]["feedback"] == "neutral"


def test_real_case_feedback_requires_case_id(feedback_store):
    request = _request("useful", EVENT_1)
    request.case_id = ""

    with pytest.raises(HTTPException) as exc_info:
        feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(CANDIDATE_ID, request)

    assert exc_info.value.status_code == 422
