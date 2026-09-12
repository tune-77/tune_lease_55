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


def test_durable_duplicate_repairs_missing_local_materialization(feedback_store, monkeypatch):
    state_path, feedback_path = feedback_store
    request = _request("useful", EVENT_1)
    durable_event = {
        "event_type": "judgment_asset_candidate_feedback",
        "payload": {
            "event_id": EVENT_1,
            "supersedes_event_id": "",
            "candidate_id": CANDIDATE_ID,
            "feedback": "useful",
            "disposition": "helped",
            "case_id": "case-1",
            "review_id": 7,
            "recorded_at": "2026-09-11T00:00:00Z",
        },
    }
    monkeypatch.setenv("K_SERVICE", "test-service")
    monkeypatch.setattr(
        feedback_loop,
        "read_judgment_asset_feedback_events",
        lambda _case_id, _review_id: [durable_event],
    )
    monkeypatch.setattr(
        feedback_loop,
        "record_judgment_asset_feedback_event",
        lambda _payload: {"ok": True, "duplicate": True},
    )

    repaired = feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, request
    )
    duplicate = feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, request
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))[CANDIDATE_ID]
    assert state["use_count"] == 1
    assert state["useful_count"] == 1
    assert [row["event_id"] for row in read_feedback_rows(feedback_path)] == [EVENT_1]
    assert repaired["duplicate"] is True
    assert duplicate["duplicate"] is True


def test_cloudrun_writeback_preserves_comment_and_edited_claim(feedback_store):
    captured: list[dict] = []

    class BackgroundTasksCapture:
        def add_task(self, _function, **kwargs):
            captured.append(kwargs)

    request = _request("neutral", EVENT_1)
    request.comment = "案件実績に合わせた補足"
    request.edited_claim = "受注実績と稼働率を合わせて確認する。"

    feedback_loop.post_judgment_asset_candidate_feedback(CANDIDATE_ID, request, BackgroundTasksCapture())
    feedback_loop.post_judgment_asset_candidate_feedback(CANDIDATE_ID, request, BackgroundTasksCapture())

    assert len(captured) == 2
    assert captured[-1]["event_type"] == "judgment_asset_candidate_feedback"
    assert captured[-1]["payload"]["comment"] == "案件実績に合わせた補足"
    assert captured[-1]["payload"]["edited_claim"] == "受注実績と稼働率を合わせて確認する。"
    assert captured[-1]["payload"]["recorded_at"] == "2026-09-11T00:00:00Z"


def test_correction_supersedes_old_effectiveness(feedback_store):
    state_path, feedback_path = feedback_store
    initial = _request("useful", EVENT_1)
    initial.edited_claim = "受注見込みだけを確認する。"
    feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(CANDIDATE_ID, initial)
    correction = _request("neutral", EVENT_2, supersedes_event_id=EVENT_1)
    correction.edited_claim = "受注根拠と返済原資を確認する。"
    corrected = feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, correction
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))[CANDIDATE_ID]
    assert state["use_count"] == 1
    assert state["useful_count"] == 0
    assert state["neutral_count"] == 1
    assert state["edited_claim"] == "受注根拠と返済原資を確認する。"
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


def test_current_head_can_be_restored_after_page_reload(feedback_store, monkeypatch):
    state_path, feedback_path = feedback_store
    feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, _request("useful", EVENT_1)
    )
    feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(
        CANDIDATE_ID, _request("neutral", EVENT_2, supersedes_event_id=EVENT_1)
    )
    durable_rows = read_feedback_rows(feedback_path)
    feedback_path.unlink()
    durable_events = [{"event_type": "judgment_asset_candidate_feedback", "payload": {**row, "candidate_id": row["rule_id"], "disposition": row["outcome"]}} for row in durable_rows]
    monkeypatch.setenv("K_SERVICE", "test-service")
    def read_durable_events(case_id, review_id):
        assert (case_id, review_id) == ("case-1", 7)
        return durable_events
    monkeypatch.setattr(feedback_loop, "read_judgment_asset_feedback_events", read_durable_events)
    monkeypatch.setattr(feedback_loop, "record_judgment_asset_feedback_event", lambda payload: {"ok": True})

    heads = feedback_loop._candidate_feedback_heads("case-1", 7)

    assert heads[CANDIDATE_ID]["event_id"] == EVENT_2
    assert heads[CANDIDATE_ID]["feedback"] == "neutral"
    state_path.write_text(json.dumps({CANDIDATE_ID: {"use_count": 5, "useful_count": 5}}))
    feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(CANDIDATE_ID, _request("rejected", EVENT_3, supersedes_event_id=EVENT_2))
    state = json.loads(state_path.read_text())
    assert (state[CANDIDATE_ID]["useful_count"], state[CANDIDATE_ID]["rejected_count"]) == (5, 1)


def test_real_case_feedback_requires_case_id(feedback_store):
    request = _request("useful", EVENT_1)
    request.case_id = ""

    with pytest.raises(HTTPException) as exc_info:
        feedback_loop._update_autoresearch_judgment_asset_candidate_feedback(CANDIDATE_ID, request)

    assert exc_info.value.status_code == 422
