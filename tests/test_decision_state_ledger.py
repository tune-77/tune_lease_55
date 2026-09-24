from __future__ import annotations

import json

import pytest

from decision_state_ledger import (
    DecisionStateEventError,
    append_event,
    audit_ledger,
    build_decision_changed_event,
    build_event,
    load_events,
)
from scripts.build_judgment_state_projection import project_events


def _evidence() -> list[dict[str, str]]:
    return [
        {
            "kind": "test",
            "id": "source-1",
            "locator": "test://source-1",
            "quote_policy": "summary_only",
        }
    ]


def test_append_event_is_idempotent(tmp_path):
    path = tmp_path / "events.jsonl"
    event = build_decision_changed_event(
        case_id="case-1",
        before="承認",
        after="条件付",
        reason="代替返済原資が未確認",
        source="test",
        record_id=10,
        occurred_at="2026-09-14T01:00:00+00:00",
    )

    first = append_event(path, event)
    second = append_event(path, {**event, "event_id": "jse_duplicate"})

    assert first["recorded"] is True
    assert second["duplicate"] is True
    assert len(load_events(path)) == 1


def test_ai_actor_cannot_finalize_decision():
    with pytest.raises(DecisionStateEventError, match="human actor"):
        build_event(
            event_type="decision_changed",
            aggregate_type="case",
            aggregate_id="case-1",
            actor_type="shion",
            actor_role="advisor",
            surface="test",
            transition_subject="decision",
            transition_action="finalized",
            transition_from={"stance": "conditional"},
            transition_to={"stance": "approve"},
            reason_summary="AIだけで確定しようとした",
            evidence_refs=_evidence(),
        )


def test_projection_replays_decision_and_asset_feedback():
    decision = build_event(
        event_type="decision_changed",
        aggregate_type="case",
        aggregate_id="case-2",
        actor_type="human",
        actor_role="screening_officer",
        surface="test",
        transition_subject="decision",
        transition_action="finalized",
        transition_from={"stance": "conditional"},
        transition_to={"stance": "reject"},
        reason_summary="受注急減を重く見た",
        evidence_refs=_evidence(),
        occurred_at="2026-09-14T02:00:00+00:00",
        idempotency_key="final-case-2",
    )
    asset = build_event(
        event_type="judgment_asset_evaluated",
        aggregate_type="case",
        aggregate_id="case-2",
        actor_type="human",
        actor_role="screening_reviewer",
        surface="test",
        transition_subject="judgment_asset:rule-1",
        transition_action="helped",
        transition_from=None,
        transition_to={"outcome": "helped"},
        reason_summary="確認観点として役立った",
        evidence_refs=_evidence(),
        occurred_at="2026-09-14T03:00:00+00:00",
        idempotency_key="asset-case-2",
        payload={"rule_id": "rule-1"},
    )

    projection = project_events([asset, decision])
    state = projection["cases"][0]

    assert state["current_decision"]["stance"] == "reject"
    assert state["current_decision"]["status"] == "final"
    assert state["current_decision"]["decided_by"] == "human"
    assert state["judgment_asset_evaluations"][0]["rule_id"] == "rule-1"


def test_correction_removes_superseded_event_from_current_state():
    original = build_decision_changed_event(
        case_id="case-3",
        before="承認",
        after="否決",
        reason="入力値を誤認した",
        source="test",
        record_id=30,
        occurred_at="2026-09-14T01:00:00+00:00",
    )
    correction = build_event(
        event_type="event_corrected",
        aggregate_type="case",
        aggregate_id="case-3",
        actor_type="human",
        actor_role="screening_officer",
        surface="test",
        transition_subject="event",
        transition_action="invalidated",
        reason_summary="入力値の訂正",
        evidence_refs=_evidence(),
        supersedes=[original["event_id"]],
        occurred_at="2026-09-14T02:00:00+00:00",
        idempotency_key="correct-case-3",
    )

    projection = project_events([original, correction])

    assert projection["summary"]["corrected_events"] == 1
    assert projection["cases"] == []


def test_audit_reports_invalid_json_and_duplicate_key(tmp_path):
    path = tmp_path / "events.jsonl"
    event = build_decision_changed_event(
        case_id="case-4",
        before="承認",
        after="条件付",
        reason="追加資料が必要",
        source="test",
        record_id=40,
        occurred_at="2026-09-14T01:00:00+00:00",
    )
    duplicate = {**event, "event_id": "jse_second"}
    path.write_text(
        json.dumps(event, ensure_ascii=False) + "\n"
        + json.dumps(duplicate, ensure_ascii=False) + "\n"
        + "not-json\n",
        encoding="utf-8",
    )

    audit = audit_ledger(path)

    assert audit["valid_events"] == 2
    assert audit["duplicate_idempotency_keys"] == 1
    assert audit["invalid_lines"] == 1
