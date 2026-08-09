import json
from pathlib import Path

from api.routers import feedback_loop
from judgment_asset_bandit import (
    append_feedback_event,
    build_bandit_signals,
    read_feedback_rows,
    signal_for_candidate,
)


def test_helped_feedback_adds_positive_rank_bonus_for_canonical_candidate():
    signals = build_bandit_signals(
        [
            {"rule_id": "asset-life", "outcome": "helped", "used_at": "2026-07-28T01:00:00"},
            {"rule_id": "asset-life", "outcome": "used", "used_at": "2026-07-28T02:00:00"},
        ]
    )

    signal = signal_for_candidate({"id": "cr-asset-life"}, signals)

    assert signal.total_feedback == 2
    assert signal.counts["helped"] == 1
    assert signal.reward_score > 0
    assert signal.rank_bonus > 0
    assert signal.last_feedback_at == "2026-07-28T02:00:00"


def test_rejected_feedback_penalizes_candidate():
    signals = build_bandit_signals(
        [
            {"candidate_id": "candidate-a", "feedback": "rejected"},
            {"candidate_id": "candidate-a", "outcome": "rejected"},
        ]
    )

    signal = signal_for_candidate({"id": "candidate-a"}, signals)

    assert signal.total_feedback == 2
    assert signal.counts["rejected"] == 2
    assert signal.reward_score < 0
    assert signal.rank_bonus < 0


def test_append_feedback_event_writes_jsonl(tmp_path):
    path = tmp_path / "feedback.jsonl"

    assert append_feedback_event(
        asset_id="candidate-b",
        outcome="useful",
        case_id="case-1",
        note="稟議条件に使えた",
        source="test",
        path=path,
    )

    rows = read_feedback_rows(path)

    assert len(rows) == 1
    assert rows[0]["rule_id"] == "candidate-b"
    assert rows[0]["outcome"] == "helped"
    assert json.loads(path.read_text(encoding="utf-8"))["source"] == "test"


def test_screening_review_feedback_uses_repo_data_ledger():
    expected = Path(feedback_loop._REPO_ROOT) / "data" / "judgment_asset_usage_feedback.jsonl"

    assert feedback_loop._JUDGMENT_ASSET_USAGE_FEEDBACK_LOG == expected
    assert "/api/data/" not in str(expected)
