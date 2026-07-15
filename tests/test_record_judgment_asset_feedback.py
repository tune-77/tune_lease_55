import json

import pytest

from scripts import record_judgment_asset_feedback as feedback


def test_build_feedback_entry_normalizes_required_fields():
    entry = feedback.build_feedback_entry(
        rule_id=" rule-1 ",
        outcome="HELPED",
        case_id=" case-001 ",
        note="稟議コメントに使えた",
        used_at="2026-07-15T12:00:00",
    )

    assert entry["rule_id"] == "rule-1"
    assert entry["outcome"] == "helped"
    assert entry["case_id"] == "case-001"
    assert entry["used_at"] == "2026-07-15T12:00:00"


def test_invalid_feedback_outcome_is_rejected():
    with pytest.raises(ValueError):
        feedback.build_feedback_entry(rule_id="rule-1", outcome="maybe")


def test_append_feedback_writes_jsonl(tmp_path):
    path = tmp_path / "feedback.jsonl"
    entry = feedback.build_feedback_entry(
        rule_id="rule-1",
        outcome="challenged",
        note="案件に合わなかった",
        used_at="2026-07-15T12:00:00",
    )

    feedback.append_feedback(path, entry)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [entry]
