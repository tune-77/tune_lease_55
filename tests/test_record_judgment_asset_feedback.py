import json
from pathlib import Path

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


def _write_canonical(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rules": [
                    {
                        "id": "active-1",
                        "status": "active",
                        "concept": "asset_life",
                        "canonical_statement": "経済的寿命と満了後出口を確認する。",
                        "private": False,
                    },
                    {
                        "id": "demoted-1",
                        "status": "demoted",
                        "concept": "demo_ops",
                        "canonical_statement": "デモ説明を整える。",
                        "private": False,
                    },
                    {
                        "id": "private-1",
                        "status": "active",
                        "concept": "private_memory",
                        "canonical_statement": "非公開メモ。",
                        "private": True,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_load_active_rules_excludes_demoted_and_private_rules(tmp_path):
    canonical = tmp_path / "canonical.json"
    _write_canonical(canonical)

    active_rules = feedback.load_active_rules(canonical)

    assert list(active_rules) == ["active-1"]


def test_resolve_rule_id_accepts_exact_id_or_unique_concept(tmp_path):
    canonical = tmp_path / "canonical.json"
    _write_canonical(canonical)
    active_rules = feedback.load_active_rules(canonical)

    assert feedback.resolve_rule_id("active-1", active_rules) == "active-1"
    assert feedback.resolve_rule_id("asset_life", active_rules) == "active-1"


def test_resolve_rule_id_rejects_unknown_or_inactive_rules(tmp_path):
    canonical = tmp_path / "canonical.json"
    _write_canonical(canonical)
    active_rules = feedback.load_active_rules(canonical)

    with pytest.raises(ValueError):
        feedback.resolve_rule_id("demo_ops", active_rules)
    with pytest.raises(ValueError):
        feedback.resolve_rule_id("missing", active_rules)


def test_main_dry_run_validates_without_writing(tmp_path, capsys):
    canonical = tmp_path / "canonical.json"
    feedback_jsonl = tmp_path / "feedback.jsonl"
    _write_canonical(canonical)

    result = feedback.main(
        [
            "asset_life",
            "helped",
            "--case-id",
            "case-001",
            "--used-at",
            "2026-07-25T10:00:00",
            "--canonical-json",
            str(canonical),
            "--feedback-jsonl",
            str(feedback_jsonl),
            "--dry-run",
        ]
    )

    assert result == 0
    assert not feedback_jsonl.exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["rule_id"] == "active-1"
    assert printed["outcome"] == "helped"


def test_main_records_only_active_rule(tmp_path):
    canonical = tmp_path / "canonical.json"
    feedback_jsonl = tmp_path / "feedback.jsonl"
    _write_canonical(canonical)

    result = feedback.main(
        [
            "active-1",
            "used",
            "--canonical-json",
            str(canonical),
            "--feedback-jsonl",
            str(feedback_jsonl),
        ]
    )

    assert result == 0
    rows = [json.loads(line) for line in feedback_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [row["rule_id"] for row in rows] == ["active-1"]
