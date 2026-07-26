import json
from pathlib import Path

from scripts import obsidian_memory_effectiveness_report as memeff


def _write_note(path: Path, *, title: str, note_type: str = "lease_rule") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "---",
                f"type: {note_type}",
                f"title: {title}",
                "domain: credit",
                "tags: [補助金, 条件付き承認]",
                "source: user_judgment",
                "confidence: medium",
                "status: active",
                "updated: 2026-07-25",
                "---",
                "",
                f"# {title}",
                "",
                "## 要点",
                "",
                "補助金案件では未採択時の返済原資を見る。",
            ]
        ),
        encoding="utf-8",
    )


def test_build_report_marks_recalled_from_eval_and_index(tmp_path, monkeypatch):
    monkeypatch.setattr(memeff, "PROJECT_ROOT", tmp_path)
    note = tmp_path / "knowledge_base" / "okf_lease_concepts" / "rules" / "subsidy.md"
    _write_note(note, title="補助金前提案件")
    rel = "knowledge_base/okf_lease_concepts/rules/subsidy.md"

    payload = memeff.build_report(
        knowledge_dir=tmp_path / "knowledge_base" / "okf_lease_concepts",
        memory_index={"items": [{"source_path": rel, "last_used_at": ""}]},
        feedback_rows=[],
        okf_eval={"cases": [{"id": "c1", "passed": True, "rank": 1, "expected_path_any": [rel]}]},
        target_date="2026-07-25",
    )

    record = payload["records"][0]
    assert payload["summary"]["recalled"] == 1
    assert record["state"] == "recalled"
    assert record["recalled_count"] == 1
    assert record["index_records"] == 1


def test_feedback_helped_promotes_to_validated(tmp_path, monkeypatch):
    monkeypatch.setattr(memeff, "PROJECT_ROOT", tmp_path)
    note = tmp_path / "knowledge_base" / "okf_lease_concepts" / "rules" / "subsidy.md"
    _write_note(note, title="補助金前提案件")
    rel = "knowledge_base/okf_lease_concepts/rules/subsidy.md"

    payload = memeff.build_report(
        knowledge_dir=tmp_path / "knowledge_base" / "okf_lease_concepts",
        memory_index={"items": []},
        feedback_rows=[{"rule_id": rel, "outcome": "helped", "note": "補助金前提案件で使えた"}],
        okf_eval={},
        target_date="2026-07-25",
    )

    record = payload["records"][0]
    assert record["state"] == "validated"
    assert record["human_accept_count"] == 1
    assert record["effectiveness_score"] > 0


def test_challenged_more_than_helped_marks_noisy(tmp_path, monkeypatch):
    monkeypatch.setattr(memeff, "PROJECT_ROOT", tmp_path)
    note = tmp_path / "knowledge_base" / "okf_lease_concepts" / "rules" / "subsidy.md"
    _write_note(note, title="補助金前提案件")
    rel = "knowledge_base/okf_lease_concepts/rules/subsidy.md"

    payload = memeff.build_report(
        knowledge_dir=tmp_path / "knowledge_base" / "okf_lease_concepts",
        memory_index={"items": []},
        feedback_rows=[{"rule_id": rel, "outcome": "challenged"}],
        okf_eval={"cases": [{"id": "c1", "passed": True, "rank": 1, "expected_path_any": [rel]}]},
        target_date="2026-07-25",
    )

    assert payload["records"][0]["state"] == "noisy"


def test_markdown_and_state_jsonl_are_stable(tmp_path):
    payload = {
        "date": "2026-07-25",
        "mode": "read_only_observation",
        "guardrail": "no changes",
        "knowledge_dir": "knowledge_base/okf_lease_concepts",
        "summary": {"total": 1, "dormant": 0, "recalled": 1, "used": 0, "validated": 0, "noisy": 0},
        "records": [
            {
                "obsidian_ref": "knowledge_base/okf_lease_concepts/rules/a.md",
                "title": "A",
                "memory_type": "lease_rule",
                "domain": "credit",
                "state": "recalled",
                "effectiveness_score": 10.0,
                "recalled_count": 1,
                "used_count": 0,
                "human_accept_count": 0,
                "challenged_count": 0,
                "rejected_count": 0,
                "next_action": "確認する。",
            }
        ],
    }
    markdown = memeff.build_markdown(payload)
    assert "# Obsidian Memory Effectiveness" in markdown
    assert "State: `recalled`" in markdown

    state_path = tmp_path / "state.jsonl"
    memeff.write_state_jsonl(state_path, payload)
    rows = [json.loads(line) for line in state_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["date"] == "2026-07-25"
    assert rows[0]["state"] == "recalled"
