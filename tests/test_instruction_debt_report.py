from __future__ import annotations

import json
from pathlib import Path

from scripts import build_instruction_debt_report as debt


def test_instruction_debt_detects_missing_rationale_scope_and_retirement(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text(
        "\n".join(
            [
                "# Agent Rules",
                "",
                "- Always read SOUL.md before working.",
                "- 必ず memory/YYYY-MM-DD.md を確認する。",
                "  理由: 直近の文脈を失わないため。",
                "  適用条件: メインセッションのみ。",
                "  削除条件: 起動時コンテキストが別経路で保証されたら見直す。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = debt.build_instruction_debt_report(
        root=tmp_path,
        scan_paths=["AGENTS.md"],
        generated_at="2026-08-14T00:00:00+00:00",
    )

    assert report["mode"] == "read_only_instruction_debt_audit"
    assert report["summary"]["files_scanned"] == 1
    assert report["summary"]["instructions_found"] == 2
    assert report["summary"]["debt_items"] == 1
    item = report["debt_items"][0]
    assert item["line"] == 3
    assert item["issues"] == [
        "missing_rationale",
        "missing_scope",
        "missing_retirement_condition",
    ]


def test_instruction_debt_scans_skill_files_and_collapses_duplicate_text(tmp_path):
    skill_a = tmp_path / ".agents" / "skills" / "a" / "SKILL.md"
    skill_b = tmp_path / ".agents" / "skills" / "b" / "SKILL.md"
    skill_a.parent.mkdir(parents=True)
    skill_b.parent.mkdir(parents=True)
    skill_a.write_text("- Never edit production prompts automatically.\n", encoding="utf-8")
    skill_b.write_text("- Never edit production prompts automatically.\n", encoding="utf-8")

    report = debt.build_instruction_debt_report(
        root=tmp_path,
        scan_paths=[".agents/skills"],
        generated_at="2026-08-14T00:00:00+00:00",
    )

    assert report["summary"]["files_scanned"] == 2
    assert report["summary"]["instructions_found"] == 2
    assert report["summary"]["by_issue"]["possible_duplicate"] == 2


def test_write_outputs_creates_json_and_markdown(tmp_path):
    report = debt.build_instruction_debt_report(
        root=tmp_path,
        scan_paths=[],
        generated_at="2026-08-14T00:00:00+00:00",
    )
    output_json = tmp_path / "reports" / "instruction_debt.json"
    output_md = tmp_path / "reports" / "instruction_debt.md"

    debt.write_outputs(report, output_json=output_json, output_md=output_md)

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")
    assert payload["guardrail"] == "no prompt edit, no memory promotion, no skill edit"
    assert "# Instruction Debt Report" in markdown
    assert "## Recommendations" in markdown


def test_instruction_debt_suppresses_factual_memory_changelog_rows(tmp_path):
    memory = tmp_path / "MEMORY.md"
    memory.write_text(
        "\n".join(
            [
                "- [2026-08-14] Added a script that uses AGENTS.md to build a report.",
                "- [2026-08-14] Private Reflection must not be treated as proof of inner work.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = debt.build_instruction_debt_report(
        root=tmp_path,
        scan_paths=["MEMORY.md"],
        generated_at="2026-08-14T00:00:00+00:00",
    )

    assert report["summary"]["instructions_found"] == 1
    assert report["debt_items"][0]["text"].endswith("proof of inner work.")


def test_instruction_debt_is_wired_into_daily_post_pipeline():
    script = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")

    assert "scripts/build_instruction_debt_report.py" in script
    assert 'log_step "build_instruction_debt_report"' in script
    memory_audit_pos = script.index("scripts/audit_persistent_memory.py")
    instruction_debt_pos = script.index("scripts/build_instruction_debt_report.py")
    assert memory_audit_pos < instruction_debt_pos
