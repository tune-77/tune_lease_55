import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import measure_skill_usage


def test_codex_skill_file_reads_counts_only_tool_inputs(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "rollout.jsonl"
    rows = [
        {
            "timestamp": "2026-09-06T00:00:00Z",
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "input": "sed -n '1,80p' .agents/skills/git-ship/SKILL.md",
            },
        },
        {
            "timestamp": "2026-09-06T00:01:00Z",
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call_output",
                "output": ".agents/skills/git-ship/SKILL.md",
            },
        },
    ]
    session.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(measure_skill_usage, "CODEX_SESSIONS", tmp_path)

    counts = measure_skill_usage.codex_skill_file_reads(
        ["git-ship"],
        datetime(2026, 9, 5, tzinfo=timezone.utc),
        datetime(2026, 9, 7, tzinfo=timezone.utc),
    )

    assert counts == {"git-ship": 1}


def test_report_retains_per_invocation_audit_fields(tmp_path: Path, monkeypatch) -> None:
    claude_projects = tmp_path / "claude"
    codex_sessions = tmp_path / "codex"
    claude_projects.mkdir()
    codex_sessions.mkdir()
    claude_rows = [
        {"timestamp": "2026-09-06T00:00:00Z", "type": "user", "message": {"content": "Gitship"}},
        {
            "timestamp": "2026-09-06T00:00:01Z",
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "id": "skill-1", "name": "Skill", "input": {"skill": "git-ship"}}]},
        },
        {
            "timestamp": "2026-09-06T00:00:02Z",
            "type": "user",
            "message": {"content": [{"type": "tool_result", "tool_use_id": "skill-1", "is_error": False}]},
        },
        {"timestamp": "2026-09-06T00:00:03Z", "type": "user", "message": {"content": "修正して"}},
    ]
    (claude_projects / "session.jsonl").write_text(
        "\n".join(json.dumps(row) for row in claude_rows), encoding="utf-8"
    )
    monkeypatch.setattr(measure_skill_usage, "CLAUDE_PROJECTS", claude_projects)
    monkeypatch.setattr(measure_skill_usage, "CODEX_SESSIONS", codex_sessions)
    monkeypatch.setattr(measure_skill_usage, "SKILL_ROOTS", ())
    monkeypatch.setattr(measure_skill_usage, "CLAUDE_STATE", tmp_path / "missing.json")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "captured_at": "2026-09-05T00:00:00+00:00",
                "skills": ["git-ship"],
                "claude_usage_counts": {"git-ship": 0},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "usage.md"

    measure_skill_usage.write_report(
        baseline, output, datetime(2026, 9, 7, tzinfo=timezone.utc)
    )

    audit = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert audit["invocations"] == [
        {
            "skill_name": "git-ship",
            "invoked_at": "2026-09-06T00:00:01+00:00",
            "source": "claude",
            "invocation_type": "skill_tool",
            "explicit_or_auto": "explicit",
            "completed": True,
            "user_rework_signal": True,
        }
    ]
