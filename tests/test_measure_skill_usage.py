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
