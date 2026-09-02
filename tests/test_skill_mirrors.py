from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("skill_name", ["git-ship", "re-lease-count"])
def test_claude_and_codex_skill_mirrors_match(skill_name: str) -> None:
    claude_skill = ROOT / ".claude" / "skills" / skill_name / "SKILL.md"
    codex_skill = ROOT / ".agents" / "skills" / skill_name / "SKILL.md"

    assert claude_skill.read_bytes() == codex_skill.read_bytes(), (
        f"{skill_name} drifted between .claude and .agents; "
        "update both mirrors in the same change"
    )
