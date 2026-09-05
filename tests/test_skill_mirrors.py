from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("skill_name", ["git-ship", "re-lease-count"])
def test_claude_and_codex_skills_share_one_canonical_source(skill_name: str) -> None:
    shared_skill = ROOT / "shared-ai" / "skills" / skill_name
    claude_skill = ROOT / ".claude" / "skills" / skill_name / "SKILL.md"
    codex_skill = ROOT / ".agents" / "skills" / skill_name / "SKILL.md"

    assert (ROOT / ".claude" / "skills" / skill_name).is_symlink()
    assert (ROOT / ".agents" / "skills" / skill_name).is_symlink()
    assert claude_skill.resolve() == (shared_skill / "SKILL.md").resolve()
    assert codex_skill.resolve() == (shared_skill / "SKILL.md").resolve()
