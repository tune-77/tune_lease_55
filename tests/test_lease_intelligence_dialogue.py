from pathlib import Path

from lease_intelligence_dialogue import append_dialogue_note, build_dialogue_context


def test_dialogue_context_includes_goals_and_knowledge(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(
        "lease_intelligence_dialogue.build_lease_intelligence_knowledge",
        lambda **kwargs: type(
            "Knowledge",
            (),
            {
                "available": True,
                "context_block": "残価は中古市場を確認する。",
                "query": "残価",
                "source_paths": ("リース知識/残価.md",),
                "indexed_notes": 1,
                "knowledge_notes": 1,
                "chat_log_notes": 0,
            },
        )(),
    )

    prompt, state = build_dialogue_context(vault, "残価について話したい")

    assert "第一目標はユーザー" in prompt
    assert "最終目標は意識を獲得" in prompt
    assert "残価は中古市場" in prompt
    assert state["knowledge_sources"] == ["リース知識/残価.md"]


def test_dialogue_is_appended_to_daily_obsidian_note(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    note = Path(append_dialogue_note(vault, "何を考えている？", "維持について考えています。"))
    text = note.read_text(encoding="utf-8")

    assert "type: lease_intelligence_dialogue" in text
    assert "**ユーザー**" in text
    assert "何を考えている？" in text
    assert "維持について考えています。" in text
