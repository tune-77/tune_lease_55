from datetime import date

from scripts.build_obsidian_memory_insight_report import (
    build_report,
    build_thinking_cards,
    build_top_candidates,
    collect_candidates,
    load_notes,
    render_markdown,
)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_obsidian_memory_insight_extracts_candidates_and_cards(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    _write(
        vault / "Daily" / "2026-07-14.md",
        (
            "# 2026-07-14\n\n"
            "- Userはハッカソン環境を壊さないで、Obsidian関連の効率化をしてほしい。\n"
            "- 判断資産候補は現場で使える文面か、採用・修正・却下で確認する。\n"
        ),
    )
    _write(
        vault
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Private Reflection"
        / "2026-07-14.md",
        (
            "# 非公開の内省\n\n"
            "- 私の責任: 私はユーザーが何を望んだかを確定する前に、内省らしい言葉へすり替えていた。\n"
            "- 更新する信念: 内省は要求、誤読、次回変更を一組で残す。\n"
        ),
    )
    _write(
        vault / "Projects" / "tune_lease_55" / "Research" / "2026-07-14_research.md",
        "- 業界統計は要確認だが、出典と反証条件を残す必要がある。\n",
    )

    notes = load_notes(vault, end_date=date(2026, 7, 14), days=3)
    candidates = collect_candidates(notes)
    cards = build_thinking_cards(candidates)
    report = build_report(vault, notes, candidates, cards)
    markdown = render_markdown(report)

    types = {item["candidate_type"] for item in candidates}
    assert "user_preference" in types
    assert "judgment_rule" in types
    assert "reflection_update" in types
    assert "research_material" in types
    assert not any(
        item["candidate_type"] == "user_preference" and item["surface"] == "private_reflection"
        for item in candidates
    )
    assert build_top_candidates(candidates)
    assert report["top_candidates"]
    assert cards
    assert "Top 20 Promotion Candidates" in markdown
    assert "Deep Reasoning Cards" in markdown
    assert "inspection_only_no_rag_no_prompt_no_cloudrun_no_obsidian_write" in markdown


def test_obsidian_memory_insight_marks_technical_noise(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    _write(
        vault / "Daily" / "2026-07-14.md",
        "# 2026-07-14\n\n- pytest tests/test_example.py -q passed\n",
    )

    notes = load_notes(vault, end_date=date(2026, 7, 14), days=1)
    candidates = collect_candidates(notes)

    assert candidates
    assert candidates[0]["candidate_type"] == "noise"


def test_obsidian_memory_insight_does_not_promote_assistantish_text_as_user_preference(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    _write(
        vault / "Projects" / "tune_lease_55" / "AI Chat" / "Cloud Run Conversation Log" / "2026-07-14.md",
        "Tune、リース審査における優先順位を整理できます。ご確認してください。\n",
    )

    notes = load_notes(vault, end_date=date(2026, 7, 14), days=1)
    candidates = collect_candidates(notes)

    assert candidates
    assert all(item["candidate_type"] != "user_preference" for item in candidates)
