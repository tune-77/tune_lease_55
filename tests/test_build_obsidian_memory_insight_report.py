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
    _write(
        vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Dialogue" / "2026-07-14.md",
        "- 判断資産候補は現場で使える文面か、採用・修正・却下で確認する。\n",
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

    assert candidates == []


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


def test_obsidian_memory_insight_limits_daily_to_user_preferences(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    _write(
        vault / "Daily" / "2026-07-28.md",
        "\n".join(
            [
                "- Userは朝報を普通のニュース要約にしてほしいと修正した。",
                "- 判断資産候補の提示順にbandit-style報酬学習を追加。",
                "- pytest tests/test_x.py passed.",
            ]
        ),
    )

    notes = load_notes(vault, end_date=date(2026, 7, 28), days=1)
    candidates = collect_candidates(notes)

    assert candidates
    assert {item["candidate_type"] for item in candidates} == {"user_preference"}
    assert all(item["surface"] == "daily" for item in candidates)


def test_obsidian_memory_insight_skips_meta_operation_sentences(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    _write(
        vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Private Reflection" / "2026-07-28.md",
        "\n".join(
            [
                "- 観測レポートだけで終わらせず、退屈の原因を1つ選んで小さく変える。",
                "- まだ分からないこと: 品質ゲートで弾かれた理由が次回の実回答で減るか。",
                "- 私の見落とし: Userが求めた確認を先に固定せず、一般論へ逃げた。",
            ]
        ),
    )

    notes = load_notes(vault, end_date=date(2026, 7, 28), days=1)
    candidates = collect_candidates(notes)
    claims = [item["claim"] for item in candidates]

    assert claims == ["私の見落とし: Userが求めた確認を先に固定せず、一般論へ逃げた。"]
