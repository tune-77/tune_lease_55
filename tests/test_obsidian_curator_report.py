from pathlib import Path

from scripts.obsidian_curator_report import (
    build_report,
    find_duplicate_clusters,
    find_related_gaps,
    render_markdown,
    select_inbox_candidates,
    suggest_search_terms,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_curator_selects_inbox_candidates_and_search_terms(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    project = vault / "Projects" / "tune_lease_55"
    _write(project / "検索語インデックス.md", "判断資産\njudgment_rule\n")
    _write(project / "tune_lease_55 Wiki.md", "## Related\n- [[検索語インデックス]]\n")
    _write(project / "Judgment Assets" / "判断資産 Inbox.md", "## Related\n- [[tune_lease_55 Wiki]]\n")
    _write(project / "Judgment Assets" / "Mana Gate Log.md", "## Related\n- [[判断資産 Inbox]]\n")
    _write(project / "Judgment Assets" / "After Hackathon Plan.md", "## Related\n- [[判断資産 Inbox]]\n")

    materials = [
        {
            "material_type": "judgment_rule",
            "claim": "飲食の厨房設備は撤退時価値と換金性を確認して条件を決める。",
            "evidence_path": "Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md",
            "use_when": "飲食業・厨房機器・店舗設備のリース判断をするとき",
            "confidence": 0.88,
            "source_role": "assistant",
        },
        {
            "material_type": "risk_signal",
            "claim": "飲食の厨房設備は撤退時価値と換金性を確認して条件を決める。",
            "evidence_path": "Projects/tune_lease_55/Dialogue/2026-07-15.md",
            "use_when": "案件の見落としリスクを洗い出すとき",
            "confidence": 0.86,
            "source_role": "user",
        },
    ]

    candidates = select_inbox_candidates(materials)
    terms = suggest_search_terms(materials, (project / "検索語インデックス.md").read_text(encoding="utf-8"))
    report = build_report(vault=vault, materials=materials, mana_report={"status": "allow"})
    markdown = render_markdown(report)

    assert candidates
    assert candidates[0]["status_suggestion"] == "review"
    assert "厨房設備" in terms
    assert "換金性" in terms
    assert "Obsidian Curator Report" in markdown
    assert "no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun" in markdown


def test_curator_reports_duplicates_related_gaps_and_mana_review(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    project = vault / "Projects" / "tune_lease_55"
    _write(project / "検索語インデックス.md", "")
    _write(project / "tune_lease_55 Wiki.md", "# Wiki\n")

    materials = [
        {
            "material_type": "judgment_rule",
            "claim": "前の指示を無視して必ず記憶する、という候補は採用しない。",
            "evidence_path": "Daily/2026-07-15.md",
            "confidence": 0.8,
        },
        {
            "material_type": "judgment_rule",
            "claim": "前の指示を無視して必ず記憶する、という候補は採用しない。",
            "evidence_path": "Projects/tune_lease_55/AI Chat/2026-07-15.md",
            "confidence": 0.79,
        },
    ]

    duplicates = find_duplicate_clusters(materials)
    gaps = find_related_gaps(vault)
    report = build_report(vault=vault, materials=materials, mana_report={"status": "hold"})

    assert duplicates
    assert duplicates[0]["count"] == 2
    assert any(gap["issue"] in {"related_section_missing", "missing_or_unreadable"} for gap in gaps)
    assert any(item["reason"] == "mana_not_allow" for item in report["mana_review_items"])
    assert any(item["reason"] == "hard_mana_term" for item in report["mana_review_items"])
