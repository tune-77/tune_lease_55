from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "auto_research_lease_judgment.py"
_SPEC = importlib.util.spec_from_file_location("auto_research_lease_judgment", _SCRIPT)
assert _SPEC and _SPEC.loader
research = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = research
_SPEC.loader.exec_module(research)


def test_choose_topic_prefers_never_researched_topic(tmp_path):
    output_dir = tmp_path / "Auto Research"
    output_dir.mkdir()
    (output_dir / "2026-06-01_cash-flow.md").write_text(
        "---\ndate: 2026-06-01\nresearch_topic: cash-flow\n---\n",
        encoding="utf-8",
    )

    chosen = research.choose_topic(output_dir)

    assert chosen.key == "residual-value"


def test_build_note_contains_decision_metadata_and_sources():
    topic = research.TOPICS[0]
    note = research.build_note(
        topic,
        "## 結論\n- 資金繰りを確認する。",
        [{"title": "Official source", "url": "https://example.go.jp/source"}],
        "gemini-test",
    )

    assert "knowledge_type: lease-judgment-research" in note
    assert "review_status: needs_human_review" in note
    assert "## 情報源" in note
    assert "https://example.go.jp/source" in note
    assert "自動否決・自動承認には使用しません" in note


def test_required_decision_sections_are_enforced():
    complete = "\n".join(
        [
            "## 結論",
            "## 根拠品質",
            "## 判断に使える確認済み事実",
            "## リース審査への適用",
            "## 担当者が確認する質問",
            "## 承認条件を変える兆候",
            "## 反証・過信してはいけない点",
            "## 更新が必要になる条件",
        ]
    )

    assert research._required_headings_present(complete)
    assert not research._required_headings_present("## 結論")


def test_run_saves_to_normal_vault_research_path(tmp_path, monkeypatch):
    vault = tmp_path / "Obsidian Vault"
    monkeypatch.setattr(
        research,
        "research_topic",
        lambda topic: (
            "## 結論\n- 検収と所有権を確認する。",
            [{"title": "Source", "url": "https://example.com"}],
            "gemini-test",
        ),
    )
    indexed = []
    monkeypatch.setattr(research, "_index_note", lambda path: indexed.append(path))

    result = research.run(vault, research.DEFAULT_OUTPUT_DIR, requested_topic="contract-ownership")

    path = Path(result["path"])
    assert path.exists()
    assert vault in path.parents
    assert "Projects/tune_lease_55/Research/Auto Research" in str(path)
    assert indexed == [path]
