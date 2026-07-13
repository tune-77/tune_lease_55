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


def test_substantive_sections_rejects_youkakunin_only_body():
    weak = "\n\n".join(f"## {title}\n要確認" for title in research._REQUIRED_SECTION_TITLES)

    assert research._required_headings_present(weak)
    assert not research._substantive_sections_present(weak)


def test_substantive_sections_accepts_real_items():
    body = "\n\n".join(
        f"## {title}\n- {title}について、対象業種・設備・時期に接続して確認する具体事項を残す。"
        for title in research._REQUIRED_SECTION_TITLES
    )

    assert research._substantive_sections_present(body)


def test_fallback_decision_body_is_substantive():
    topic = research.TOPICS[0]
    body = research._fallback_decision_body(
        topic,
        "中小企業の資金繰り悪化では、売上回収遅延と短期借入依存が返済余力に影響する。",
        [{"title": "SMRJ", "url": "https://www.smrj.go.jp", "quality": "primary"}],
    )

    assert research._substantive_sections_present(body)


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
    refreshed = []
    monkeypatch.setattr(
        research,
        "_refresh_judgment_asset_candidates",
        lambda vault_arg, output_dir: refreshed.append((vault_arg, output_dir)) or {"candidates": 1},
    )

    result = research.run(vault, research.DEFAULT_OUTPUT_DIR, requested_topic="contract-ownership")

    path = Path(result["path"])
    assert path.exists()
    assert vault in path.parents
    assert "Projects/tune_lease_55/Research/Auto Research" in str(path)
    assert indexed == [path]
    assert refreshed == [(vault, research.DEFAULT_OUTPUT_DIR)]
    assert result["judgment_asset_candidates"]["candidates"] == 1


def test_run_keeps_research_note_when_candidate_refresh_fails(tmp_path, monkeypatch):
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
    monkeypatch.setattr(research, "_index_note", lambda path: None)
    monkeypatch.setattr(
        research,
        "_refresh_judgment_asset_candidates",
        lambda vault_arg, output_dir: (_ for _ in ()).throw(RuntimeError("candidate refresh failed")),
    )

    result = research.run(vault, research.DEFAULT_OUTPUT_DIR, requested_topic="contract-ownership")

    assert Path(result["path"]).exists()
    assert "candidate refresh failed" in result["judgment_asset_candidates"]["error"]
