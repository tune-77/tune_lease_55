from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


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
            {"attempts": 1, "retried": False, "outcome": "ok"},
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
            {"attempts": 1, "retried": False, "outcome": "ok"},
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


def test_run_reports_grounding_telemetry(tmp_path, monkeypatch):
    """接地検索の試行回数をレポートへ載せる。retryの二重課金を後から数えるため。"""
    vault = tmp_path / "Obsidian Vault"
    monkeypatch.setattr(
        research,
        "research_topic",
        lambda topic: (
            "## 結論\n- 検収と所有権を確認する。",
            [{"title": "Source", "url": "https://example.com"}],
            "gemini-test",
            {
                "attempts": 2,
                "retried": True,
                "outcome": "ok",
                "per_attempt": [
                    {"attempt": 1, "text_chars": 900, "source_count": 0},
                    {"attempt": 2, "text_chars": 1400, "source_count": 12},
                ],
            },
        ),
    )
    monkeypatch.setattr(research, "_index_note", lambda path: None)
    monkeypatch.setattr(research, "_refresh_judgment_asset_candidates", lambda v, o: {"candidates": 0})

    result = research.run(vault, research.DEFAULT_OUTPUT_DIR, requested_topic="contract-ownership")

    assert result["grounding"]["retried"] is True
    assert result["grounding"]["per_attempt"][0]["source_count"] == 0, (
        "1回目が接地ゼロだった事実が消えると、最も高い課金が見えないままになる"
    )


def test_emit_grounding_telemetry_writes_one_greppable_line(capsys):
    research._emit_grounding_telemetry(
        research.TOPICS[0],
        {"attempts": 2, "retried": True, "per_attempt": [], "outcome": "unknown"},
        outcome="no_sources",
    )

    err = capsys.readouterr().err.strip()
    assert err.startswith("[autoresearch-grounding] ")
    assert '"outcome":"no_sources"' in err
    assert len(err.splitlines()) == 1, "1行でないとログ集計時に壊れる"


def test_research_models_default_to_grounding_capable_models(monkeypatch):
    monkeypatch.delenv("GEMINI_RESEARCH_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_RESEARCH_FALLBACK_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")  # チャット既定に引きずられない
    assert research.research_models() == ("gemini-3.5-flash", "gemini-3.1-pro-preview")
    monkeypatch.setenv("GEMINI_RESEARCH_MODEL", "x-model")
    assert research.research_models()[0] == "x-model"


def test_retry_uses_fallback_model_and_logs_tokens(monkeypatch, capsys):
    """1回目が接地なしなら上位モデルで再試行し、両方だめなら保存しない（モデルとトークン数を記録）。"""
    import types as pytypes

    import api.vertex_agent_search as vas
    from google import genai

    monkeypatch.setattr(vas, "get_config", lambda: pytypes.SimpleNamespace(enabled=True, project_id="p"))
    monkeypatch.setattr(vas, "_access_token", lambda: "t")
    monkeypatch.delenv("GEMINI_RESEARCH_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_RESEARCH_FALLBACK_MODEL", raising=False)
    called: list[str] = []

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            called.append(model)
            usage = pytypes.SimpleNamespace(prompt_token_count=10, candidates_token_count=20, thoughts_token_count=5, total_token_count=35)
            candidate = pytypes.SimpleNamespace(grounding_metadata=None)
            return pytypes.SimpleNamespace(text="記憶だけの回答", candidates=[candidate], usage_metadata=usage)

    monkeypatch.setattr(genai, "Client", lambda **kw: pytypes.SimpleNamespace(models=FakeModels()))
    topic = research.choose_topic(research.Path("/nonexistent"), "建設業の農業参入")
    with pytest.raises(RuntimeError, match="no verifiable source URLs"):
        research.research_topic(topic)
    assert called == ["gemini-3.5-flash", "gemini-3.1-pro-preview"]
    line = next(row for row in capsys.readouterr().err.splitlines() if "[autoresearch-grounding]" in row)
    assert '"model":"gemini-3.1-pro-preview"' in line and '"total":35' in line


def test_research_organ_explains_missing_sources(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    from api.routers import vault_hub

    def fail(*args):
        raise RuntimeError("Gemini research returned no verifiable source URLs; note was not saved")

    monkeypatch.setattr(research, "run", fail)
    monkeypatch.setattr(vault_hub, "_research_organ_vault_path", lambda: Path("/tmp/vault"))  # CI には Vault が無い
    monkeypatch.setitem(sys.modules, "scripts.auto_research_lease_judgment", research)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(vault_hub.run_research_organ(vault_hub.ResearchOrganRunRequest(topic="建設業の農業参入")))
    assert caught.value.status_code == 422
    assert "テーマを具体的に" in caught.value.detail and "もう一度" in caught.value.detail
