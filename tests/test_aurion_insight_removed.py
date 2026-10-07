"""@AI_Insight_Evolved の結論・Web抜粋・待機の廃止（REV-494）。

結論は 9/10 から毎日同じフォールバック定型文、Web抜粋も結論とこのノートでしか使われていなかった。
midnight はノート同期・RAG 再構築・DB 監査だけを行い、朝報は Insight を書かずリンクもしない。
"""
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import aurion_core_daily as acd


def test_removed_functions_are_gone():
    for name in (
        "cross_reasoning_loop",
        "_generate_conclusions_with_llm",
        "_hold_step",
        "web_tactical_search",
        "WEB_SOURCES",
        "write_evolved_insight",
    ):
        assert not hasattr(acd, name), name


def test_status_lines_no_longer_mention_web_search():
    assert "WEB TACTICAL SEARCH" not in acd.status_lines({"status": "completed"}, {"status": "completed"})


@pytest.fixture
def midnight(tmp_path, monkeypatch):
    monkeypatch.setenv("AURION_MIN_RUNTIME_SECONDS", "0")
    monkeypatch.setattr(acd, "STATE_DIR", tmp_path)
    monkeypatch.setattr(acd, "LOG_DIR", tmp_path)
    monkeypatch.setattr(acd, "SYNC_ROOT", tmp_path / "sync")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(acd, "reindex_vault_b", lambda: {"status": "completed"})
    monkeypatch.setattr(acd, "audit_db", lambda: {"counts": []})
    monkeypatch.setattr(acd, "collect_recent_improvements", lambda: {"files": []})
    monkeypatch.setattr(acd, "notify", lambda *a, **k: None)
    monkeypatch.setattr(acd, "_send_slack", lambda *a, **k: None)
    return tmp_path


def test_midnight_keeps_sync_rag_and_db_audit_without_reasoning(midnight):
    assert acd.run_midnight() == 0

    state = json.loads(next(midnight.glob("state_*.json")).read_text(encoding="utf-8"))
    assert state["sync"] == {"status": "completed"}
    assert state["vault_b_rag"] == {"status": "completed"}
    assert state["db"] == {"counts": []}
    assert "reasoning" not in state and "web" not in state


def test_morning_report_writes_report_only(tmp_path, monkeypatch):
    report = tmp_path / "@AI_Daily_Report_2026-10-08_0600.md"
    monkeypatch.setattr(acd, "_mkdirs", lambda: None)
    monkeypatch.setattr(acd, "read_latest_state", lambda: {"sync": {}, "errors": []})
    monkeypatch.setattr(acd, "audit_db", lambda: {"counts": []})
    monkeypatch.setattr(acd, "collect_recent_improvements", lambda: {})
    monkeypatch.setattr(acd, "read_codex_auto_queue", lambda: {})
    monkeypatch.setattr(acd, "collect_improvement_declaration_gaps", lambda: {"count": 0})
    monkeypatch.setattr(acd, "write_morning_report", lambda *a, **k: report)
    monkeypatch.setattr(acd, "notify", lambda *a, **k: None)
    sent: list[str] = []
    monkeypatch.setattr(acd, "_send_slack", sent.append)

    assert acd.run_morning_report() == 0

    assert sent and "AURION CORE 朝報" in sent[0]
    assert not list(tmp_path.glob("@AI_Insight_Evolved_*.md"))
