"""@AI_Insight_Evolved ノートの停止フラグ（REV-491）。

9/10 以降「3. 結論」が毎日同じフォールバック定型文で、読み手（extract_wiki_vault_insights）も
「新規改善案なし」を返し続けていたため、既定で書かない。朝報そのものは従来どおり書く。
"""
from pathlib import Path

import pytest

from scripts import aurion_core_daily as acd


@pytest.fixture
def morning(tmp_path, monkeypatch):
    calls: dict[str, int] = {"insight": 0, "web": 0}
    report = tmp_path / "report.md"

    def _insight(*_args, **_kwargs) -> Path:
        calls["insight"] += 1
        return tmp_path / "@AI_Insight_Evolved_2026-10-07.md"

    def _web(*_args, **_kwargs):
        calls["web"] += 1
        return {"findings": []}

    monkeypatch.setattr(acd, "_mkdirs", lambda: None)
    monkeypatch.setattr(acd, "read_latest_state", lambda: {"sync": {}, "errors": []})
    monkeypatch.setattr(acd, "audit_db", lambda: {"counts": []})
    monkeypatch.setattr(acd, "collect_recent_improvements", lambda: {})
    monkeypatch.setattr(acd, "read_codex_auto_queue", lambda: {})
    monkeypatch.setattr(acd, "collect_improvement_declaration_gaps", lambda: {"count": 0})
    monkeypatch.setattr(acd, "write_morning_report", lambda *a, **k: report)
    monkeypatch.setattr(acd, "write_evolved_insight", _insight)
    monkeypatch.setattr(acd, "web_tactical_search", _web)
    monkeypatch.setattr(acd, "cross_reasoning_loop", lambda *a, **k: {"conclusions": []})
    monkeypatch.setattr(acd, "notify", lambda *a, **k: None)
    monkeypatch.setattr(acd, "_send_slack", lambda *a, **k: None)
    return calls


def test_insight_is_off_by_default(morning, monkeypatch):
    monkeypatch.delenv("AURION_EVOLVED_INSIGHT_ENABLED", raising=False)

    assert acd.run_morning_report() == 0

    assert morning == {"insight": 0, "web": 0}


def test_insight_can_be_turned_back_on(morning, monkeypatch):
    monkeypatch.setenv("AURION_EVOLVED_INSIGHT_ENABLED", "1")

    assert acd.run_morning_report() == 0

    assert morning["insight"] == 1


@pytest.mark.parametrize("value,expected", [("", False), ("0", False), ("1", True), ("on", True)])
def test_flag_values(monkeypatch, value, expected):
    monkeypatch.setenv("AURION_EVOLVED_INSIGHT_ENABLED", value)

    assert acd.evolved_insight_enabled() is expected
