"""GET /api/shion/action-ledger/summary（対話室UI・監査用、read-only）。"""

import json

import api.routers.improvement as improvement
from api import shion_action_ledger as ledger


def test_returns_cached_report_when_days_matches(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    cached = {"days": 7, "total": 3, "by_action": {}, "pending_approval_count": 1}
    (reports_dir / "agent_action_ledger_latest.json").write_text(
        json.dumps(cached), encoding="utf-8"
    )
    monkeypatch.setattr(improvement, "_candidate_report_dirs", lambda: [reports_dir])

    result = improvement.get_shion_action_ledger_summary(days=7)

    assert result == cached


def test_falls_back_to_live_summary_when_days_differ(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "agent_action_ledger_latest.json").write_text(
        json.dumps({"days": 7, "total": 99}), encoding="utf-8"
    )
    monkeypatch.setattr(improvement, "_candidate_report_dirs", lambda: [reports_dir])

    ledger_path = tmp_path / "shion_action_ledger.jsonl"
    monkeypatch.setattr(ledger, "LEDGER_PATH", ledger_path)
    ledger.log_action("daily_report", summary="test", path=ledger_path)

    result = improvement.get_shion_action_ledger_summary(days=30)

    assert result["days"] == 30
    assert result["total"] == 1


def test_falls_back_to_live_summary_when_report_missing(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(improvement, "_candidate_report_dirs", lambda: [reports_dir])

    ledger_path = tmp_path / "shion_action_ledger.jsonl"
    monkeypatch.setattr(ledger, "LEDGER_PATH", ledger_path)

    result = improvement.get_shion_action_ledger_summary(days=7)

    assert result["total"] == 0
    assert result["pending_approval_count"] == 0
