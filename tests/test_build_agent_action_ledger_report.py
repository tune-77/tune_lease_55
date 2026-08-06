import datetime as dt
from pathlib import Path

from api import shion_action_ledger as ledger
from scripts import build_agent_action_ledger_report as report_mod


def test_build_summary_counts_actions_and_pending_approval(tmp_path: Path):
    path = tmp_path / "shion_action_ledger.jsonl"
    ledger.log_action("daily_report", summary="重大異常なし", path=path)
    ledger.log_action(
        "codex_request_drafted",
        summary="Codexキュー生成",
        requires_user_approval=True,
        target="reports/codex_auto_queue_20260806.json",
        path=path,
    )
    ledger.log_action(
        "user_decision_recorded",
        summary="承認済み",
        requires_user_approval=True,
        user_approved=True,
        path=path,
    )

    entries = ledger.read_actions(path=path)
    summary = report_mod.build_summary(entries, days=7)

    assert summary["total"] == 3
    assert summary["by_action"]["daily_report"] == 1
    assert summary["pending_approval_count"] == 1
    assert summary["pending_approval"][0]["action"] == "codex_request_drafted"


def test_build_summary_excludes_old_entries(tmp_path: Path):
    path = tmp_path / "shion_action_ledger.jsonl"
    old_entry = ledger.log_action("daily_report", summary="古い", path=path)
    old_entry["timestamp"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).isoformat(timespec="seconds")

    summary = report_mod.build_summary([old_entry], days=7)
    assert summary["total"] == 0


def test_write_report_creates_markdown(tmp_path: Path):
    path = tmp_path / "shion_action_ledger.jsonl"
    ledger.log_action("system_watch", summary="正常", path=path)
    entries = ledger.read_actions(path=path)
    summary = report_mod.build_summary(entries, days=7)

    md_path = tmp_path / "agent_action_ledger_latest.md"
    report_mod.write_report(summary, md_path)

    content = md_path.read_text(encoding="utf-8")
    assert "紫苑 Agent Action Ledger" in content
    assert "system_watch: 1" in content
