import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyze_pipeline_health.py"
_spec = importlib.util.spec_from_file_location("analyze_pipeline_health", _SCRIPT)
health_mod = importlib.util.module_from_spec(_spec)
sys.modules["analyze_pipeline_health"] = health_mod
_spec.loader.exec_module(health_mod)


def test_aggregate_tracks_latest_exit_code():
    entries = [
        {"ts": "2026-07-14T19:00:00Z", "run_date": "20260715", "step": "sync_cloudsql_to_obsidian", "exit_code": 1},
        {"ts": "2026-07-15T01:00:00Z", "run_date": "20260715", "step": "sync_cloudsql_to_obsidian", "exit_code": 0},
    ]

    counts = health_mod.aggregate(entries)

    assert counts["sync_cloudsql_to_obsidian"]["bad"] == 1
    assert counts["sync_cloudsql_to_obsidian"]["good"] == 1
    assert counts["sync_cloudsql_to_obsidian"]["latest_exit_code"] == 0


def test_disabled_step_skip_log_clears_stale_failure_alert():
    """REV-028a 回帰: 廃止前の失敗ログが残っていても、無効化後に記録される
    skip(exit 0) エントリで latest_exit_code=0 となり、過去検出が解決される。"""
    entries = [
        {"ts": "2026-07-18T19:00:00Z", "run_date": "20260719", "step": "sync_cloudsql_to_obsidian", "exit_code": 1},
        {"ts": "2026-07-19T19:00:00Z", "run_date": "20260720", "step": "sync_cloudsql_to_obsidian", "exit_code": 1},
        # 既定無効化後に run_daily_improvement_core.sh が記録する「意図的スキップ＝健全」
        {"ts": "2026-07-21T19:00:00Z", "run_date": "20260721", "step": "sync_cloudsql_to_obsidian", "exit_code": 0},
    ]
    counts = health_mod.aggregate(entries)
    assert counts["sync_cloudsql_to_obsidian"]["latest_exit_code"] == 0

    ledger = [
        {
            "rev_id": "REV-028a",
            "status": "pending_review",
            "pending_review": True,
            "source": "analyze_pipeline_health",
            "description": "[パイプライン自動検出] sync_cloudsql_to_obsidian が過去7日で失敗率88%",
        }
    ]
    resolved = health_mod.resolve_recovered_entries(ledger, counts, "2026-07-21T19:05:00Z")
    assert resolved == 1
    assert ledger[0]["status"] == "stale_resolved"
    assert ledger[0]["pending_review"] is False


def test_resolve_recovered_entries_marks_active_alert_stale_resolved():
    ledger = [
        {
            "rev_id": "REV-026a",
            "status": "pending_review",
            "pending_review": True,
            "source": "analyze_pipeline_health",
            "description": "[パイプライン自動検出] sync_cloudsql_to_obsidian が過去7日で失敗率100%",
        }
    ]
    counts = {
        "sync_cloudsql_to_obsidian": {
            "latest_exit_code": 0,
        }
    }

    resolved = health_mod.resolve_recovered_entries(ledger, counts, "2026-07-15T01:00:00Z")

    assert resolved == 1
    assert ledger[0]["status"] == "stale_resolved"
    assert ledger[0]["pending_review"] is False
