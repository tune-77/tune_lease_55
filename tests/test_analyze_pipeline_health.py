import importlib.util
import sys
import json
from datetime import datetime, timezone
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


def test_main_persists_recovered_entries_even_without_new_penalties(tmp_path, monkeypatch):
    """復旧済み整理は、新規の失敗率超過がない日にも保存される。"""
    log_path = tmp_path / "pipeline_step_log.jsonl"
    ledger_path = tmp_path / "ledger_rules.json"
    now = datetime.now(timezone.utc)
    run_date = now.strftime("%Y%m%d")
    step = "check_obsidian_ops_consistency"

    entries = [
        {"ts": "2026-08-24T19:00:00Z", "run_date": run_date, "step": step, "exit_code": 1},
        {"ts": "2026-08-24T19:05:00Z", "run_date": run_date, "step": step, "exit_code": 0},
        {"ts": "2026-08-24T19:10:00Z", "run_date": run_date, "step": step, "exit_code": 0},
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )
    ledger_path.write_text(
        json.dumps(
            [
                {
                    "rev_id": "REV-303a",
                    "status": "pending_review",
                    "pending_review": True,
                    "source": "analyze_pipeline_health",
                    "description": f"[パイプライン自動検出] {step} が過去7日で失敗率50%",
                }
            ],
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(health_mod, "LOG_FILE", log_path)
    monkeypatch.setattr(health_mod, "LEDGER_FILE", ledger_path)

    health_mod.main()

    [updated] = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert updated["status"] == "stale_resolved"
    assert updated["pending_review"] is False
    assert updated["resolved_at"]
