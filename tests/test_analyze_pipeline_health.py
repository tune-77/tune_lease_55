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


def test_resolve_recovered_entries_closes_retired_non_blocking_step():
    ledger = [
        {
            "rev_id": "REV-358a",
            "status": "pending_review",
            "pending_review": True,
            "source": "analyze_pipeline_health",
            "description": "[パイプライン自動検出] gist_update が過去7日で失敗率50%",
        }
    ]
    counts = {
        "gist_update": {
            "latest_exit_code": 1,
        }
    }

    resolved = health_mod.resolve_recovered_entries(ledger, counts, "2026-09-10T10:32:50Z")

    assert resolved == 1
    assert ledger[0]["status"] == "stale_resolved"
    assert ledger[0]["pending_review"] is False
    assert "任意配布ステップ" in ledger[0]["resolution_reason"]


def test_resolution_cutoff_picks_latest_matching_resolved_at():
    ledger = [
        {
            "status": "stale_resolved",
            "source": "analyze_pipeline_health",
            "description": "[パイプライン自動検出] eval_shion_memory_recall が過去7日で失敗率75%",
            "resolved_at": "2026-09-10T10:32:50Z",
        },
        {
            "status": "stale_resolved",
            "source": "analyze_pipeline_health",
            "description": "[パイプライン自動検出] eval_shion_memory_recall が過去7日で失敗率69%",
            "resolved_at": "2026-09-16T10:00:00Z",
        },
        {
            "status": "pending_review",
            "source": "analyze_pipeline_health",
            "description": "[パイプライン自動検出] eval_shion_memory_recall が過去7日で失敗率69%",
            "resolved_at": "2026-09-17T00:00:00Z",
        },
    ]

    cutoff = health_mod.resolution_cutoff(ledger, "eval_shion_memory_recall")

    assert cutoff == "2026-09-16T10:00:00Z"


def test_resolution_cutoff_empty_when_no_resolution_recorded():
    ledger = [
        {
            "status": "pending_review",
            "source": "analyze_pipeline_health",
            "description": "[パイプライン自動検出] eval_shion_memory_recall が過去7日で失敗率69%",
        }
    ]

    assert health_mod.resolution_cutoff(ledger, "eval_shion_memory_recall") == ""


def test_aggregate_ignores_entries_before_last_resolution_cutoff():
    """REV-304a→REV-392a再発の再発防止: 解決(resolved_at)より前の失敗ログは、
    コード修正後の再判定に持ち込まない。"""
    entries = [
        {"ts": "2026-09-05T19:00:00Z", "run_date": "20260906", "step": "eval_shion_memory_recall", "exit_code": 1},
        {"ts": "2026-09-06T19:00:00Z", "run_date": "20260907", "step": "eval_shion_memory_recall", "exit_code": 1},
        {"ts": "2026-09-17T19:00:00Z", "run_date": "20260918", "step": "eval_shion_memory_recall", "exit_code": 0},
    ]
    cutoffs = {"eval_shion_memory_recall": "2026-09-16T10:00:00Z"}

    counts = health_mod.aggregate(entries, cutoffs)

    assert counts["eval_shion_memory_recall"]["bad"] == 0
    assert counts["eval_shion_memory_recall"]["good"] == 1


def test_main_does_not_reopen_step_from_pre_resolution_failures(tmp_path, monkeypatch):
    """すでにstale_resolvedなステップは、解決前の失敗ログだけが7日ウィンドウに
    残っていても重複REVを起票しない（REV-304a→REV-392a型の重複再発防止）。"""
    log_path = tmp_path / "pipeline_step_log.jsonl"
    ledger_path = tmp_path / "ledger_rules.json"
    step = "eval_shion_memory_recall"
    run_date = datetime.now(timezone.utc).strftime("%Y%m%d")

    entries = [
        {"ts": "2026-09-05T19:00:00Z", "run_date": run_date, "step": step, "exit_code": 1},
        {"ts": "2026-09-06T19:00:00Z", "run_date": run_date, "step": step, "exit_code": 1},
        {"ts": "2026-09-07T19:00:00Z", "run_date": run_date, "step": step, "exit_code": 1},
    ]
    log_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
        encoding="utf-8",
    )
    ledger_path.write_text(
        json.dumps(
            [
                {
                    "rev_id": "REV-392a",
                    "status": "stale_resolved",
                    "pending_review": False,
                    "source": "analyze_pipeline_health",
                    "description": f"[パイプライン自動検出] {step} が過去7日で失敗率69%（9/13件, 5日失敗）",
                    "resolved_at": "2099-01-01T00:00:00Z",
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

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(ledger) == 1
    assert ledger[0]["rev_id"] == "REV-392a"


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
