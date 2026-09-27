"""週次ゲートで走るステップが失敗検知に乗ることを担保する（REV-420）。

背景: 詳細サイドカー6本は DETAILED_SIDECAR_REPORT_FREQUENCY=weekly（既定で月曜のみ）
でゲートされたため、7日ウィンドウでは実測が最大1件しか溜まらず MIN_TOTAL_RUNS=3 を
永久に満たせなかった。毎週必ず失敗しても penalty_steps に乗らない状態だった。
"""

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyze_pipeline_health.py"
_spec = importlib.util.spec_from_file_location("analyze_pipeline_health", _SCRIPT)
health_mod = importlib.util.module_from_spec(_spec)
sys.modules["analyze_pipeline_health"] = health_mod
_spec.loader.exec_module(health_mod)

WEEKLY_STEP = "build_instruction_debt_report"
DAILY_STEP = "sync_cloudsql_to_obsidian"


def _entry(step, days_ago, exit_code):
    moment = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {
        "ts": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_date": moment.strftime("%Y%m%d"),
        "step": step,
        "exit_code": exit_code,
    }


def test_weekly_steps_are_disjoint_from_the_daily_graph_step():
    """build_judgment_asset_graph は週次ゲート内だが else 側が同名で exit 0 を記録する
    （run_daily_improvement_post.sh:375,378）ため毎日ログが出る。週次扱いにしない。"""
    assert "build_judgment_asset_graph" not in health_mod.WEEKLY_STEPS
    assert WEEKLY_STEP in health_mod.WEEKLY_STEPS
    assert health_mod.MAX_LOOKBACK_DAYS == health_mod.WEEKLY_LOOKBACK_DAYS


def test_step_thresholds_differ_between_weekly_and_daily_steps():
    assert health_mod.step_lookback_days(WEEKLY_STEP) == health_mod.WEEKLY_LOOKBACK_DAYS
    assert health_mod.step_lookback_days(DAILY_STEP) == health_mod.LOOKBACK_DAYS
    assert health_mod.step_min_total_runs(WEEKLY_STEP) == health_mod.WEEKLY_MIN_TOTAL_RUNS
    assert health_mod.step_min_total_runs(DAILY_STEP) == health_mod.MIN_TOTAL_RUNS
    assert health_mod.step_auto_fix_min_days(WEEKLY_STEP) == health_mod.WEEKLY_AUTO_FIX_MIN_FAILURE_DAYS
    assert health_mod.step_auto_fix_min_days(DAILY_STEP) == health_mod.AUTO_FIX_MIN_FAILURE_DAYS


def test_filter_to_step_windows_keeps_weekly_history_but_trims_daily():
    """二段フィルタの要点: 読み込み窓を広げても通常ステップは7日で切られる。"""
    entries = [
        _entry(WEEKLY_STEP, 21, 1),
        _entry(WEEKLY_STEP, 14, 1),
        _entry(DAILY_STEP, 21, 1),
        _entry(DAILY_STEP, 1, 1),
    ]

    kept = health_mod.filter_to_step_windows(entries)

    kept_steps = [(e["step"], e["run_date"]) for e in kept]
    assert len(kept) == 3
    assert (DAILY_STEP, entries[2]["run_date"]) not in kept_steps


def _run_main(tmp_path, monkeypatch, entries, ledger=None):
    log_path = tmp_path / "pipeline_step_log.jsonl"
    ledger_path = tmp_path / "ledger_rules.json"
    log_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
        encoding="utf-8",
    )
    ledger_path.write_text(json.dumps(ledger or [], ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(health_mod, "LOG_FILE", log_path)
    monkeypatch.setattr(health_mod, "LEDGER_FILE", ledger_path)
    health_mod.main()
    return json.loads(ledger_path.read_text(encoding="utf-8"))


def test_weekly_step_failing_every_week_is_detected(tmp_path, monkeypatch):
    """4週で2回しか走らない週次ステップでも、両方失敗すれば起票される。
    修正前は MIN_TOTAL_RUNS=3 に届かず永久に検知できなかった。"""
    entries = [_entry(WEEKLY_STEP, 21, 1), _entry(WEEKLY_STEP, 14, 1)]

    ledger = _run_main(tmp_path, monkeypatch, entries)

    assert len(ledger) == 1
    assert WEEKLY_STEP in ledger[0]["description"]
    assert f"過去{health_mod.WEEKLY_LOOKBACK_DAYS}日" in ledger[0]["description"]
    assert ledger[0]["status"] == "pending_review"


def test_daily_step_still_requires_three_runs_inside_seven_days(tmp_path, monkeypatch):
    """通常ステップの閾値は変えない。7日窓に2件しかなければ起票しない。"""
    entries = [_entry(DAILY_STEP, 2, 1), _entry(DAILY_STEP, 1, 1), _entry(DAILY_STEP, 21, 1)]

    ledger = _run_main(tmp_path, monkeypatch, entries)

    assert ledger == []


def test_weekly_step_auto_fix_needs_two_failing_weeks_and_a_rule(tmp_path, monkeypatch):
    """週次ステップは2週連続失敗＋patch_json型ルールがあれば自動修正を許可する。"""
    entries = [_entry(WEEKLY_STEP, 21, 1), _entry(WEEKLY_STEP, 14, 1)]
    existing = [
        {
            "rev_id": "REV-401",
            "type": "patch_json",
            "status": "resolved",
            "description": f"{WEEKLY_STEP} の設定値を修正",
        }
    ]

    ledger = _run_main(tmp_path, monkeypatch, entries, existing)

    created = [e for e in ledger if e.get("source") == "analyze_pipeline_health"]
    assert len(created) == 1
    assert created[0]["auto_fix_allowed"] is True
    assert created[0]["type"] == "patch_json"


def test_negative_control_same_logs_are_invisible_without_the_weekly_window(tmp_path, monkeypatch):
    """逆向きの確認: WEEKLY_STEPS を空にすると同じログで検知できなくなる。
    このテストが通ることで、検知が週次窓の導入によるものだと確認できる。"""
    monkeypatch.setattr(health_mod, "WEEKLY_STEPS", frozenset())
    entries = [_entry(WEEKLY_STEP, 21, 1), _entry(WEEKLY_STEP, 14, 1)]

    ledger = _run_main(tmp_path, monkeypatch, entries)

    assert ledger == []
