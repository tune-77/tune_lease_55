from __future__ import annotations

import json
from pathlib import Path

from scripts import run_pipeline_auto_recovery as recovery


def test_checked_in_ledger_seeds_every_allowlisted_recovery_recipe() -> None:
    root = Path(__file__).resolve().parents[1]
    ledger_path = root / "api" / "rule_engine" / "ledger_rules.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    catalog = recovery.build_recovery_catalog(ledger)
    eligible_steps = {
        str(recipe.get("step") or "")
        for recipe in catalog["recipes"]
        if int(recipe.get("resolved_incident_count") or 0) > 0
    }

    assert eligible_steps == set(recovery.RECOVERY_RECIPES)


def test_catalog_uses_resolved_incidents_as_evidence_only() -> None:
    ledger = [
        {
            "rev_id": "REV-418a",
            "category": "pipeline_fix",
            "status": "stale_resolved",
            "description": "[パイプライン自動検出] check_shion_memory_health が失敗",
            "resolution_reason": "手動で基準移行して復旧",
            "resolved_at": "2026-09-27T21:08:45Z",
        },
        {
            "rev_id": "REV-X",
            "category": "pipeline_fix",
            "status": "pending_review",
            "description": "check_shion_memory_health が失敗",
            "resolution_reason": "rm -rf /",
        },
    ]

    catalog = recovery.build_recovery_catalog(ledger)

    [recipe] = [r for r in catalog["recipes"] if r["step"] == "check_shion_memory_health"]
    assert recipe["resolved_incident_count"] == 1
    assert recipe["learned_from"][0]["rev_id"] == "REV-418a"
    assert "rm -rf" not in json.dumps(catalog, ensure_ascii=False)


def test_plan_only_selects_allowlisted_latest_failures() -> None:
    latest = {
        "check_shion_memory_health": {"ts": "2026-09-28T01:00:00Z", "exit_code": 1},
        "dangerous_unknown_step": {"ts": "2026-09-28T01:01:00Z", "exit_code": 1},
        "build_agent_worklog_digest": {"ts": "2026-09-28T01:02:00Z", "exit_code": 0},
    }

    plans = recovery.plan_recoveries(latest, {"attempts": {}}, "20260928")

    assert [plan["step"] for plan in plans] == ["check_shion_memory_health"]


def test_plan_does_not_retry_same_step_on_same_run_date() -> None:
    latest = {"check_shion_memory_health": {"ts": "2026-09-28T01:00:00Z", "exit_code": 1}}
    state = {"attempts": {"20260928:check_shion_memory_health": {"success": False}}}

    assert recovery.plan_recoveries(latest, state, "20260928") == []


def test_plan_requires_resolved_incident_evidence() -> None:
    latest = {"check_shion_memory_health": {"ts": "2026-09-28T01:00:00Z", "exit_code": 1}}

    plans = recovery.plan_recoveries(
        latest,
        {"attempts": {}},
        "20260928",
        eligible_steps=set(),
    )

    assert plans == []


def test_plan_rotates_away_from_more_frequently_attempted_recipe() -> None:
    latest = {
        "check_shion_memory_health": {"ts": "2026-09-28T01:00:00Z", "exit_code": 1},
        "build_agent_worklog_digest": {"ts": "2026-09-28T01:01:00Z", "exit_code": 1},
    }
    state = {"attempts": {"20260927:check_shion_memory_health": {"success": False}}}

    plans = recovery.plan_recoveries(
        latest,
        state,
        "20260928",
        eligible_steps=set(latest),
    )

    assert [plan["step"] for plan in plans] == [
        "build_agent_worklog_digest",
        "check_shion_memory_health",
    ]


def test_memory_health_recipe_migrates_legacy_layer_baseline(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    records = [
        {
            "id": f"long-{index}",
            "memory_type": "factual_memory",
            "memory_layer": "long_term",
            "status": "active",
        }
        for index in range(3)
    ] + [
        {
            "id": "mid-1",
            "memory_type": "dialogue_memory",
            "memory_layer": "mid_term",
            "status": "active",
        }
    ]
    (data / "shion_memory_index.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    (data / "shion_memory_health_state.json").write_text(
        json.dumps({"total": 5, "by_type": {}, "by_status": {}}),
        encoding="utf-8",
    )
    (data / "shion_memory_index_previous_summary.json").write_text(
        json.dumps(
            {
                "total": 5,
                "by_type": {"dialogue_memory": 2, "factual_memory": 3},
                "by_status": {"active": 5},
                "by_layer": {"long_term": 3, "mid_term": 2},
            }
        ),
        encoding="utf-8",
    )

    success, detail = recovery._recover_memory_health(tmp_path)

    state = json.loads((data / "shion_memory_health_state.json").read_text(encoding="utf-8"))
    assert success is True
    assert state["total"] == 4
    assert state["by_layer"] == {"long_term": 3, "mid_term": 1}
    assert "層別基準へ移行" in detail


def test_memory_health_recipe_refuses_large_unexplained_drop(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    records = [
        {
            "id": f"long-{index}",
            "memory_type": "factual_memory",
            "memory_layer": "long_term",
            "status": "active",
        }
        for index in range(3)
    ]
    (data / "shion_memory_index.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    (data / "shion_memory_health_state.json").write_text(
        json.dumps({"total": 100, "by_type": {}, "by_status": {}}),
        encoding="utf-8",
    )

    success, detail = recovery._recover_memory_health(tmp_path)

    state = json.loads((data / "shion_memory_health_state.json").read_text(encoding="utf-8"))
    assert success is False
    assert state["total"] == 100
    assert "自動移行しない" in detail


def test_memory_health_recipe_refuses_mismatched_snapshot(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    records = [
        {
            "id": f"long-{index}",
            "memory_type": "factual_memory",
            "memory_layer": "long_term",
            "status": "active",
        }
        for index in range(4)
    ]
    (data / "shion_memory_index.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    (data / "shion_memory_health_state.json").write_text(
        json.dumps({"total": 5, "by_type": {}, "by_status": {}}),
        encoding="utf-8",
    )
    (data / "shion_memory_index_previous_summary.json").write_text(
        json.dumps({"total": 6, "by_layer": {"long_term": 4, "mid_term": 2}}),
        encoding="utf-8",
    )

    success, detail = recovery._recover_memory_health(tmp_path)

    state = json.loads((data / "shion_memory_health_state.json").read_text(encoding="utf-8"))
    assert success is False
    assert state["total"] == 5
    assert "総件数が一致しない" in detail


def test_memory_health_recipe_preserves_absolute_drop_guard(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    records = [
        {
            "id": f"long-{index}",
            "memory_type": "factual_memory",
            "memory_layer": "long_term",
            "status": "active",
        }
        for index in range(899)
    ]
    (data / "shion_memory_index.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    (data / "shion_memory_health_state.json").write_text(
        json.dumps({"total": 1000, "by_type": {}, "by_status": {}}),
        encoding="utf-8",
    )
    (data / "shion_memory_index_previous_summary.json").write_text(
        json.dumps({"total": 1000, "by_layer": {"long_term": 1000}}),
        encoding="utf-8",
    )

    success, detail = recovery._recover_memory_health(tmp_path)

    state = json.loads((data / "shion_memory_health_state.json").read_text(encoding="utf-8"))
    assert success is False
    assert state["total"] == 1000
    assert "1000 → 899 件（-101）" in detail


def test_execute_plan_appends_success_only_after_verification(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "pipeline.jsonl"
    plan = {
        "step": "build_agent_worklog_digest",
        "recipe_id": "rerun_agent_worklog_digest_after_source_refresh",
    }
    monkeypatch.setitem(recovery.HANDLERS, plan["recipe_id"], lambda root: (True, "verified"))

    result = recovery.execute_plan(plan, root=tmp_path, log_path=log_path, run_date="20260928")

    [entry] = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert result["success"] is True
    assert entry["step"] == "build_agent_worklog_digest"
    assert entry["exit_code"] == 0
    assert entry["recovery"] is True


def test_execute_plan_does_not_claim_success_when_verification_fails(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "pipeline.jsonl"
    plan = {
        "step": "build_agent_worklog_digest",
        "recipe_id": "rerun_agent_worklog_digest_after_source_refresh",
    }
    monkeypatch.setitem(recovery.HANDLERS, plan["recipe_id"], lambda root: (False, "still broken"))

    result = recovery.execute_plan(plan, root=tmp_path, log_path=log_path, run_date="20260928")

    assert result["success"] is False
    assert not log_path.exists()
