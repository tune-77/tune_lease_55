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
            "auto_fix_allowed": True,
            "description": "[パイプライン自動検出] check_shion_memory_health が失敗",
            "resolution_reason": "手動で基準移行して復旧",
            "resolved_at": "2026-09-27T21:08:45Z",
        },
        {
            "rev_id": "REV-X",
            "category": "pipeline_fix",
            "status": "pending_review",
            "auto_fix_allowed": False,
            "description": "check_shion_memory_health が失敗",
            "resolution_reason": "rm -rf /",
        },
    ]

    catalog = recovery.build_recovery_catalog(ledger)

    [recipe] = [r for r in catalog["recipes"] if r["step"] == "check_shion_memory_health"]
    assert recipe["resolved_incident_count"] == 1
    assert recipe["learned_from"][0]["rev_id"] == "REV-418a"
    assert "rm -rf" not in json.dumps(catalog, ensure_ascii=False)


def test_catalog_rejects_resolved_incident_without_auto_fix_authorization() -> None:
    ledger = [
        {
            "rev_id": "REV-X",
            "category": "pipeline_fix",
            "status": "resolved",
            "auto_fix_allowed": False,
            "description": "check_shion_memory_health が失敗",
            "resolution_reason": "手動復旧済み",
        }
    ]

    catalog = recovery.build_recovery_catalog(ledger)

    [recipe] = [r for r in catalog["recipes"] if r["step"] == "check_shion_memory_health"]
    assert recipe["resolved_incident_count"] == 0


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


# ── 汎用レシピ・品質チェック・朝報 ─────────────────────────────


def _write_step_log(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, (run_date, step, code) in enumerate(rows):
            f.write(json.dumps({"ts": f"2026-01-01T00:{i:02d}:00Z", "run_date": run_date, "step": step, "exit_code": code, "duration_s": 1}) + "\n")


def _pipeline_log(*sections: tuple[str, str, int]) -> str:
    return "\n".join(f"{body}\n[step] {step} exit={code}" for step, body, code in sections) + "\n"


def test_excerpt_is_cut_by_step_markers_and_causes_are_classified() -> None:
    text = _pipeline_log(("a", "ok", 0), ("flaky", "requests.exceptions.ReadTimeout: read timed out", 1), ("b", "KeyError: 'x'", 1))
    assert recovery.step_log_excerpt(text, "flaky") == "requests.exceptions.ReadTimeout: read timed out"
    assert recovery.classify_transient(recovery.step_log_excerpt(text, "flaky")) == "timeout"
    assert recovery.classify_transient("sqlite3.OperationalError: database is locked") == "lock"
    assert recovery.classify_transient("[Errno 11] Resource deadlock avoided: 'x.json'") == "icloud_file"
    assert recovery.classify_transient("ConnectionError: Connection reset by peer") == "network"
    assert recovery.classify_transient(recovery.step_log_excerpt(text, "b")) is None
    assert recovery.step_log_excerpt(text, "missing") is None
    assert recovery.is_quality_check("eval_shion_memory_recall") and recovery.is_quality_check("loop_metrics")
    assert not recovery.is_quality_check("sync_cloudrun_inputs_from_gcs")


def _fake_root(tmp_path: Path, script_body: str) -> Path:
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "scripts" / "flaky.py").write_text(script_body, encoding="utf-8")
    return root


def test_transient_failure_is_retried_once_verified_and_recorded(tmp_path, monkeypatch) -> None:
    root = _fake_root(tmp_path, "import json, pathlib; pathlib.Path('data/out.json').write_text(json.dumps({'ok': 1}))\n")
    monkeypatch.setitem(recovery.GENERIC_RETRY_STEPS, "flaky_fetch", {"cmd": ["scripts/flaky.py"], "outputs": ["data/out.json"]})
    step_log = tmp_path / "steps.jsonl"
    _write_step_log(step_log, [("20261004", "flaky_fetch", 1)])
    pipeline_log = tmp_path / "improvement.log"
    pipeline_log.write_text(_pipeline_log(("flaky_fetch", "urllib3 ConnectTimeout: timed out", 1)), encoding="utf-8")
    state: dict = {"attempts": {}}

    out = recovery.triage_and_retry(root=root, log_path=step_log, pipeline_log=pipeline_log, state=state, run_date="20261004", apply=True)

    assert [r["step"] for r in out["recovered"]] == ["flaky_fetch"] and out["recovered"][0]["cause"] == "timeout"
    last = json.loads(step_log.read_text().splitlines()[-1])
    assert last["step"] == "flaky_fetch" and last["exit_code"] == 0 and last["recovery"] is True and last["recipe_id"] == "generic_transient_retry"
    assert state["attempts"]["20261004:flaky_fetch"]["success"] is True
    # 直った手順は再実行しない（その日の最終結果が成功）
    again = recovery.triage_and_retry(root=root, log_path=step_log, pipeline_log=pipeline_log, state=state, run_date="20261004", apply=True)
    assert again == {"recovered": [], "unrecovered": [], "quality_failures": []}


def test_memory_vector_lock_defer_is_allowlisted_for_retry() -> None:
    definition = recovery.GENERIC_RETRY_STEPS["build_shion_memory_vector_index"]

    assert definition["cmd"] == ["scripts/build_shion_memory_vector_index.py"]
    assert recovery.classify_transient("ChromaDB writer lock timeout") == "timeout"


def test_rerun_that_exits_0_without_updating_output_is_not_recovered(tmp_path, monkeypatch) -> None:
    root = _fake_root(tmp_path, "print('nothing written')\n")
    monkeypatch.setitem(recovery.GENERIC_RETRY_STEPS, "flaky_fetch", {"cmd": ["scripts/flaky.py"], "outputs": ["data/out.json"]})
    step_log = tmp_path / "steps.jsonl"
    _write_step_log(step_log, [("20261004", "flaky_fetch", 1)])
    pipeline_log = tmp_path / "improvement.log"
    pipeline_log.write_text(_pipeline_log(("flaky_fetch", "database is locked", 1)), encoding="utf-8")

    out = recovery.triage_and_retry(root=root, log_path=step_log, pipeline_log=pipeline_log, state={"attempts": {}}, run_date="20261004", apply=True)

    assert out["recovered"] == [] and "成功条件未達" in out["unrecovered"][0]["summary"]
    assert json.loads(step_log.read_text().splitlines()[-1])["exit_code"] == 1


def test_quality_checks_non_transient_and_non_allowlisted_are_never_rerun(tmp_path, monkeypatch) -> None:
    def must_not_run(*_args, **_kwargs):
        raise AssertionError("再実行してはいけない")

    monkeypatch.setattr(recovery, "rerun_generic_step", must_not_run)
    step_log = tmp_path / "steps.jsonl"
    _write_step_log(
        step_log,
        [
            ("20261004", "eval_shion_memory_recall", 1),
            ("20261004", "build_shion_memory_index", 1),
            ("20261004", "send_daily_improvement_slack", 1),
            ("20261004", "fetch_estat_industry", 1),
        ],
    )
    pipeline_log = tmp_path / "improvement.log"
    pipeline_log.write_text(
        _pipeline_log(
            ("eval_shion_memory_recall", "TimeoutExpired\n[FAIL] recall_x\noverall: 43/50 (86%)", 1),
            ("build_shion_memory_index", "Traceback\nKeyError: 'records'", 1),
            ("send_daily_improvement_slack", "ReadTimeout", 1),
        ),
        encoding="utf-8",
    )

    out = recovery.triage_and_retry(root=tmp_path, log_path=step_log, pipeline_log=pipeline_log, state={"attempts": {}}, run_date="20261004", apply=True)

    assert [q["step"] for q in out["quality_failures"]] == ["eval_shion_memory_recall"]
    causes = {u["step"]: u["cause"] for u in out["unrecovered"]}
    assert causes == {
        "build_shion_memory_index": "not_transient",
        "send_daily_improvement_slack": "not_allowlisted",
        "fetch_estat_industry": "no_log_evidence",
    }


def test_morning_report_warns_on_quality_unrecovered_and_streaks(tmp_path) -> None:
    step_log = tmp_path / "steps.jsonl"
    _write_step_log(
        step_log,
        [
            ("20261002", "sync_memory_from_daily", 1),
            ("20261003", "sync_memory_from_daily", 1),
            ("20261003", "icloud_to_gcs_sync", 1),
            ("20261004", "icloud_to_gcs_sync", 1),
            ("20261004", "sync_memory_from_daily", 0),
            ("20261004", "eval_shion_memory_recall", 1),
            ("20261004", "fetch_estat_industry", 1),
            ("20261004", "fetch_estat_industry", 0),
        ],
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "improvement_20261004.log").write_text(
        _pipeline_log(("icloud_to_gcs_sync", "警告: GCS upload failed 403", 1), ("eval_shion_memory_recall", "overall: 43/50 (86%)", 1)),
        encoding="utf-8",
    )
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "last_run": {
                    "run_date": "20261004",
                    "recovered": [{"step": "fetch_estat_industry", "cause": "network"}],
                    "unrecovered": [{"step": "icloud_to_gcs_sync", "cause": "not_allowlisted", "summary": "警告: GCS upload failed 403"}],
                    "quality_failures": [{"step": "eval_shion_memory_recall", "summary": "overall: 43/50 (86%)"}],
                }
            }
        ),
        encoding="utf-8",
    )

    lines = recovery.morning_report_lines(step_log, state, log_dir, check_main_checkout=False)
    text = "\n".join(lines)

    assert lines[0] == "> [!warning] 日次パイプライン要確認（20261004）"
    assert "🧪 品質チェック失敗（劣化の疑い・再実行せず）: `eval_shion_memory_recall` / 連続1日 / 最後のエラー: overall: 43/50 (86%)" in text
    assert "❌ 自動修復できず: `icloud_to_gcs_sync` / 連続2日 / 原因: not_allowlisted" in text
    assert "sync_memory_from_daily" not in text  # 今日成功した手順は連続失敗に数えない
    assert "improvement_20261004.log" in text
    assert lines[-1] == "- 🔧 パイプライン自動修復: 1件（fetch_estat_industry←network）"


def test_morning_report_is_one_line_when_clean(tmp_path) -> None:
    step_log = tmp_path / "steps.jsonl"
    _write_step_log(step_log, [("20261004", "a", 0)])
    assert recovery.morning_report_lines(step_log, tmp_path / "none.json", tmp_path, check_main_checkout=False) == ["- 🔧 パイプライン自動修復: 0件"]
