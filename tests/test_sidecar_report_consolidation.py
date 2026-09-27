from pathlib import Path


def test_recursive_self_improvement_runs_only_once_after_post_sync():
    core = Path("scripts/run_daily_improvement_core.sh").read_text(encoding="utf-8")
    post = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")

    assert "scripts/recursive_self_improvement.py" not in core
    assert post.count("scripts/recursive_self_improvement.py") == 1
    sync_pos = post.index("sync_improvement_reports.py", post.index("scripts/check_ledger_consistency.py"))
    recursive_pos = post.index("scripts/recursive_self_improvement.py")
    assert sync_pos < recursive_pos


def test_memory_detail_reports_feed_json_to_single_human_facing_sentinel():
    core = Path("scripts/run_daily_improvement_core.sh").read_text(encoding="utf-8")
    post = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")

    assert "scripts/build_memory_engineering_report.py\" --json-only" in core
    for script_name in (
        "build_shion_memory_effect_report.py",
        "audit_persistent_memory.py",
        "obsidian_memory_effectiveness_report.py",
    ):
        script_pos = post.index(script_name)
        assert "--json-only" in post[script_pos : script_pos + 240]
    assert "scripts/build_shion_memory_sentinel_report.py" in post


def test_detailed_growth_and_ops_sidecars_are_weekly_by_default():
    post = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")

    assert 'DETAILED_SIDECAR_REPORT_FREQUENCY="${DETAILED_SIDECAR_REPORT_FREQUENCY:-weekly}"' in post
    assert 'DETAILED_SIDECAR_REPORT_FREQUENCY}" = "daily"' in post
    assert 'DETAILED_SIDECAR_REPORT_FREQUENCY}" = "weekly"' in post
    assert '"$(date +%u)" = "1"' in post
    assert 'if [ "${RUN_DETAILED_SIDECAR_REPORTS}" = "1" ]; then' in post
    assert "detailed_growth_sidecars_skipped" in post
    assert "detailed_ops_sidecars_skipped" in post
    assert "shion_architecture_layer_audit_skipped" in post
