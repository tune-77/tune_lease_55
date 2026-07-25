from pathlib import Path


def test_daily_post_pipeline_stops_distribution_when_mana_is_not_allow():
    script = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")

    mana_pos = script.index("scripts/mana_obsidian_curator.py")
    status_pos = script.index("MANA_STATUS=")
    gate_pos = script.index('if [ "${MANA_STATUS}" != "allow" ]')
    eval_pos = script.index("scripts/build_shion_eval_candidates.py")
    gcs_pos = script.index("scripts/icloud_to_gcs_sync.py")
    slack_pos = script.index("scripts/send_daily_improvement_slack.py")
    curator_pos = script.index("scripts/obsidian_curator_report.py")
    growth_pos = script.index("scripts/judgment_asset_growth_report.py")
    eval_growth_pos = script.index("scripts/evaluate_shion_growth.py")
    graph_pos = script.index("scripts/build_judgment_asset_graph.py")
    terms_pos = script.index("scripts/screening_terms_audit.py")

    assert mana_pos < status_pos < curator_pos < growth_pos < eval_growth_pos < graph_pos < terms_pos < slack_pos < gate_pos < eval_pos < gcs_pos
    assert '--mana-report "${MANA_REPORT_JSON}"' in script
    assert '--screening-terms-report "${SCREENING_TERMS_REPORT_JSON}"' in script
    assert "審査用語監査を生成" in script
    assert "評価候補生成と GCS Vault 配布を停止" in script
    assert "exit 0" in script[gate_pos:eval_pos]


def test_daily_post_pipeline_repairs_private_reflection_before_stopping():
    script = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")

    assert "MANA_MAX_REFLECTION_REPAIRS" in script
    assert "mana_wants_reflection_repair" in script
    assert "private_reflection_not_meaningful" in script
    assert "reflection_handoff_incomplete" in script
    assert "reflection_too_similar" in script
    assert "Private Reflection が弱いため、再生成して Mana に再判定" in script

    loop_pos = script.index("while true; do")
    repair_pos = script.index("mana_wants_reflection_repair", loop_pos)
    reflection_pos = script.index("lease_intelligence_reflection.py", repair_pos)
    delta_pos = script.index("build_reflection_delta", reflection_pos)
    stop_pos = script.index("Mana が allow ではないため")

    assert loop_pos < repair_pos < reflection_pos < delta_pos < stop_pos


def test_daily_post_pipeline_refreshes_reports_after_ledger_sync_before_slack():
    script = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")

    ledger_pos = script.index("scripts/check_ledger_consistency.py")
    sync_pos = script.index("sync_improvement_reports.py", ledger_pos)
    recursive_pos = script.index("scripts/recursive_self_improvement.py", sync_pos)
    quality_pos = script.index("scripts/analyze_improvement_quality.py", recursive_pos)
    loop_pos = script.index("scripts/loop_metrics.py", quality_pos)
    proof_pos = script.index("scripts/build_loop_proof.py", loop_pos)
    slack_pos = script.index("scripts/send_daily_improvement_slack.py")

    assert ledger_pos < sync_pos < recursive_pos < quality_pos < loop_pos < proof_pos < slack_pos
    assert "--from-ledger" in script[sync_pos:recursive_pos]
    assert "sync_improvement_reports_post" in script
    assert "recursive_self_improvement_post" in script
    assert "loop_metrics_post" in script


def test_daily_post_pipeline_runs_judgment_asset_graph_weekly_by_default():
    script = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")

    freq_pos = script.index("JUDGMENT_ASSET_GRAPH_FREQUENCY")
    graph_gate_pos = script.index("RUN_JUDGMENT_ASSET_GRAPH=0")
    graph_script_pos = script.index("scripts/build_judgment_asset_graph.py")
    sync_pos = script.index("sync_graph_to_public")

    assert freq_pos < graph_gate_pos < graph_script_pos < sync_pos
    assert 'JUDGMENT_ASSET_GRAPH_FREQUENCY:-weekly' in script
    assert 'JUDGMENT_ASSET_GRAPH_FREQUENCY}" = "daily"' in script
    assert 'date +%u' in script
    assert "判断資産グラフlatestが無いため public 同期をスキップ" in script
