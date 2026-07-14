from pathlib import Path


def test_daily_post_pipeline_stops_distribution_when_mana_is_not_allow():
    script = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")

    mana_pos = script.index("scripts/mana_obsidian_curator.py")
    status_pos = script.index("MANA_STATUS=")
    gate_pos = script.index('if [ "${MANA_STATUS}" != "allow" ]')
    eval_pos = script.index("scripts/build_shion_eval_candidates.py")
    gcs_pos = script.index("scripts/icloud_to_gcs_sync.py")
    slack_pos = script.index("scripts/send_daily_improvement_slack.py")

    assert mana_pos < status_pos < slack_pos < gate_pos < eval_pos < gcs_pos
    assert '--mana-report "${MANA_REPORT_JSON}"' in script
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
