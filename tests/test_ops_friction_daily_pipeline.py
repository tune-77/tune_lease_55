from pathlib import Path


def test_ops_friction_doctor_wired_into_daily_post_after_sentinel():
    script = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")

    sentinel_pos = script.index("scripts/build_shion_memory_sentinel_report.py")
    ops_pos = script.index("scripts/ops_friction_doctor.py")
    graph_pos = script.index("scripts/build_obsidian_graph_judgment_effect.py")

    assert sentinel_pos < ops_pos < graph_pos
    assert "--apply-safe" in script[ops_pos:graph_pos]
    assert "reports/ops_friction_latest.json" in script[ops_pos:graph_pos]
    assert "reports/ops_friction_latest.md" in script[ops_pos:graph_pos]
    assert 'log_step "ops_friction_doctor"' in script[ops_pos:graph_pos]
