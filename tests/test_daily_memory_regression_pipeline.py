from pathlib import Path


def test_daily_pipeline_runs_cloudrun_memory_regression_tests():
    script = Path("scripts/run_daily_improvement_core.sh").read_text(encoding="utf-8")

    assert "test_find_vault_refreshes_cloudrun_gcs_vault" in script
    assert "tests/test_chat_context_builder.py" in script
    assert "tests/test_chat_mid_term_memory.py" in script
    assert "tests/test_build_cloud_chat_memory_pack.py" in script
    assert "tests/test_sync_memory_from_daily_layers.py" in script
    assert "tests/test_build_shion_timeline_delta.py" in script
    assert 'log_step "memory_chat_regression_tests"' in script

    eval_pos = script.index("eval_shion_memory_recall.py")
    regression_pos = script.index("memory_chat_regression_tests")
    contradiction_pos = script.index("detect_shion_memory_contradictions.py")
    assert eval_pos < regression_pos < contradiction_pos
