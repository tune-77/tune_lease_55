import os
import subprocess
from pathlib import Path


def test_daily_pipeline_runs_cloudrun_memory_regression_tests():
    script = Path("scripts/run_daily_improvement_core.sh").read_text(encoding="utf-8")
    post_script = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")

    assert "test_find_vault_refreshes_cloudrun_gcs_vault" in script
    assert "test_daily_digest_prefers_note_date_over_gcs_download_mtime" in script
    assert "tests/test_chat_context_builder.py" in script
    assert "tests/test_chat_mid_term_memory.py" in script
    assert "tests/test_build_cloud_chat_memory_pack.py" in script
    assert "tests/test_sync_memory_from_daily_layers.py" in script
    assert "tests/test_build_shion_timeline_delta.py" in script
    assert 'log_step "memory_chat_regression_tests"' in script

    regression_pos = script.index("memory_chat_regression_tests")
    contradiction_pos = script.index("detect_shion_memory_contradictions.py")
    assert regression_pos < contradiction_pos

    final_index_pos = post_script.index("build_shion_memory_index_after_auto_promotions")
    eval_pos = post_script.index("eval_shion_memory_recall.py")
    assert final_index_pos < eval_pos


def test_daily_pipeline_has_single_run_lock(tmp_path):
    script = Path("scripts/run_daily_improvement_pipeline.sh").read_text(encoding="utf-8")

    assert "acquire_pipeline_lock" in script
    assert "install_pipeline_lock_traps" in script
    assert "kill -0" in script
    assert "二重起動を正常スキップ" in script

    lock_dir = tmp_path / "pipeline.lock"
    lock_dir.mkdir()
    (lock_dir / "pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    env = os.environ.copy()
    env["PIPELINE_LOCK_DIR"] = str(lock_dir)
    result = subprocess.run(
        ["bash", "scripts/run_daily_improvement_pipeline.sh"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "二重起動を正常スキップ" in result.stdout


def test_daily_pipeline_does_not_publish_to_gist():
    core_script = Path("scripts/run_daily_improvement_core.sh").read_text(encoding="utf-8")

    assert "gh gist" not in core_script
    assert "GIST_ID" not in core_script
    assert not Path(".github/workflows/pipeline_heartbeat.yml").exists()
