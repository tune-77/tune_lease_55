from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cloudrun_indexes_only_after_gcs_vault_sync_completes() -> None:
    source = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
    start = source.index("    def _run_gcs_vault_sync():")
    end = source.index("    _gcs_th.Thread(", start)
    worker = source[start:end]

    assert worker.index("_sync_gcs_vault_if_enabled()") < worker.index("run_indexing(vault_path)")
    assert 'gcs_sync.get("local_dir")' in worker
    assert ".wait(timeout=600)" not in source


def test_cloudrun_api_keeps_cpu_available_for_background_sync() -> None:
    deploy = (ROOT / "scripts" / "deploy_cloud_run_api.sh").read_text(encoding="utf-8")

    assert "--no-cpu-throttling" in deploy
