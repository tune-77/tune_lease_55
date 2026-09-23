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

    # ChromaDBのGCSスナップショット復元導入後も、復元できない場合は
    # その場のフル索引（埋め込みモデル呼び出し）にフォールバックするため、
    # --no-cpu-throttling はまだ外せない（本番計測で確認してから外す）。
    assert "--no-cpu-throttling" in deploy


def test_cloudrun_restores_chroma_snapshot_before_uvicorn_starts() -> None:
    """restore_chroma_snapshot.py が bundle 復元後・uvicorn起動前に呼ばれることを固定する。

    api/main.py の起動時フル索引（_run_gcs_vault_sync）より前に、GCSの
    ビルド済みChromaDBを復元できれば、api/knowledge/indexer.py の差分判定
    （mtime比較）が効いて再埋め込みコストを大きく減らせる。
    """
    start_script = (ROOT / "scripts" / "start_api_cloud_run.sh").read_text(encoding="utf-8")

    restore_pos = start_script.index("restore_chroma_snapshot.py")
    uvicorn_pos = start_script.index("uvicorn api.main:app")
    assert restore_pos < uvicorn_pos


def test_chroma_snapshot_module_mirrors_db_snapshot_pattern() -> None:
    """api/cloudrun_db_snapshot.py と同じenabled判定・GCSLock方式を踏襲していることを固定する。"""
    source = (ROOT / "api" / "knowledge" / "chroma_snapshot.py").read_text(encoding="utf-8")

    assert "def is_snapshot_enabled" in source
    assert "def snapshot_and_upload" in source
    assert "def restore" in source
    assert "GCSLock" in source
