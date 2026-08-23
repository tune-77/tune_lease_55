"""Cloud Run上のlease_data.dbをGCSへ定期スナップショットし、再起動時に復元する。

Cloud Runのコンテナローカルディスクはコールドスタート・インスタンス再起動の
たびに消え、バンドル（.cloudrun_bundle）の内容へ巻き戻る。demoモードはそれが
仕様だが、非demoモード（実データ）では審査結果登録が再起動のたびに消失する。

既存のprompt_feedback_log.jsonl向けパターン（scripts/restore_prompt_feedback_snapshot.py /
scripts/sync_cloudrun_inputs_from_gcs.pyの_upload_prompt_feedback_snapshot）を踏襲し、
DBファイル用に拡張したもの。生きたSQLiteファイルをそのままアップロードすると
WAL/ジャーナル不整合を持ち込む恐れがあるため、`VACUUM INTO`で一貫性のある
コピーを作ってからアップロードする。
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_HISTORY_KEEP = 5


def _bucket_name() -> str:
    value = (os.environ.get("GCS_BUCKET", "tune-lease-55-data") or "").strip()
    if value.startswith("gs://"):
        value = value[5:]
    return value.split("/", 1)[0]


def _snapshot_prefix() -> str:
    prefix = os.environ.get("GCS_SNAPSHOT_PREFIX", "cloudrun-snapshots/").strip("/") or "cloudrun-snapshots"
    return prefix


def _latest_blob_name(db_filename: str) -> str:
    return f"{_snapshot_prefix()}/{db_filename}"


def _history_blob_name(db_filename: str, ts: str) -> str:
    return f"{_snapshot_prefix()}/history/{db_filename}.{ts}"


def is_snapshot_enabled() -> bool:
    """Cloud Run系の非demo実行だけをスナップショット対象にする。"""
    mode = os.environ.get("CLOUDRUN_DATA_MODE", "").strip().lower()
    if mode == "demo":
        return False
    if mode:
        return True
    return bool(os.environ.get("K_SERVICE", "").strip())


def _vacuum_copy(src_path: str, dst_path: str) -> None:
    conn = sqlite3.connect(src_path)
    try:
        conn.execute("VACUUM INTO ?", (dst_path,))
    finally:
        conn.close()


def snapshot_and_upload(db_path: str | None = None) -> dict:
    """lease_data.dbの一貫性コピーを作り、GCSへアップロードする。失敗しても例外は投げない。"""
    from runtime_paths import get_db_path

    db_path = db_path or get_db_path()
    result: dict = {"enabled": is_snapshot_enabled(), "uploaded": False}
    if not result["enabled"]:
        return result
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        result["reason"] = "db_missing_or_empty"
        return result

    try:
        from google.cloud import storage
    except Exception as exc:
        result["reason"] = f"storage_unavailable: {exc}"
        return result

    db_filename = os.path.basename(db_path)
    tmp_dir = tempfile.mkdtemp(prefix="cloudrun_db_snapshot_")
    tmp_path = os.path.join(tmp_dir, db_filename)
    try:
        _vacuum_copy(db_path, tmp_path)

        from scripts.gcs_lock import GCSLock, GCSLockError

        bucket_name = _bucket_name()
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        try:
            with GCSLock(
                bucket_name=bucket_name,
                lock_path=f"{_snapshot_prefix()}/.lock",
                writer="cloudrun-api",
                target_file=_latest_blob_name(db_filename),
            ):
                bucket.blob(_latest_blob_name(db_filename)).upload_from_filename(tmp_path)
                bucket.blob(_history_blob_name(db_filename, ts)).upload_from_filename(tmp_path)
                _prune_history(bucket, db_filename)
        except GCSLockError as exc:
            result["reason"] = f"lock_timeout: {exc}"
            return result

        result["uploaded"] = True
        result["blob"] = f"gs://{bucket_name}/{_latest_blob_name(db_filename)}"
        print(f"[CloudRunDBSnapshot] アップロード完了: {result['blob']}")
        return result
    except Exception as exc:
        result["reason"] = str(exc)
        print(f"[CloudRunDBSnapshot] アップロード失敗（非致命的）: {exc}")
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _prune_history(bucket, db_filename: str) -> None:
    """world/history/配下の古い世代を直近_HISTORY_KEEP件だけ残して削除する。"""
    prefix = f"{_snapshot_prefix()}/history/{db_filename}."
    blobs = sorted(bucket.list_blobs(prefix=prefix), key=lambda b: b.name)
    stale = blobs[:-_HISTORY_KEEP] if len(blobs) > _HISTORY_KEEP else []
    for blob in stale:
        try:
            blob.delete()
        except Exception:
            pass
