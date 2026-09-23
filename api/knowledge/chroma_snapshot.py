"""ChromaDB永続ディレクトリ（api/chroma_db）をGCSへスナップショットし、
起動時に復元する。

api/cloudrun_db_snapshot.py（lease_data.db向け）と同じ理由・同じパターン:
Cloud Runのコンテナローカルディスクはコールドスタートのたびに消える。
起動のたびに埋め込みモデルでVault全体を再ベクトル化していたのが
`--no-cpu-throttling`（アイドル課金）が必要だった一因のため、前回ビルド済みの
ChromaDBをGCSから復元できれば、起動時の索引は差分（変更分のみ）で済む
（api/knowledge/indexer.py のmtime比較は既に差分方式）。

ディレクトリ全体をtarにまとめて1オブジェクトとして up/download する
（多数の小ファイルより単純でアトミック）。
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from pathlib import Path

_HISTORY_KEEP = 3


def _bucket_name() -> str:
    value = (os.environ.get("GCS_BUCKET", "tune-lease-55-data") or "").strip()
    if value.startswith("gs://"):
        value = value[5:]
    return value.split("/", 1)[0]


def _snapshot_prefix() -> str:
    prefix = os.environ.get("GCS_SNAPSHOT_PREFIX", "cloudrun-snapshots/").strip("/") or "cloudrun-snapshots"
    return prefix


def _blob_name() -> str:
    return f"{_snapshot_prefix()}/chroma_db.tar.gz"


def _history_blob_name(ts: str) -> str:
    return f"{_snapshot_prefix()}/history/chroma_db.tar.gz.{ts}"


def is_snapshot_enabled() -> bool:
    """lease_data.dbスナップショットと同じ判定（demoモードは対象外）。"""
    mode = os.environ.get("CLOUDRUN_DATA_MODE", "").strip().lower()
    if mode == "demo":
        return False
    if mode:
        return True
    return bool(os.environ.get("K_SERVICE", "").strip())


def snapshot_and_upload(chroma_dir: str | None = None) -> dict:
    """ChromaDBディレクトリをtar化してGCSへアップロードする。失敗しても例外は投げない。"""
    from api.knowledge.vector_store import _CHROMA_DIR

    chroma_dir = chroma_dir or _CHROMA_DIR
    result: dict = {"enabled": is_snapshot_enabled(), "uploaded": False}
    if not result["enabled"]:
        return result
    if not os.path.isdir(chroma_dir) or not os.listdir(chroma_dir):
        result["reason"] = "chroma_dir_missing_or_empty"
        return result

    try:
        from google.cloud import storage
    except Exception as exc:
        result["reason"] = f"storage_unavailable: {exc}"
        return result

    tmp_dir = tempfile.mkdtemp(prefix="chroma_snapshot_")
    tmp_tar = os.path.join(tmp_dir, "chroma_db.tar.gz")
    try:
        with tarfile.open(tmp_tar, "w:gz") as tar:
            tar.add(chroma_dir, arcname="chroma_db")

        from datetime import datetime, timezone

        from scripts.gcs_lock import GCSLock, GCSLockError

        bucket_name = _bucket_name()
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        try:
            with GCSLock(
                bucket_name=bucket_name,
                lock_path=f"{_snapshot_prefix()}/.chroma_lock",
                writer="cloudrun-api-chroma",
                target_file=_blob_name(),
            ):
                bucket.blob(_blob_name()).upload_from_filename(tmp_tar)
                bucket.blob(_history_blob_name(ts)).upload_from_filename(tmp_tar)
                _prune_history(bucket)
        except GCSLockError as exc:
            result["reason"] = f"lock_timeout: {exc}"
            return result

        result["uploaded"] = True
        result["blob"] = f"gs://{bucket_name}/{_blob_name()}"
        print(f"[ChromaSnapshot] アップロード完了: {result['blob']}")
        return result
    except Exception as exc:
        result["reason"] = str(exc)
        print(f"[ChromaSnapshot] アップロード失敗（非致命的）: {exc}")
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _prune_history(bucket) -> None:
    prefix = f"{_snapshot_prefix()}/history/chroma_db.tar.gz."
    blobs = sorted(bucket.list_blobs(prefix=prefix), key=lambda b: b.name)
    stale = blobs[:-_HISTORY_KEEP] if len(blobs) > _HISTORY_KEEP else []
    for blob in stale:
        try:
            blob.delete()
        except Exception:
            pass


def restore(chroma_dir: str | None = None) -> dict:
    """GCSスナップショットを取得し、chroma_dirへ展開する（非破壊: 既存があればスキップ）。

    失敗しても例外は投げない。呼び出し側（起動時）は復元できなければ
    その場のフル索引（既存フォールバック）にそのまま流れる。
    """
    from api.knowledge.vector_store import _CHROMA_DIR

    chroma_dir = chroma_dir or _CHROMA_DIR
    result: dict = {"enabled": is_snapshot_enabled(), "restored": False}
    if not result["enabled"]:
        return result
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        result["reason"] = "chroma_dir_already_populated"
        return result

    try:
        from google.cloud import storage
        from google.api_core.exceptions import NotFound
    except Exception as exc:
        result["reason"] = f"storage_unavailable: {exc}"
        return result

    tmp_dir = tempfile.mkdtemp(prefix="chroma_restore_")
    try:
        bucket_name = _bucket_name()
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(_blob_name())
        tmp_tar = os.path.join(tmp_dir, "chroma_db.tar.gz")
        blob.download_to_filename(tmp_tar)

        with tarfile.open(tmp_tar, "r:gz") as tar:
            tar.extractall(tmp_dir, filter="data")

        extracted = os.path.join(tmp_dir, "chroma_db")
        os.makedirs(os.path.dirname(chroma_dir) or ".", exist_ok=True)
        shutil.move(extracted, chroma_dir)

        result["restored"] = True
        result["blob"] = f"gs://{bucket_name}/{_blob_name()}"
        print(f"[ChromaSnapshot] 復元完了: {result['blob']} → {chroma_dir}")
        return result
    except NotFound:
        result["reason"] = "snapshot_not_found"
        print("[ChromaSnapshot] スナップショット未作成: その場でフル索引します")
        return result
    except Exception as exc:
        result["reason"] = str(exc)
        print(f"[ChromaSnapshot] 復元失敗（非致命的、フル索引にフォールバック）: {exc}")
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
