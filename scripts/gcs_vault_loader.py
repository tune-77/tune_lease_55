"""Download Obsidian .md files from a GCS vault prefix to a local directory.

Environment variables:
    GCS_BUCKET         GCS バケット名（デフォルト: tune-lease-55-data）
    GCS_VAULT_PREFIX   バケット内のプレフィックス（デフォルト: vault/）
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "tune-lease-55-data")
GCS_VAULT_PREFIX = os.environ.get("GCS_VAULT_PREFIX", "vault/")
_DEFAULT_LOCAL_DIR = Path("/tmp/gcs_vault")
_DEFAULT_DOWNLOAD_WORKERS = 8
_MAX_DOWNLOAD_WORKERS = 32


def _bucket_name(value: str) -> str:
    """gs://bucket/prefix 形式でも Storage API の bucket 名へ正規化する。"""
    normalized = (value or "").strip()
    if normalized.startswith("gs://"):
        normalized = normalized[5:]
    return normalized.split("/", 1)[0]


def _safe_relative_path(blob_name: str, prefix: str) -> Path | None:
    """GCS blob 名を dest_dir 配下の安全な相対パスへ変換する。"""
    rel = blob_name[len(prefix):]
    if not rel:
        return None
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        logger.warning("[gcs_vault_loader] skipped unsafe blob path: %s", blob_name)
        return None
    return path


def _prune_stale_markdown(dest: Path, expected_paths: set[Path]) -> int:
    """GCS に存在しないローカル .md を削除する。"""
    removed = 0
    for local_md in sorted(dest.rglob("*.md")):
        try:
            rel = local_md.relative_to(dest)
        except ValueError:
            continue
        if rel in expected_paths:
            continue
        local_md.unlink()
        removed += 1
    return removed


def _download_worker_count(value: int | None = None) -> int:
    """並列ダウンロード数を1〜32に制限する。"""
    if value is None:
        try:
            value = int(os.environ.get("GCS_VAULT_DOWNLOAD_WORKERS", _DEFAULT_DOWNLOAD_WORKERS))
        except (TypeError, ValueError):
            value = _DEFAULT_DOWNLOAD_WORKERS
    return max(1, min(_MAX_DOWNLOAD_WORKERS, value))


def download_vault(
    *,
    dest_dir: Path | None = None,
    bucket: str | None = None,
    prefix: str | None = None,
    max_workers: int | None = None,
) -> Path:
    """GCS の vault プレフィックス配下の .md を dest_dir へダウンロードする。

    Returns:
        dest_dir (ダウンロード先ディレクトリの Path)
    """
    from google.cloud import storage  # type: ignore[import-untyped]

    bkt = _bucket_name(bucket or GCS_BUCKET)
    pfx = prefix or GCS_VAULT_PREFIX
    dest = dest_dir or _DEFAULT_LOCAL_DIR
    dest.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    blobs = list(client.list_blobs(bkt, prefix=pfx, timeout=30))
    md_blobs: list[tuple[object, Path]] = []
    for blob in blobs:
        if not blob.name.endswith(".md"):
            continue
        rel = _safe_relative_path(blob.name, pfx)
        if rel is None:
            continue
        md_blobs.append((blob, rel))

    pruned = _prune_stale_markdown(dest, {rel for _, rel in md_blobs})

    def _download(item: tuple[object, Path]) -> int:
        blob, rel = item
        local = dest / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local), timeout=30)
        return 1

    workers = min(_download_worker_count(max_workers), len(md_blobs)) if md_blobs else 1
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gcs-vault-download") as executor:
        downloaded = sum(executor.map(_download, md_blobs))

    logger.info(
        "[gcs_vault_loader] downloaded %d .md files with %d workers, "
        "pruned %d stale files from gs://%s/%s to %s",
        downloaded,
        workers,
        pruned,
        bkt,
        pfx,
        dest,
    )
    return dest


def load_vault_texts(
    *,
    dest_dir: Path | None = None,
    bucket: str | None = None,
    prefix: str | None = None,
) -> list[str]:
    """GCS vault をダウンロードし、.md ファイルのテキスト一覧を返す。"""
    vault_dir = download_vault(dest_dir=dest_dir, bucket=bucket, prefix=prefix)
    texts: list[str] = []
    for path in sorted(vault_dir.rglob("*.md")):
        try:
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            pass
    return texts
