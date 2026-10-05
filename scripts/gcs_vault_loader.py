"""Download Obsidian .md files from a GCS vault prefix to a local directory.

Environment variables:
    GCS_BUCKET         GCS バケット名（デフォルト: tune-lease-55-data）
    GCS_VAULT_PREFIX   バケット内のプレフィックス（デフォルト: vault/）
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "tune-lease-55-data")
GCS_VAULT_PREFIX = os.environ.get("GCS_VAULT_PREFIX", "vault/")
_DEFAULT_LOCAL_DIR = Path("/tmp/gcs_vault")
_DEFAULT_DOWNLOAD_WORKERS = 8
_MAX_DOWNLOAD_WORKERS = 32
_MIRROR_MARKER = ".gcs-vault-mirror"
_MAX_STALE_RATIO = 0.25
_MAX_STALE_FILES = 100


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


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _assert_safe_mirror_destination(dest: Path) -> None:
    """実VaultをGCSミラー先として誤指定する事故を拒否する。"""
    resolved = dest.resolve()
    if (resolved / ".obsidian").exists():
        raise RuntimeError(f"Refusing to use an Obsidian vault as a GCS mirror: {resolved}")

    managed_mirror = (resolved / _MIRROR_MARKER).exists()
    for env_name in ("OBSIDIAN_VAULT", "OBSIDIAN_VAULT_PATH"):
        configured = (os.environ.get(env_name) or "").strip()
        if configured and resolved == Path(configured).expanduser().resolve() and not managed_mirror:
            raise RuntimeError(f"Refusing to use {env_name} as a GCS mirror: {resolved}")

    temp_roots = {Path("/tmp").resolve(), Path("/private/tmp").resolve(), Path(tempfile.gettempdir()).resolve()}
    in_temp = any(resolved == root or _is_relative_to(resolved, root) for root in temp_roots)
    if not in_temp and not managed_mirror:
        raise RuntimeError(
            f"Unmanaged GCS mirror destination: {resolved}. "
            f"Create {resolved / _MIRROR_MARKER} only after confirming this is a disposable mirror."
        )


def _plan_stale_markdown(dest: Path, expected_paths: set[Path]) -> list[Path]:
    """隔離対象を算出し、異常な大量変更ならミラー更新前に停止する。"""
    local_paths = {
        local_md.relative_to(dest)
        for local_md in dest.rglob("*.md")
        if _is_relative_to(local_md, dest)
    }
    if not expected_paths:
        raise RuntimeError("Refusing to quarantine every local note because the GCS listing is empty")
    stale_paths = sorted(local_paths - expected_paths)
    if not stale_paths:
        return []

    stale_ratio = len(stale_paths) / len(local_paths)
    too_many_files = len(stale_paths) > _MAX_STALE_FILES
    too_large_ratio = stale_ratio > _MAX_STALE_RATIO
    if (too_many_files or too_large_ratio) and os.environ.get("GCS_VAULT_ALLOW_LARGE_QUARANTINE") != "1":
        raise RuntimeError(
            f"Refusing unusually large GCS mirror change: {len(stale_paths)}/{len(local_paths)} notes are stale"
        )
    return stale_paths


def _quarantine_stale_markdown(dest: Path, stale_paths: list[Path]) -> int:
    """検証済みの stale .md を、削除せずミラー外へ隔離する。"""
    if not stale_paths:
        return 0
    quarantine_root = dest.parent / f".{dest.name}-quarantine" / datetime.now().strftime("%Y%m%dT%H%M%S%f")
    moved = 0
    for rel in stale_paths:
        local_md = dest / rel
        target = quarantine_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(local_md), str(target))
        moved += 1
    return moved


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
    _assert_safe_mirror_destination(dest)
    (dest / _MIRROR_MARKER).touch(exist_ok=True)

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

    expected_paths = {rel for _, rel in md_blobs}
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}-sync-", dir=dest.parent))

    def _download(item: tuple[object, Path]) -> int:
        blob, rel = item
        local = staging / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local), timeout=30)
        return 1

    workers = min(_download_worker_count(max_workers), len(md_blobs)) if md_blobs else 1
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gcs-vault-download") as executor:
            downloaded = sum(executor.map(_download, md_blobs))
        # 既存ミラーへ1件でも置換する前に、大量欠落や空一覧を検出する。
        stale_paths = _plan_stale_markdown(dest, expected_paths)
        for _, rel in md_blobs:
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / rel, target)
        quarantined = _quarantine_stale_markdown(dest, stale_paths)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info(
        "[gcs_vault_loader] downloaded %d .md files with %d workers, "
        "quarantined %d stale files from gs://%s/%s to %s",
        downloaded,
        workers,
        quarantined,
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
