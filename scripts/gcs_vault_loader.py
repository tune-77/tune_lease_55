"""Download Obsidian .md files from a GCS vault prefix to a local directory.

Environment variables:
    GCS_BUCKET         GCS バケット名（デフォルト: tune-lease-55-data）
    GCS_VAULT_PREFIX   バケット内のプレフィックス（デフォルト: vault/）
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "tune-lease-55-data")
GCS_VAULT_PREFIX = os.environ.get("GCS_VAULT_PREFIX", "vault/")
_DEFAULT_LOCAL_DIR = Path("/tmp/gcs_vault")


def download_vault(
    *,
    dest_dir: Path | None = None,
    bucket: str | None = None,
    prefix: str | None = None,
) -> Path:
    """GCS の vault プレフィックス配下の .md を dest_dir へダウンロードする。

    Returns:
        dest_dir (ダウンロード先ディレクトリの Path)
    """
    from google.cloud import storage  # type: ignore[import-untyped]

    bkt = bucket or GCS_BUCKET
    pfx = prefix or GCS_VAULT_PREFIX
    dest = dest_dir or _DEFAULT_LOCAL_DIR
    dest.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    blobs = list(client.list_blobs(bkt, prefix=pfx))
    downloaded = 0
    for blob in blobs:
        if not blob.name.endswith(".md"):
            continue
        rel = blob.name[len(pfx):]
        if not rel:
            continue
        local = dest / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local))
        downloaded += 1

    logger.info("[gcs_vault_loader] downloaded %d .md files from gs://%s/%s to %s", downloaded, bkt, pfx, dest)
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
