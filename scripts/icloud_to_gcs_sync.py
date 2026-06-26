"""iCloud → GCS 差分アップロードスクリプト（タイプ2ファイル用）

Obsidian Vault の *.md ファイルを GCS に差分アップロードする。
ローカルファイルの mtime をカスタムメタデータ `local_mtime` として GCS に保存し、
次回実行時に比較して変化があるファイルのみアップロードする。

Usage:
    python scripts/icloud_to_gcs_sync.py
"""
import os
import sys
from pathlib import Path
from typing import Optional

from google.cloud import storage

GCS_BUCKET = os.environ.get("GCS_BUCKET", "tune-lease-55-data")
GCS_VAULT_PREFIX = os.environ.get("GCS_VAULT_PREFIX", "vault/")
LOCAL_VAULT_DIR = os.environ.get(
    "LOCAL_VAULT_DIR",
    "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault",
)


def _local_mtime_str(path: Path) -> str:
    """ローカルファイルの mtime を ISO 形式文字列で返す。"""
    return str(path.stat().st_mtime)


def _gcs_local_mtime(blob: storage.Blob) -> Optional[str]:
    """GCS オブジェクトのカスタムメタデータ `local_mtime` を返す。なければ None。"""
    meta = blob.metadata or {}
    return meta.get("local_mtime")


def upload_file(
    bucket: storage.Bucket,
    local_path: Path,
    vault_dir: Path,
    gcs_prefix: str,
) -> str:
    """単一ファイルを差分チェックしてアップロードする。戻り値は操作の説明文字列。"""
    rel = local_path.relative_to(vault_dir)
    gcs_path = gcs_prefix + rel.as_posix()
    local_mtime = _local_mtime_str(local_path)

    blob = bucket.blob(gcs_path)
    try:
        blob.reload()
        gcs_mtime = _gcs_local_mtime(blob)
    except Exception:
        gcs_mtime = None

    if gcs_mtime is not None and gcs_mtime == local_mtime:
        return f"[SKIP]  {rel}"

    blob.metadata = {"local_mtime": local_mtime}
    blob.upload_from_filename(str(local_path))
    return f"[UP]    {rel} → gs://{bucket.name}/{gcs_path}"


def collect_md_files(vault_dir: Path) -> list[Path]:
    """Vault 内の *.md ファイルを再帰列挙する。.obsidian/ 配下は除外。"""
    result = []
    for p in vault_dir.rglob("*.md"):
        if ".obsidian" in p.parts:
            continue
        result.append(p)
    return result


def main() -> None:
    vault_dir = Path(LOCAL_VAULT_DIR)
    print(f"iCloud → GCS 同期開始: {vault_dir} → gs://{GCS_BUCKET}/{GCS_VAULT_PREFIX}")

    md_files = collect_md_files(vault_dir)
    print(f"対象ファイル数: {len(md_files)}")

    if not md_files:
        print("アップロード対象ファイルなし。正常終了。")
        return

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    uploaded = 0
    skipped = 0
    for path in md_files:
        result = upload_file(bucket, path, vault_dir, GCS_VAULT_PREFIX)
        print(result)
        if result.startswith("[UP]"):
            uploaded += 1
        else:
            skipped += 1

    print(f"完了: アップロード {uploaded} 件 / スキップ {skipped} 件")


if __name__ == "__main__":
    main()
