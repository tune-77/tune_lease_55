"""iCloud → GCS 差分アップロードスクリプト（タイプ2ファイル用）

Obsidian Vault の *.md ファイルを GCS に差分アップロードする。
ローカルファイルの mtime をカスタムメタデータ `local_mtime` として GCS に保存し、
次回実行時に比較して変化があるファイルのみアップロードする。

Usage:
    python scripts/icloud_to_gcs_sync.py
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from google.cloud import storage

GCS_BUCKET = os.environ.get("GCS_BUCKET", "tune-lease-55-data")
GCS_VAULT_PREFIX = os.environ.get("GCS_VAULT_PREFIX", "vault/")
GCS_UPLOAD_BACKEND = os.environ.get("GCS_UPLOAD_BACKEND", "gcloud").lower()
LOCAL_VAULT_DIR = os.environ.get(
    "LOCAL_VAULT_DIR",
    "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault",
)
INCLUDED_REL_PREFIXES = tuple(
    item.strip().strip("/")
    for item in os.environ.get(
        "GCS_VAULT_INCLUDED_PREFIXES",
        ",".join(
            [
                "リース知識",
                "Projects/tune_lease_55/Research",
                "Projects/tune_lease_55/News",
                "Projects/tune_lease_55/Asset Knowledge",
                "Projects/tune_lease_55/Asset Finance",
                "Projects/tune_lease_55/Lease Intelligence/Public",
                "Projects/tune_lease_55/Industry",
            ]
        ),
    ).split(",")
    if item.strip()
)
EXCLUDED_REL_PREFIXES = tuple(
    item.strip().strip("/")
    for item in os.environ.get(
        "GCS_VAULT_EXCLUDED_PREFIXES",
        ",".join(
            [
                "Daily",
                "Private Reflection",
                "チャット記録",
                "Codex",
                "Projects/tune_lease_55/Cloud SQL Summaries",
                "Projects/tune_lease_55/Cloud Run Inputs",
            ]
        ),
    ).split(",")
    if item.strip()
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
    try:
        blob.upload_from_filename(str(local_path))
    except Exception as exc:
        if not _upload_with_gcloud(local_path, bucket.name, gcs_path, local_mtime):
            raise exc
    return f"[UP]    {rel} → gs://{bucket.name}/{gcs_path}"


def _upload_with_gcloud(local_path: Path, bucket_name: str, gcs_path: str, local_mtime: str) -> bool:
    """ADCが壊れている環境では、ログイン済み gcloud CLI でアップロードする。"""
    try:
        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                f"--custom-metadata=local_mtime={local_mtime}",
                str(local_path),
                f"gs://{bucket_name}/{gcs_path}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return True
    except Exception as fallback_exc:
        print(f"警告: gcloud storage cp フォールバックも失敗しました: {type(fallback_exc).__name__}", file=sys.stderr)
        return False


def _matches_prefix(rel: str, prefixes: tuple[str, ...]) -> bool:
    return any(rel == prefix or rel.startswith(prefix + "/") for prefix in prefixes)


def collect_md_files(vault_dir: Path) -> list[Path]:
    """Cloud Run AIチャットに渡してよい Obsidian *.md だけを列挙する。"""
    result = []
    for p in vault_dir.rglob("*.md"):
        if ".obsidian" in p.parts:
            continue
        rel = p.relative_to(vault_dir).as_posix()
        if EXCLUDED_REL_PREFIXES and _matches_prefix(rel, EXCLUDED_REL_PREFIXES):
            continue
        if INCLUDED_REL_PREFIXES and not _matches_prefix(rel, INCLUDED_REL_PREFIXES):
            continue
        result.append(p)
    return result


def main() -> None:
    vault_dir = Path(LOCAL_VAULT_DIR)
    print(f"iCloud → GCS 同期開始: {vault_dir} → gs://{GCS_BUCKET}/{GCS_VAULT_PREFIX}")

    md_files = collect_md_files(vault_dir)
    print(f"対象ファイル数: {len(md_files)}")
    print(f"許可prefix: {', '.join(INCLUDED_REL_PREFIXES) or '(all)'}")
    print(f"除外prefix: {', '.join(EXCLUDED_REL_PREFIXES) or '(none)'}")

    if not md_files:
        print("アップロード対象ファイルなし。正常終了。")
        return

    bucket = None
    if GCS_UPLOAD_BACKEND != "gcloud":
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)

    uploaded = 0
    skipped = 0
    for path in md_files:
        if GCS_UPLOAD_BACKEND == "gcloud":
            rel = path.relative_to(vault_dir)
            gcs_path = GCS_VAULT_PREFIX + rel.as_posix()
            if not _upload_with_gcloud(path, GCS_BUCKET, gcs_path, _local_mtime_str(path)):
                raise RuntimeError(f"gcloud upload failed: {rel}")
            result = f"[UP]    {rel} → gs://{GCS_BUCKET}/{gcs_path}"
        else:
            assert bucket is not None
            result = upload_file(bucket, path, vault_dir, GCS_VAULT_PREFIX)
        print(result)
        if result.startswith("[UP]"):
            uploaded += 1
        else:
            skipped += 1

    print(f"完了: アップロード {uploaded} 件 / スキップ {skipped} 件")


if __name__ == "__main__":
    main()
