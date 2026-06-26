"""GCS → ローカル同期スクリプト（タイプ1ファイル用）

GCS 上の紫苑ファイル（mind.json, world_view.json 等）をローカルキャッシュに
ダウンロードする。GCS の `mtime` カスタムメタデータとローカルの mtime を比較し、
GCS が新しい場合のみ上書きする（読み取り専用のためロック不要）。

Usage:
    python scripts/gcs_to_local_sync.py
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from google.cloud import storage

GCS_BUCKET = os.environ.get("GCS_BUCKET", "tune-lease-55-data")
LOCAL_SHION_DIR = os.environ.get("LOCAL_SHION_DIR", "./data/shion/")

# taip1対象ファイル: (GCS パス, 必須フラグ)
_SYNC_FILES = [
    ("shion/mind.json", True),
    ("shion/world_view.json", True),
    ("shion/keypoints.json", False),
]


def _parse_gcs_mtime(blob: storage.Blob) -> Optional[datetime]:
    """GCS カスタムメタデータの `mtime` を UTC datetime として返す。なければ None。"""
    meta = blob.metadata or {}
    raw = meta.get("mtime")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _local_mtime(local_path: Path) -> Optional[datetime]:
    """ローカルファイルの mtime を UTC datetime として返す。なければ None。"""
    if not local_path.exists():
        return None
    ts = local_path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def sync_file(
    bucket: storage.Bucket,
    gcs_path: str,
    local_dir: Path,
    required: bool,
) -> str:
    """単一ファイルを同期する。戻り値は操作の説明文字列。"""
    blob = bucket.blob(gcs_path)

    try:
        blob.reload()
    except Exception:
        if required:
            return f"[SKIP]  {gcs_path} — GCS に存在しない（必須ファイル）"
        return f"[SKIP]  {gcs_path} — GCS に存在しない"

    gcs_mtime = _parse_gcs_mtime(blob)
    filename = Path(gcs_path).name
    local_path = local_dir / filename
    local_mtime = _local_mtime(local_path)

    if local_mtime is not None and gcs_mtime is not None and local_mtime >= gcs_mtime:
        return f"[SKIP]  {gcs_path} — ローカルが最新（local={local_mtime.isoformat()}, gcs={gcs_mtime.isoformat()}）"

    local_dir.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    return f"[DL]    {gcs_path} → {local_path}"


def main() -> None:
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    local_dir = Path(LOCAL_SHION_DIR)

    print(f"GCS → ローカル同期開始: gs://{GCS_BUCKET}/shion/ → {local_dir}")

    for gcs_path, required in _SYNC_FILES:
        result = sync_file(bucket, gcs_path, local_dir, required)
        print(result)

    print("同期完了")


if __name__ == "__main__":
    main()
