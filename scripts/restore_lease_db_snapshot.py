#!/usr/bin/env python3
"""Restore lease_data.db from its GCS snapshot at Cloud Run startup.

api/cloudrun_db_snapshot.py が実行中プロセスから定期的にアップロードした
一貫性スナップショットを、コールドスタート直後・start_api_cloud_run.shの
seed_dir（バンドル復元、cp -Rn=非破壊）より前に復元する。これにより
GCSスナップショットがあればバンドルより優先され、無ければバンドル初期値に
自然にフォールバックする。demoモードは自動リセットが仕様のため対象外
（後段のdemoモード強制上書きブロックでどのみち上書きされる）。

失敗しても起動をブロックしないよう、常にexit 0で終了する。
"""

from __future__ import annotations

import os
from pathlib import Path


def _bucket_name() -> str:
    value = (os.environ.get("GCS_BUCKET", "tune-lease-55-data") or "").strip()
    if value.startswith("gs://"):
        value = value[5:]
    return value.split("/", 1)[0]


def _blob_name(db_filename: str) -> str:
    prefix = os.environ.get("GCS_SNAPSHOT_PREFIX", "cloudrun-snapshots/").strip("/") or "cloudrun-snapshots"
    return f"{prefix}/{db_filename}"


def _target_path() -> Path:
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    db_filename = os.environ.get("LEASE_DB_FILENAME", "lease_data.db")
    return data_dir / db_filename


def main() -> None:
    if os.environ.get("CLOUDRUN_DATA_MODE", "").strip().lower() == "demo":
        print("[RestoreLeaseDB] demoモードのためスキップ（バンドルの自動リセットが仕様）")
        return

    target = _target_path()
    if target.exists() and target.stat().st_size > 0:
        print(f"[RestoreLeaseDB] 既存データがあるためスキップ: {target}")
        return

    try:
        from google.cloud import storage
    except Exception as exc:
        print(f"[RestoreLeaseDB] google.cloud.storage が利用できないためスキップ: {exc}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".downloading")
    try:
        bucket_name = _bucket_name()
        blob_name = _blob_name(target.name)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        # 途中失敗で不完全なファイルがtargetに残ると、後続のバンドル復元
        # （cp -Rn、既存ファイル優先）がフォールバックできなくなる。一時ファイルへ
        # 落としてから成功時のみ os.replace で置き換える（アトミック）。
        blob.download_to_filename(str(tmp_path))
        os.replace(tmp_path, target)
        print(f"[RestoreLeaseDB] 復元完了: gs://{bucket_name}/{blob_name} → {target}")
    except Exception as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"[RestoreLeaseDB] 復元スキップ（非致命的）: {exc}")


if __name__ == "__main__":
    main()
