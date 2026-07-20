#!/usr/bin/env python3
"""Restore the cumulative prompt_feedback_log.jsonl snapshot from GCS at Cloud Run startup.

Cloud Runのローカルディスクはコールドスタートのたびに空になるため、夜間パイプライン
（scripts/sync_cloudrun_inputs_from_gcs.py）がアップロードした累積スナップショットを
起動時にここで復元する。失敗しても起動をブロックしないよう、常にexit 0で終了する。
"""

from __future__ import annotations

import os
from pathlib import Path


def _bucket_name() -> str:
    value = (os.environ.get("GCS_BUCKET", "tune-lease-55-data") or "").strip()
    if value.startswith("gs://"):
        value = value[5:]
    return value.split("/", 1)[0]


def _snapshot_blob_name() -> str:
    prefix = os.environ.get("GCS_SNAPSHOT_PREFIX", "cloudrun-snapshots/").strip("/") or "cloudrun-snapshots"
    return f"{prefix}/prompt_feedback_log.jsonl"


def _target_path() -> Path:
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    return data_dir / "prompt_feedback_log.jsonl"


def main() -> None:
    target = _target_path()
    if target.exists() and target.stat().st_size > 0:
        print(f"[RestorePromptFeedback] 既存データがあるためスキップ: {target}")
        return

    try:
        from google.cloud import storage
    except Exception as exc:
        print(f"[RestorePromptFeedback] google.cloud.storage が利用できないためスキップ: {exc}")
        return

    try:
        bucket_name = _bucket_name()
        blob_name = _snapshot_blob_name()
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        text = blob.download_as_text()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"[RestorePromptFeedback] 復元完了: gs://{bucket_name}/{blob_name} → {target}")
    except Exception as exc:
        print(f"[RestorePromptFeedback] 復元スキップ（非致命的）: {exc}")


if __name__ == "__main__":
    main()
