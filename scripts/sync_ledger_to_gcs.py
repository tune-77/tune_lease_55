#!/usr/bin/env python3
"""ローカル ledger.jsonl（macOS想定パス）を GCS へミラーする。

Cloud Run コンテナは Path.home() がローカル macOS のログパスに届かないため、
api/main.py の適用済み判定（_gcs_ledger_entries）が読む唯一のソース。
ここでの書き込みが失敗しても日次パイプライン全体は止めない（呼び出し側で || true）。

Usage:
    python scripts/sync_ledger_to_gcs.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

GCS_BUCKET = os.environ.get("GCS_BUCKET", "tune-lease-55-data")
GCS_LEDGER_MIRROR_PATH = os.environ.get("GCS_LEDGER_MIRROR_PATH", "ledger/ledger.jsonl").strip("/")
LEDGER_PATH = Path(os.environ.get("LEDGER_PATH") or (Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl"))


def _bucket_name() -> str:
    value = (GCS_BUCKET or "").strip()
    if value.startswith("gs://"):
        value = value[5:]
    return value.split("/", 1)[0]


def sync() -> bool:
    if not LEDGER_PATH.exists():
        print(f"[sync_ledger_to_gcs] ledger not found, skip: {LEDGER_PATH}")
        return True
    bucket_name = _bucket_name()
    if not bucket_name:
        print("[sync_ledger_to_gcs] GCS_BUCKET が未設定のためスキップ")
        return True
    try:
        from google.cloud import storage  # type: ignore[import-untyped]

        client = storage.Client()
        blob = client.bucket(bucket_name).blob(GCS_LEDGER_MIRROR_PATH)
        blob.upload_from_filename(str(LEDGER_PATH), content_type="application/jsonl; charset=utf-8")
        print(f"[sync_ledger_to_gcs] uploaded {LEDGER_PATH} -> gs://{bucket_name}/{GCS_LEDGER_MIRROR_PATH}")
        return True
    except Exception as exc:
        print(f"[sync_ledger_to_gcs] 失敗（次回リトライ想定・パイプラインは継続）: {exc}")
        return False


if __name__ == "__main__":
    sys.exit(0 if sync() else 1)
