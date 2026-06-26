"""Write Cloud Run-originated user inputs back to GCS as append-only events."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


GCS_BUCKET = os.environ.get("GCS_BUCKET", "tune-lease-55-data")
GCS_INPUT_PREFIX = os.environ.get("GCS_INPUT_PREFIX", "cloudrun-inputs/")
LOCAL_FALLBACK_PATH = Path(__file__).parent.parent / "data" / "cloudrun_input_writeback_failures.jsonl"
PII_KEYS = {
    "name",
    "company_name",
    "address",
    "phone",
    "email",
    "representative",
    "hojin_name",
}


def _writeback_enabled() -> bool:
    raw = os.environ.get("CLOUDRUN_INPUT_WRITEBACK_ENABLED", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(os.environ.get("K_SERVICE", "").strip())


def _bucket_name() -> str:
    value = (GCS_BUCKET or "").strip()
    if value.startswith("gs://"):
        value = value[5:]
    return value.split("/", 1)[0]


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if str(key).lower() in PII_KEYS else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _event_path(now: datetime) -> str:
    prefix = GCS_INPUT_PREFIX.strip("/") or "cloudrun-inputs"
    return f"{prefix}/{now.date().isoformat()}/events.jsonl"


def _fallback(entry: dict[str, Any], reason: str) -> None:
    try:
        LOCAL_FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        fallback_entry = {**entry, "writeback_error": reason}
        with LOCAL_FALLBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(fallback_entry, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass


def build_cloudrun_input_event(
    *,
    event_type: str,
    surface: str,
    payload: dict[str, Any],
    source: str = "cloud-run",
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "event_id": str(uuid4()),
        "ts": now.isoformat(),
        "event_type": str(event_type or "input").strip() or "input",
        "surface": str(surface or "unknown").strip() or "unknown",
        "source": source,
        "cloud_run": {
            "service": os.environ.get("K_SERVICE", ""),
            "revision": os.environ.get("K_REVISION", ""),
            "configuration": os.environ.get("K_CONFIGURATION", ""),
        },
        "payload": _redact(payload or {}),
    }


def record_cloudrun_input_event(
    *,
    event_type: str,
    surface: str,
    payload: dict[str, Any],
    source: str = "cloud-run",
) -> dict[str, Any]:
    """Append a Cloud Run input event to GCS.

    This function never raises to callers. If writeback is disabled or GCS fails,
    it returns a status dict and records a local fallback when possible.
    """
    entry = build_cloudrun_input_event(
        event_type=event_type,
        surface=surface,
        payload=payload,
        source=source,
    )
    if not _writeback_enabled():
        return {"ok": False, "skipped": True, "reason": "writeback_disabled", "event_id": entry["event_id"]}

    try:
        from google.cloud import storage  # type: ignore[import-untyped]
        from scripts.gcs_lock import GCSLock

        bucket_name = _bucket_name()
        if not bucket_name:
            raise RuntimeError("GCS_BUCKET is empty")
        now = datetime.now(timezone.utc)
        gcs_path = _event_path(now)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"

        with GCSLock(bucket_name=bucket_name, target_file=gcs_path, ttl_seconds=30):
            try:
                current = blob.download_as_text()
            except Exception:
                current = ""
            blob.metadata = {"mtime": now.isoformat(), "event_type": "cloudrun_input"}
            blob.upload_from_string(current + line, content_type="application/jsonl; charset=utf-8")
        return {"ok": True, "skipped": False, "gcs_path": gcs_path, "event_id": entry["event_id"]}
    except Exception as exc:
        _fallback(entry, str(exc))
        return {"ok": False, "skipped": False, "reason": str(exc), "event_id": entry["event_id"]}
