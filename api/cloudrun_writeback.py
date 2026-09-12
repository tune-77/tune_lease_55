"""Write Cloud Run-originated user inputs back to GCS as append-only events."""

from __future__ import annotations

import json
import os
import hashlib
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
    "note",
    "reason",
    "passion_text",
    "asset_location",
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


def _judgment_feedback_path(case_id: str, review_id: Any) -> str:
    key = hashlib.sha256(f"{case_id}|{review_id}".encode()).hexdigest()
    return f"{GCS_INPUT_PREFIX.strip('/') or 'cloudrun-inputs'}/judgment-feedback/{key}.jsonl"


def read_judgment_asset_feedback_events(case_id: str, review_id: Any) -> list[dict[str, Any]]:
    if not _writeback_enabled() or not str(case_id or "").strip():
        return []
    try:
        from google.api_core.exceptions import NotFound  # type: ignore[import-untyped]
        from google.cloud import storage  # type: ignore[import-untyped]
        blob = storage.Client().bucket(_bucket_name()).blob(_judgment_feedback_path(case_id, review_id))
        try:
            text = blob.download_as_text()
        except NotFound:
            return []
        return [row for line in text.splitlines() if line.strip() for row in [json.loads(line)] if isinstance(row, dict)]
    except Exception:
        return []


def record_judgment_asset_feedback_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and durably append one feedback event under a shared review lock."""
    if not _writeback_enabled():
        return {"ok": False, "skipped": True, "reason": "writeback_disabled"}
    entry = build_cloudrun_input_event(event_type="judgment_asset_candidate_feedback", surface="screening", payload=payload)
    try:
        from google.api_core.exceptions import NotFound  # type: ignore[import-untyped]
        from google.cloud import storage  # type: ignore[import-untyped]
        from scripts.gcs_lock import GCSLock
        path = _judgment_feedback_path(str(payload.get("case_id") or ""), payload.get("review_id"))
        bucket = storage.Client().bucket(_bucket_name())
        blob = bucket.blob(path)
        with GCSLock(bucket_name=_bucket_name(), target_file=path, ttl_seconds=30):
            try:
                blob.reload(); current = blob.download_as_text(); generation = blob.generation
            except NotFound:
                current, generation = "", 0
            entries = [row for line in current.splitlines() if line.strip() for row in [json.loads(line)] if isinstance(row, dict)]
            payloads = [row.get("payload") for row in entries if isinstance(row.get("payload"), dict)]
            same = next((row for row in payloads if row.get("event_id") == payload.get("event_id")), None)
            if same:
                return {"ok": True, "duplicate": True}
            related = [row for row in payloads if row.get("candidate_id") == payload.get("candidate_id")]
            superseded = {row.get("supersedes_event_id") for row in related if row.get("supersedes_event_id")}
            heads = [row for row in related if row.get("event_id") not in superseded]
            predecessor = next((row for row in heads if row.get("event_id") == payload.get("supersedes_event_id")), None)
            if (payload.get("supersedes_event_id") and not predecessor) or (not payload.get("supersedes_event_id") and heads):
                return {"ok": False, "conflict": True, "current_event_id": str(heads[-1].get("event_id") or "") if heads else ""}
            daily_blob = bucket.blob(_event_path(datetime.now(timezone.utc)))
            try:
                daily_blob.reload(); daily_text = daily_blob.download_as_text(); daily_generation = daily_blob.generation
            except NotFound:
                daily_text, daily_generation = "", 0
            daily_blob.upload_from_string(daily_text + json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n", content_type="application/jsonl; charset=utf-8", if_generation_match=daily_generation)
            blob.upload_from_string(current + json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n", content_type="application/jsonl; charset=utf-8", if_generation_match=generation)
        return {"ok": True, "duplicate": False}
    except Exception as exc:
        return {"ok": False, "skipped": False, "reason": str(exc)}


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
        from google.api_core.exceptions import NotFound  # type: ignore[import-untyped]
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
                blob.reload()
                current = blob.download_as_text()
                generation = blob.generation
            except NotFound:
                current = ""
                generation = 0
            blob.metadata = {"mtime": now.isoformat(), "event_type": "cloudrun_input"}
            blob.upload_from_string(
                current + line,
                content_type="application/jsonl; charset=utf-8",
                if_generation_match=generation,
            )
        return {"ok": True, "skipped": False, "gcs_path": gcs_path, "event_id": entry["event_id"]}
    except Exception as exc:
        _fallback(entry, str(exc))
        return {"ok": False, "skipped": False, "reason": str(exc), "event_id": entry["event_id"]}
