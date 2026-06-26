#!/usr/bin/env python3
"""Download Cloud Run input event logs from GCS for local nightly processing."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from google.cloud import storage


GCS_BUCKET = os.environ.get("GCS_BUCKET", "tune-lease-55-data")
GCS_INPUT_PREFIX = os.environ.get("GCS_INPUT_PREFIX", "cloudrun-inputs/")
LOCAL_INPUT_DIR = Path(os.environ.get("LOCAL_CLOUDRUN_INPUT_DIR", "data/cloudrun_inputs"))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WIZARD_INPUT_LOG = PROJECT_ROOT / "data" / "wizard_input_log.jsonl"
RAG_FEEDBACK_LOG = PROJECT_ROOT / "data" / "rag_feedback_log.jsonl"
RAG_HIT_LOG = PROJECT_ROOT / "data" / "rag_hit_log.jsonl"
WIZARD_TRACKED_FIELDS = [
    "company_name",
    "nenshu",
    "op_profit",
    "acquisition_cost",
    "asset_name",
    "passion_text",
    "industry_detail",
    "asset_detail",
    "asset_purpose",
    "asset_location",
]


def _bucket_name() -> str:
    value = (GCS_BUCKET or "").strip()
    if value.startswith("gs://"):
        value = value[5:]
    return value.split("/", 1)[0]


def _date_range(days: int) -> Iterable[date]:
    today = datetime.now(timezone.utc).date()
    count = max(1, int(days))
    for offset in range(count):
        yield today - timedelta(days=offset)


def _event_blob_name(day: date) -> str:
    prefix = GCS_INPUT_PREFIX.strip("/") or "cloudrun-inputs"
    return f"{prefix}/{day.isoformat()}/events.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _append_jsonl_dedup(path: Path, rows: list[dict]) -> int:
    existing = _load_jsonl(path)
    merged = _merge_events(existing, rows)
    _write_jsonl(path, merged)
    return max(0, len(merged) - len(existing))


def _merge_events(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for row in existing + incoming:
        event_id = str(row.get("event_id") or "")
        key = event_id or json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    merged.sort(key=lambda row: str(row.get("ts") or ""))
    return merged


def _wizard_entry_from_event(event: dict) -> dict | None:
    if event.get("event_type") != "score_calculated":
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    empty = [field for field in WIZARD_TRACKED_FIELDS if not inputs.get(field)]
    return {
        "event_id": event.get("event_id"),
        "ts": event.get("ts"),
        "total_fields": len(WIZARD_TRACKED_FIELDS),
        "empty_count": len(empty),
        "empty_fields": empty,
        "surface": "cloudrun_score_calculate",
        "source": "cloudrun_input_writeback",
    }


def _rag_entries_from_event(event: dict) -> tuple[dict | None, dict | None]:
    if event.get("event_type") != "rag_feedback":
        return None, None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    if not payload:
        return None, None
    feedback = {
        **payload,
        "event_id": event.get("event_id"),
        "source": "cloudrun_input_writeback",
    }
    hit = {
        "event_id": event.get("event_id"),
        "ts": payload.get("ts") or event.get("ts"),
        "doc_id": payload.get("doc_id"),
        "obsidian_ref": payload.get("obsidian_ref"),
        "rating": payload.get("rating"),
        "surface": payload.get("surface") or event.get("surface"),
        "hit_type": "feedback_confirmed",
        "source": "cloudrun_input_writeback",
    }
    return feedback, hit


def materialize_events(events: list[dict]) -> dict[str, int]:
    wizard_rows = [row for event in events if (row := _wizard_entry_from_event(event))]
    rag_feedback_rows: list[dict] = []
    rag_hit_rows: list[dict] = []
    for event in events:
        feedback, hit = _rag_entries_from_event(event)
        if feedback:
            rag_feedback_rows.append(feedback)
        if hit:
            rag_hit_rows.append(hit)
    return {
        "wizard_new": _append_jsonl_dedup(WIZARD_INPUT_LOG, wizard_rows) if wizard_rows else 0,
        "rag_feedback_new": _append_jsonl_dedup(RAG_FEEDBACK_LOG, rag_feedback_rows) if rag_feedback_rows else 0,
        "rag_hit_new": _append_jsonl_dedup(RAG_HIT_LOG, rag_hit_rows) if rag_hit_rows else 0,
    }


def sync_day(bucket: storage.Bucket, day: date, local_dir: Path = LOCAL_INPUT_DIR) -> dict:
    blob_name = _event_blob_name(day)
    blob = bucket.blob(blob_name)
    try:
        text = blob.download_as_text()
    except Exception:
        return {"date": day.isoformat(), "downloaded": False, "events": 0, "path": str(local_dir / f"{day.isoformat()}.jsonl")}

    incoming: list[dict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            incoming.append(item)

    local_path = local_dir / f"{day.isoformat()}.jsonl"
    existing = _load_jsonl(local_path)
    merged = _merge_events(existing, incoming)
    _write_jsonl(local_path, merged)
    materialized = materialize_events(merged)
    return {
        "date": day.isoformat(),
        "downloaded": True,
        "events": len(merged),
        "new_events": max(0, len(merged) - len(existing)),
        "materialized": materialized,
        "path": str(local_path),
    }


def main() -> None:
    days = int(os.environ.get("CLOUDRUN_INPUT_SYNC_DAYS", "3"))
    bucket_name = _bucket_name()
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    print(f"Cloud Run入力同期開始: gs://{bucket_name}/{GCS_INPUT_PREFIX} → {LOCAL_INPUT_DIR}")
    for day in _date_range(days):
        result = sync_day(bucket, day, LOCAL_INPUT_DIR)
        status = "DL" if result["downloaded"] else "SKIP"
        print(f"[{status}] {result['date']} events={result['events']} path={result['path']}")
    print("同期完了")


if __name__ == "__main__":
    main()
