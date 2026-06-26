from __future__ import annotations

import json
import sys
from datetime import date
from types import ModuleType
from unittest.mock import MagicMock


def _setup_gcs_mock() -> MagicMock:
    storage_mock = MagicMock()
    cloud_mod = sys.modules.get("google.cloud")
    if cloud_mod is None:
        cloud_mod = ModuleType("google.cloud")
        sys.modules["google.cloud"] = cloud_mod
    sys.modules["google.cloud.storage"] = storage_mock
    setattr(cloud_mod, "storage", storage_mock)
    return storage_mock


_setup_gcs_mock()

from scripts import sync_cloudrun_inputs_from_gcs as syncer  # noqa: E402


def test_merge_events_dedupes_by_event_id() -> None:
    existing = [{"event_id": "a", "ts": "2026-06-26T00:00:01Z", "x": 1}]
    incoming = [
        {"event_id": "a", "ts": "2026-06-26T00:00:01Z", "x": 2},
        {"event_id": "b", "ts": "2026-06-26T00:00:02Z", "x": 3},
    ]

    merged = syncer._merge_events(existing, incoming)

    assert [row["event_id"] for row in merged] == ["a", "b"]
    assert merged[0]["x"] == 1


def test_sync_day_downloads_and_merges_jsonl(tmp_path) -> None:
    day = date(2026, 6, 26)
    local_file = tmp_path / "2026-06-26.jsonl"
    local_file.write_text(json.dumps({"event_id": "old", "ts": "2026-06-26T00:00:00Z"}) + "\n", encoding="utf-8")

    blob = MagicMock()
    blob.download_as_text.return_value = "\n".join(
        [
            json.dumps({"event_id": "new", "ts": "2026-06-26T00:00:02Z"}),
            json.dumps({"event_id": "old", "ts": "2026-06-26T00:00:00Z"}),
        ]
    )
    bucket = MagicMock()
    bucket.blob.return_value = blob

    result = syncer.sync_day(bucket, day, tmp_path)

    rows = [json.loads(line) for line in local_file.read_text(encoding="utf-8").splitlines()]
    assert result["downloaded"] is True
    assert result["events"] == 2
    assert result["new_events"] == 1
    assert [row["event_id"] for row in rows] == ["old", "new"]
    bucket.blob.assert_called_once_with("cloudrun-inputs/2026-06-26/events.jsonl")


def test_materialize_events_writes_existing_pipeline_logs(tmp_path, monkeypatch) -> None:
    wizard_log = tmp_path / "wizard.jsonl"
    rag_feedback_log = tmp_path / "rag_feedback.jsonl"
    rag_hit_log = tmp_path / "rag_hit.jsonl"
    monkeypatch.setattr(syncer, "WIZARD_INPUT_LOG", wizard_log)
    monkeypatch.setattr(syncer, "RAG_FEEDBACK_LOG", rag_feedback_log)
    monkeypatch.setattr(syncer, "RAG_HIT_LOG", rag_hit_log)

    events = [
        {
            "event_id": "score-1",
            "ts": "2026-06-26T00:00:00Z",
            "event_type": "score_calculated",
            "surface": "score_calculate",
            "payload": {
                "inputs": {
                    "company_name": "[REDACTED]",
                    "nenshu": 1000,
                    "asset_name": "",
                }
            },
        },
        {
            "event_id": "rag-1",
            "ts": "2026-06-26T00:01:00Z",
            "event_type": "rag_feedback",
            "surface": "next_chat_rag",
            "payload": {
                "ts": "2026-06-26T00:01:00Z",
                "query": "補助金",
                "doc_id": "doc-1",
                "obsidian_ref": "note.md",
                "rating": "good",
                "surface": "next_chat_rag",
            },
        },
    ]

    result = syncer.materialize_events(events)

    wizard_rows = [json.loads(line) for line in wizard_log.read_text(encoding="utf-8").splitlines()]
    rag_rows = [json.loads(line) for line in rag_feedback_log.read_text(encoding="utf-8").splitlines()]
    hit_rows = [json.loads(line) for line in rag_hit_log.read_text(encoding="utf-8").splitlines()]
    assert result == {"wizard_new": 1, "rag_feedback_new": 1, "rag_hit_new": 1}
    assert wizard_rows[0]["surface"] == "cloudrun_score_calculate"
    assert "asset_name" in wizard_rows[0]["empty_fields"]
    assert rag_rows[0]["event_id"] == "rag-1"
    assert hit_rows[0]["hit_type"] == "feedback_confirmed"
