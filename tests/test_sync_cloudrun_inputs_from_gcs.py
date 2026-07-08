from __future__ import annotations

import json
import sqlite3
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
    archive_log = tmp_path / "archive.jsonl"
    wizard_log = tmp_path / "wizard.jsonl"
    rag_feedback_log = tmp_path / "rag_feedback.jsonl"
    rag_hit_log = tmp_path / "rag_hit.jsonl"
    screening_loop_log = tmp_path / "screening_loop.jsonl"
    local_db = tmp_path / "lease_data.db"
    monkeypatch.setattr(syncer, "CLOUDRUN_EVENT_ARCHIVE_LOG", archive_log)
    monkeypatch.setattr(syncer, "WIZARD_INPUT_LOG", wizard_log)
    monkeypatch.setattr(syncer, "RAG_FEEDBACK_LOG", rag_feedback_log)
    monkeypatch.setattr(syncer, "RAG_HIT_LOG", rag_hit_log)
    monkeypatch.setattr(syncer, "SCREENING_LOOP_FEEDBACK_LOG", screening_loop_log)
    monkeypatch.setattr(syncer, "LOCAL_LEASE_DB", local_db)

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
    assert result == {
        "all_events_new": 2,
        "wizard_new": 1,
        "rag_feedback_new": 1,
        "rag_hit_new": 1,
        "screening_loop_feedback_new": 0,
        "improvement_new": 0,
        "chat_new": 0,
        "shion_memory_usage_new": 0,
        "score_inputs_new": 1,
        "ocr_results_new": 0,
        "shion_reviews_new": 0,
        "shion_review_feedback_updated": 0,
    }
    assert wizard_rows[0]["surface"] == "cloudrun_score_calculated"
    assert "asset_name" in wizard_rows[0]["empty_fields"]
    assert rag_rows[0]["event_id"] == "rag-1"
    assert hit_rows[0]["hit_type"] == "feedback_confirmed"


def test_materialize_events_restores_shion_review_and_feedback_to_local_db(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(syncer, "CLOUDRUN_EVENT_ARCHIVE_LOG", tmp_path / "archive.jsonl")
    monkeypatch.setattr(syncer, "WIZARD_INPUT_LOG", tmp_path / "wizard.jsonl")
    monkeypatch.setattr(syncer, "RAG_FEEDBACK_LOG", tmp_path / "rag_feedback.jsonl")
    monkeypatch.setattr(syncer, "RAG_HIT_LOG", tmp_path / "rag_hit.jsonl")
    monkeypatch.setattr(syncer, "SCREENING_LOOP_FEEDBACK_LOG", tmp_path / "screening_loop.jsonl")
    local_db = tmp_path / "lease_data.db"
    monkeypatch.setattr(syncer, "LOCAL_LEASE_DB", local_db)

    events = [
        {
            "event_id": "review-evt-1",
            "ts": "2026-07-01T00:00:00Z",
            "event_type": "shion_screening_review",
            "surface": "screening",
            "payload": {
                "id": 42,
                "cloud_review_id": 42,
                "case_id": "C-001",
                "company_name": "[REDACTED]",
                "industry_major": "D 建設業",
                "industry_sub": "06 総合工事業",
                "sales_dept": "東京",
                "score": 68.5,
                "hantei": "条件付き承認",
                "q_risk": 41.2,
                "umap_anomaly_score": 12.3,
                "memory_refs": 2,
                "knowledge_refs": 1,
                "identity_used": True,
                "review_text": "条件付きなら銀行支援と物件保全を確認。",
                "form_snapshot": {"company_name": "[REDACTED]", "asset_name": "建機"},
                "result_snapshot": {"score_base": 68.5},
            },
        },
        {
            "event_id": "feedback-evt-1",
            "ts": "2026-07-01T00:01:00Z",
            "event_type": "shion_screening_review_feedback",
            "surface": "screening",
            "payload": {
                "id": 42,
                "cloud_review_id": 42,
                "user_feedback": "useful",
            },
        },
    ]

    result = syncer.materialize_events(events)

    assert result["shion_reviews_new"] == 1
    assert result["shion_review_feedback_updated"] == 1
    with sqlite3.connect(local_db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM shion_screening_reviews").fetchone()
        assert row["cloud_review_id"] == "42"
        assert row["cloud_event_id"] == "review-evt-1"
        assert row["industry_sub"] == "06 総合工事業"
        assert row["user_feedback"] == "useful"
        assert "銀行支援" in row["review_text"]


def test_materialize_events_appends_screening_loop_feedback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(syncer, "CLOUDRUN_EVENT_ARCHIVE_LOG", tmp_path / "archive.jsonl")
    monkeypatch.setattr(syncer, "WIZARD_INPUT_LOG", tmp_path / "wizard.jsonl")
    monkeypatch.setattr(syncer, "RAG_FEEDBACK_LOG", tmp_path / "rag_feedback.jsonl")
    monkeypatch.setattr(syncer, "RAG_HIT_LOG", tmp_path / "rag_hit.jsonl")
    screening_loop_log = tmp_path / "screening_loop.jsonl"
    monkeypatch.setattr(syncer, "SCREENING_LOOP_FEEDBACK_LOG", screening_loop_log)
    monkeypatch.setattr(syncer, "LOCAL_LEASE_DB", tmp_path / "lease_data.db")

    result = syncer.materialize_events([
        {
            "event_id": "loop-1",
            "ts": "2026-07-01T00:02:00Z",
            "event_type": "screening_loop_feedback",
            "surface": "screening",
            "payload": {
                "target": "issue",
                "rating": "合っている",
                "score": 70,
            },
        }
    ])

    rows = [json.loads(line) for line in screening_loop_log.read_text(encoding="utf-8").splitlines()]
    assert result["screening_loop_feedback_new"] == 1
    assert rows[0]["event_id"] == "loop-1"
    assert rows[0]["source"] == "cloudrun_input_writeback"


def test_materialize_events_appends_improvement_chat_and_memory_usage(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(syncer, "CLOUDRUN_EVENT_ARCHIVE_LOG", tmp_path / "archive.jsonl")
    monkeypatch.setattr(syncer, "WIZARD_INPUT_LOG", tmp_path / "wizard.jsonl")
    monkeypatch.setattr(syncer, "RAG_FEEDBACK_LOG", tmp_path / "rag_feedback.jsonl")
    monkeypatch.setattr(syncer, "RAG_HIT_LOG", tmp_path / "rag_hit.jsonl")
    monkeypatch.setattr(syncer, "SCREENING_LOOP_FEEDBACK_LOG", tmp_path / "screening_loop.jsonl")
    improvement_log = tmp_path / "cloudrun_improvement.jsonl"
    chat_log = tmp_path / "cloudrun_chat.jsonl"
    memory_usage_log = tmp_path / "shion_memory_usage.jsonl"
    monkeypatch.setattr(syncer, "CLOUDRUN_IMPROVEMENT_LOG", improvement_log)
    monkeypatch.setattr(syncer, "CLOUDRUN_CHAT_LOG", chat_log)
    monkeypatch.setattr(syncer, "SHION_MEMORY_USAGE_LOG", memory_usage_log)
    monkeypatch.setattr(syncer, "LOCAL_LEASE_DB", tmp_path / "lease_data.db")

    result = syncer.materialize_events([
        {
            "event_id": "improve-1",
            "ts": "2026-07-01T00:02:00Z",
            "event_type": "improvement_note",
            "surface": "chat_improvement",
            "payload": {"title": "改善ログ", "body": "Cloud Run入力を改善ログに流す"},
        },
        {
            "event_id": "chat-1",
            "ts": "2026-07-01T00:03:00Z",
            "event_type": "chat_exchange",
            "surface": "next_chat_rag",
            "payload": {
                "user_id": "default",
                "category": "lease",
                "response_mode": "shion",
                "user_message": "補助金について教えて",
                "assistant_reply": "対象設備と公募要領を確認します。",
                "metadata": {"knowledge_refs": 2},
            },
        },
        {
            "event_id": "memory-1",
            "ts": "2026-07-01T00:04:00Z",
            "event_type": "shion_memory_usage",
            "surface": "api_chat",
            "payload": {
                "ts": "2026-07-01T00:04:00Z",
                "route": "next_chat",
                "refs": ["memory/a.md", "memory/b.md"],
            },
        },
    ])

    improvement_rows = [json.loads(line) for line in improvement_log.read_text(encoding="utf-8").splitlines()]
    chat_rows = [json.loads(line) for line in chat_log.read_text(encoding="utf-8").splitlines()]
    memory_rows = [json.loads(line) for line in memory_usage_log.read_text(encoding="utf-8").splitlines()]
    assert result["improvement_new"] == 1
    assert result["chat_new"] == 1
    assert result["shion_memory_usage_new"] == 1
    assert improvement_rows[0]["source"] == "cloudrun_input_writeback"
    assert "Cloud Run入力" in improvement_rows[0]["body"]
    assert chat_rows[0]["category"] == "lease"
    assert chat_rows[0]["metadata"]["knowledge_refs"] == 2
    assert memory_rows[0]["ref_count"] == 2


def test_materialize_events_restores_score_full_and_ocr_to_quarantine_db(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(syncer, "CLOUDRUN_EVENT_ARCHIVE_LOG", tmp_path / "archive.jsonl")
    monkeypatch.setattr(syncer, "WIZARD_INPUT_LOG", tmp_path / "wizard.jsonl")
    monkeypatch.setattr(syncer, "RAG_FEEDBACK_LOG", tmp_path / "rag_feedback.jsonl")
    monkeypatch.setattr(syncer, "RAG_HIT_LOG", tmp_path / "rag_hit.jsonl")
    monkeypatch.setattr(syncer, "SCREENING_LOOP_FEEDBACK_LOG", tmp_path / "screening_loop.jsonl")
    local_db = tmp_path / "return.db"
    monkeypatch.setattr(syncer, "LOCAL_LEASE_DB", local_db)

    result = syncer.materialize_events([
        {
            "event_id": "score-full-1",
            "ts": "2026-07-01T00:03:00Z",
            "event_type": "score_full_calculated",
            "surface": "screening",
            "payload": {
                "case_id": "case-1",
                "inputs": {"industry_sub": "06 総合工事業", "nenshu": 200, "op_profit": 15},
                "result": {"score_base": 71.2, "hantei": "承認", "industry_sub": "06 総合工事業"},
            },
        },
        {
            "event_id": "ocr-1",
            "ts": "2026-07-01T00:04:00Z",
            "event_type": "ocr_extracted",
            "surface": "ocr",
            "payload": {
                "doc_type": "financial",
                "content_type": "application/pdf",
                "result": {
                    "nenshu": 200,
                    "op_profit": 15,
                    "detected_fields": ["nenshu", "op_profit"],
                    "missing_fields": ["net_assets"],
                    "confidence": 0.86,
                },
            },
        },
    ])

    assert result["score_inputs_new"] == 1
    assert result["ocr_results_new"] == 1
    with sqlite3.connect(local_db) as conn:
        conn.row_factory = sqlite3.Row
        score_row = conn.execute("SELECT * FROM cloudrun_score_inputs").fetchone()
        ocr_row = conn.execute("SELECT * FROM cloudrun_ocr_results").fetchone()
        assert score_row["event_type"] == "score_full_calculated"
        assert score_row["score"] == 71.2
        assert json.loads(score_row["inputs_json"])["nenshu"] == 200
        assert ocr_row["doc_type"] == "financial"
        assert ocr_row["confidence"] == 0.86
        assert "op_profit" in json.loads(ocr_row["detected_fields"])


def test_download_event_text_treats_missing_object_as_not_found() -> None:
    blob = MagicMock()
    blob.download_as_text.side_effect = Exception(
        "404 GET https://storage.googleapis.com/... No such object: bucket/x"
    )
    bucket = MagicMock()
    bucket.blob.return_value = blob

    text, reason = syncer._download_event_text(bucket, "bucket", "x")

    assert text is None
    assert reason == syncer.NOT_FOUND_REASON


def test_download_event_text_gcloud_not_found(monkeypatch) -> None:
    import subprocess

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1, ["gcloud"], stderr="ERROR: The following URLs matched no objects or files"
        )

    monkeypatch.setattr(syncer.subprocess, "run", fake_run)

    text, reason = syncer._download_event_text(None, "bucket", "x")

    assert text is None
    assert reason == syncer.NOT_FOUND_REASON


def test_download_event_text_reports_real_errors(monkeypatch) -> None:
    import subprocess

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1, ["gcloud"], stderr="ERROR: (gcloud.storage.cat) HTTPError 403: アクセス権がありません"
        )

    monkeypatch.setattr(syncer.subprocess, "run", fake_run)

    text, reason = syncer._download_event_text(None, "bucket", "x")

    assert text is None
    assert reason.startswith("error:")
    assert "403" in reason


def test_download_event_text_reports_missing_gcloud(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("No such file or directory: 'gcloud'")

    monkeypatch.setattr(syncer.subprocess, "run", fake_run)

    text, reason = syncer._download_event_text(None, "bucket", "x")

    assert text is None
    assert reason.startswith("error:")
    assert "gcloud" in reason


def test_main_exits_nonzero_when_download_errors(monkeypatch) -> None:
    import pytest

    monkeypatch.setattr(
        syncer,
        "sync_day",
        lambda *args, **kwargs: {
            "date": "2026-07-08",
            "downloaded": False,
            "events": 0,
            "path": "p",
            "reason": "error: gcloud コマンドが見つかりません",
        },
    )

    with pytest.raises(SystemExit) as excinfo:
        syncer.main()

    assert excinfo.value.code == 1


def test_main_exits_zero_when_days_have_no_events(monkeypatch) -> None:
    monkeypatch.setattr(
        syncer,
        "sync_day",
        lambda *args, **kwargs: {
            "date": "2026-07-08",
            "downloaded": False,
            "events": 0,
            "path": "p",
            "reason": syncer.NOT_FOUND_REASON,
        },
    )

    syncer.main()
