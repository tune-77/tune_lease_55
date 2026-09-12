from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

from google.api_core.exceptions import NotFound
from api import cloudrun_writeback as writeback


def test_build_cloudrun_input_event_redacts_pii() -> None:
    event = writeback.build_cloudrun_input_event(
        event_type="score_calculated",
        surface="score",
        payload={
            "company_name": "秘密商事",
            "industry_sub": "情報サービス業",
            "nested": {"email": "a@example.com", "asset_name": "サーバー"},
        },
    )

    assert event["payload"]["company_name"] == "[REDACTED]"
    assert event["payload"]["industry_sub"] == "情報サービス業"
    assert event["payload"]["nested"]["email"] == "[REDACTED]"
    assert event["payload"]["nested"]["asset_name"] == "サーバー"


def test_build_cloudrun_input_event_redacts_free_text() -> None:
    event = writeback.build_cloudrun_input_event(
        event_type="judgment_feedback_created",
        surface="judgment_feedback",
        payload={"reason": "社名と個人名を含む理由", "note": "内部メモ", "score": 72},
    )

    assert event["payload"]["reason"] == "[REDACTED]"
    assert event["payload"]["note"] == "[REDACTED]"
    assert event["payload"]["score"] == 72


def test_record_cloudrun_input_event_skips_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.setenv("CLOUDRUN_INPUT_WRITEBACK_ENABLED", "false")

    result = writeback.record_cloudrun_input_event(
        event_type="test",
        surface="unit",
        payload={"x": 1},
    )

    assert result["ok"] is False
    assert result["skipped"] is True
    assert result["reason"] == "writeback_disabled"


def test_fallback_writes_jsonl(tmp_path, monkeypatch) -> None:
    fallback_path = tmp_path / "failures.jsonl"
    monkeypatch.setattr(writeback, "LOCAL_FALLBACK_PATH", fallback_path)
    entry = writeback.build_cloudrun_input_event(
        event_type="test",
        surface="unit",
        payload={"company_name": "秘密商事"},
    )

    writeback._fallback(entry, "boom")

    rows = [json.loads(line) for line in fallback_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["writeback_error"] == "boom"
    assert rows[0]["payload"]["company_name"] == "[REDACTED]"


class _FakeGCSLock:
    def __init__(self, **_kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


def _install_fake_gcs(monkeypatch, blob: MagicMock) -> MagicMock:
    try:
        from google.cloud import storage
    except ImportError:
        storage = ModuleType("google.cloud.storage")
        cloud_mod = sys.modules.get("google.cloud")
        if cloud_mod is None:
            cloud_mod = ModuleType("google.cloud")
            monkeypatch.setitem(sys.modules, "google.cloud", cloud_mod)
        setattr(cloud_mod, "storage", storage)
        monkeypatch.setitem(sys.modules, "google.cloud.storage", storage)

    client_cls = MagicMock()
    client_cls.return_value.bucket.return_value.blob.return_value = blob
    monkeypatch.setattr(storage, "Client", client_cls, raising=False)

    lock_mod = ModuleType("scripts.gcs_lock")
    lock_mod.GCSLock = _FakeGCSLock
    monkeypatch.setitem(sys.modules, "scripts.gcs_lock", lock_mod)
    return client_cls


def test_record_cloudrun_input_event_appends_with_generation_match(monkeypatch) -> None:
    monkeypatch.setenv("K_SERVICE", "lease-api")
    monkeypatch.delenv("CLOUDRUN_INPUT_WRITEBACK_ENABLED", raising=False)
    blob = MagicMock()
    blob.generation = 7
    blob.download_as_text.return_value = '{"event_id":"old"}\n'
    _install_fake_gcs(monkeypatch, blob)

    result = writeback.record_cloudrun_input_event(
        event_type="test",
        surface="unit",
        payload={"x": 1},
    )

    assert result["ok"] is True
    blob.reload.assert_called_once()
    blob.upload_from_string.assert_called_once()
    assert blob.upload_from_string.call_args.kwargs["if_generation_match"] == 7
    assert blob.upload_from_string.call_args.args[0].startswith('{"event_id":"old"}\n')


def test_record_cloudrun_input_event_treats_not_found_as_new_blob(monkeypatch) -> None:
    monkeypatch.setenv("K_SERVICE", "lease-api")
    blob = MagicMock()
    blob.reload.side_effect = NotFound("missing")
    _install_fake_gcs(monkeypatch, blob)

    result = writeback.record_cloudrun_input_event(
        event_type="test",
        surface="unit",
        payload={"x": 1},
    )

    assert result["ok"] is True
    blob.download_as_text.assert_not_called()
    assert blob.upload_from_string.call_args.kwargs["if_generation_match"] == 0


def test_record_cloudrun_input_event_does_not_overwrite_on_read_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("K_SERVICE", "lease-api")
    monkeypatch.setattr(writeback, "LOCAL_FALLBACK_PATH", tmp_path / "failures.jsonl")
    blob = MagicMock()
    blob.reload.side_effect = RuntimeError("temporary gcs failure")
    _install_fake_gcs(monkeypatch, blob)

    result = writeback.record_cloudrun_input_event(
        event_type="test",
        surface="unit",
        payload={"x": 1},
    )

    assert result["ok"] is False
    blob.upload_from_string.assert_not_called()
    rows = [
        json.loads(line)
        for line in writeback.LOCAL_FALLBACK_PATH.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["writeback_error"] == "temporary gcs failure"


def test_judgment_feedback_transaction_rejects_competing_successor(monkeypatch) -> None:
    monkeypatch.setenv("K_SERVICE", "lease-api")
    index_blob, daily_blob = MagicMock(), MagicMock()
    index_blob.generation = 7
    daily_blob.reload.side_effect = NotFound("missing")
    client = _install_fake_gcs(monkeypatch, index_blob)
    client.return_value.bucket.return_value.blob.side_effect = lambda name: index_blob if "judgment-feedback" in name else daily_blob
    root = {"candidate_id": "candidate-1", "case_id": "case-1", "review_id": 7, "event_id": "root", "supersedes_event_id": "", "feedback": "useful"}
    root_entry = writeback.build_cloudrun_input_event(event_type="judgment_asset_candidate_feedback", surface="screening", payload=root)
    index_blob.download_as_text.return_value = json.dumps(root_entry) + "\n"
    first = {**root, "event_id": "first", "supersedes_event_id": "root", "feedback": "neutral"}
    assert writeback.record_judgment_asset_feedback_event(first)["ok"] is True
    index_blob.download_as_text.return_value = index_blob.upload_from_string.call_args.args[0]
    second = {**root, "event_id": "second", "supersedes_event_id": "root", "feedback": "rejected"}
    result = writeback.record_judgment_asset_feedback_event(second)
    assert result["conflict"] is True
    assert result["current_event_id"] == "first"
    assert daily_blob.upload_from_string.call_count == 1
