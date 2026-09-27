from __future__ import annotations

import json
from pathlib import Path

import pytest

from api import jev_memory_review as review


def _write_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "shadow_complete",
                "generated_at": "2026-09-27T12:00:00",
                "model": "jev-test",
                "confidence_threshold": 0.7,
                "results": [
                    {
                        "memory_id": "mem_1",
                        "shadow_rank": 1,
                        "content_preview": "中古流通を確認する",
                        "used_count": 42,
                        "composite_priority": 0.737,
                        "shadow_route": "human_review_low_confidence",
                        "scores": {
                            "time_sensitivity": 0.8,
                            "stale_harm": 0.6,
                            "durable_value": 0.2,
                            "review_information_gain": 0.7,
                        },
                        "confidence": {"minimum": 0.23},
                        "projected": {"semantic_tags": ["asset_resale"]},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    report_path = tmp_path / "report.json"
    state_path = tmp_path / "state.json"
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(review, "REPORT_PATHS", (report_path,))
    monkeypatch.setattr(review, "STATE_PATH", state_path)
    monkeypatch.setattr(review, "AUDIT_PATH", audit_path)
    return report_path, state_path, audit_path


def test_review_queue_exposes_jev_scores_and_pending_human_decision(tmp_path, monkeypatch):
    report_path, _, _ = _paths(tmp_path, monkeypatch)
    _write_report(report_path)

    payload = review.get_review_queue()

    assert payload["available"] is True
    assert payload["summary"] == {"total": 1, "pending": 1, "by_decision": {"pending": 1}}
    item = payload["items"][0]
    assert item["scores"]["stale_harm"] == 0.6
    assert item["confidence"]["minimum"] == 0.23
    assert item["semantic_tags"] == ["asset_resale"]
    assert item["human_decision"] == "pending"


def test_human_decision_is_saved_separately_without_changing_report(tmp_path, monkeypatch):
    report_path, state_path, audit_path = _paths(tmp_path, monkeypatch)
    _write_report(report_path)
    original_report = report_path.read_text(encoding="utf-8")

    item = review.save_human_decision("mem_1", decision="revise", note="現在の条件を確認する")

    assert item["human_decision"] == "revise"
    assert report_path.read_text(encoding="utf-8") == original_report
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["reviews"]["mem_1"]["decision"] == "revise"
    assert json.loads(audit_path.read_text(encoding="utf-8"))["memory_id"] == "mem_1"
    assert review.get_review_queue()["summary"]["pending"] == 0


def test_unknown_memory_cannot_be_reviewed(tmp_path, monkeypatch):
    report_path, _, _ = _paths(tmp_path, monkeypatch)
    _write_report(report_path)

    with pytest.raises(KeyError):
        review.save_human_decision("mem_missing", decision="held")


def test_missing_report_returns_explicit_empty_state(tmp_path, monkeypatch):
    _paths(tmp_path, monkeypatch)

    payload = review.get_review_queue()

    assert payload["available"] is False
    assert payload["status"] == "report_missing"
    assert payload["items"] == []


def test_delete_review_candidate_hides_only_review_row(tmp_path, monkeypatch):
    report_path, state_path, audit_path = _paths(tmp_path, monkeypatch)
    _write_report(report_path)
    original_report = report_path.read_text(encoding="utf-8")

    result = review.delete_review_candidate("mem_1")

    assert result["memory_deleted"] is False
    assert review.get_review_queue()["items"] == []
    assert report_path.read_text(encoding="utf-8") == original_report
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["reviews"]["mem_1"]["deleted_at"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["action"] == "delete_review_candidate"


def test_unknown_review_candidate_cannot_be_deleted(tmp_path, monkeypatch):
    report_path, _, _ = _paths(tmp_path, monkeypatch)
    _write_report(report_path)

    with pytest.raises(KeyError):
        review.delete_review_candidate("mem_missing")


def test_decision_cannot_resurrect_candidate_deleted_after_queue_read(tmp_path, monkeypatch):
    report_path, state_path, _ = _paths(tmp_path, monkeypatch)
    _write_report(report_path)
    original_get_queue = review.get_review_queue

    def queue_then_delete():
        queue = original_get_queue()
        state_path.write_text(
            json.dumps({"reviews": {"mem_1": {"deleted_at": "2026-09-27T13:00:00"}}}),
            encoding="utf-8",
        )
        return queue

    monkeypatch.setattr(review, "get_review_queue", queue_then_delete)

    with pytest.raises(KeyError):
        review.save_human_decision("mem_1", decision="retain")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["reviews"]["mem_1"]["deleted_at"]


def test_cloudrun_state_uses_durable_gcs_store(monkeypatch):
    durable = {"reviews": {"mem_1": {"decision": "retain"}}}
    written = []
    monkeypatch.setattr(review, "_cloudrun_state_enabled", lambda: True)
    monkeypatch.setattr(review, "_read_gcs_state", lambda: durable)
    monkeypatch.setattr(review, "_write_gcs_state", written.append)

    state = review.load_state()
    review._save_state(state)

    assert state["reviews"]["mem_1"]["decision"] == "retain"
    assert written and written[0]["reviews"]["mem_1"]["decision"] == "retain"


def test_note_over_limit_is_rejected_without_silent_truncation(tmp_path, monkeypatch):
    report_path, state_path, _ = _paths(tmp_path, monkeypatch)
    _write_report(report_path)

    with pytest.raises(ValueError, match="1000"):
        review.save_human_decision("mem_1", decision="held", note="x" * 1001)

    assert not state_path.exists()
