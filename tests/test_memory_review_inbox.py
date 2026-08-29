import json
import threading
import time
from pathlib import Path

import pytest

from api import memory_review_inbox as inbox


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_list_inbox_overlays_review_state(tmp_path, monkeypatch):
    source = tmp_path / "candidates.jsonl"
    _write_jsonl(source, [{"id": "a1", "claim": "採用したい記憶候補", "candidate_type": "judgment_rule"}])
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviews": {
                    "test_source__a1": {
                        "status": "adopted",
                        "note": "使う",
                        "edited_claim": "",
                        "reviewed_at": "2026-08-09T10:00:00",
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(inbox, "SOURCE_PATHS", {"test_source": source})
    monkeypatch.setattr(inbox, "REVIEW_STATE_PATH", state_path)

    payload = inbox.list_inbox(status="all")

    assert payload["summary"]["total"] == 1
    assert payload["summary"]["open"] == 0
    assert payload["items"][0]["status"] == "adopted"
    assert payload["items"][0]["note"] == "使う"


def test_list_inbox_auto_rejects_similar_unreviewed_candidates(tmp_path, monkeypatch):
    source = tmp_path / "candidates.jsonl"
    _write_jsonl(
        source,
        [
            {
                "id": "old",
                "claim": "雑談の相槌だけで審査行動を変えないため判断資産にしない",
                "candidate_type": "judgment_rule",
                "topic": "memory_health",
            },
            {
                "id": "new",
                "claim": "雑談の相槌だけで審査行動を変えないため判断資産にしない。",
                "candidate_type": "judgment_rule",
                "topic": "memory_health",
            },
        ],
    )
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviews": {
                    "test_source__old": {
                        "status": "rejected",
                        "note": "ノイズ",
                        "edited_claim": "",
                        "reviewed_at": "2026-08-25T10:00:00",
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(inbox, "SOURCE_PATHS", {"test_source": source})
    monkeypatch.setattr(inbox, "REVIEW_STATE_PATH", state_path)

    payload = inbox.list_inbox(status="candidate")

    assert payload["summary"]["by_status"]["rejected"] == 2
    assert payload["summary"]["open"] == 0
    assert payload["filtered_total"] == 0

    rejected_payload = inbox.list_inbox(status="rejected")
    auto_rejected = next(item for item in rejected_payload["items"] if item["source_item_id"] == "new")
    assert auto_rejected["auto_rejected"] is True
    assert auto_rejected["auto_reject_matched_inbox_id"] == "test_source__old"


def test_review_candidate_persists_rejection_pattern_and_forgets_on_override(tmp_path, monkeypatch):
    source = tmp_path / "candidates.jsonl"
    _write_jsonl(source, [{"id": "a1", "claim": "同じノイズ候補を次回から隠す", "candidate_type": "noise"}])
    state_path = tmp_path / "state.json"
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(inbox, "SOURCE_PATHS", {"test_source": source})
    monkeypatch.setattr(inbox, "REVIEW_STATE_PATH", state_path)
    monkeypatch.setattr(inbox, "REVIEW_AUDIT_PATH", audit_path)

    inbox.review_candidate("test_source__a1", decision="rejected")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["rejection_patterns"][0]["inbox_id"] == "test_source__a1"

    inbox.review_candidate("test_source__a1", decision="held")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["rejection_patterns"] == []


def test_list_inbox_reports_weekly_auto_reject_review_policy(tmp_path, monkeypatch):
    source = tmp_path / "candidates.jsonl"
    _write_jsonl(source, [{"id": "a1", "claim": "候補"}])
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviews": {"test_source__old": {"status": "rejected"}},
                "rejection_patterns": [{"inbox_id": "pattern_old", "text_hash": "abc"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(inbox, "SOURCE_PATHS", {"test_source": source})
    monkeypatch.setattr(inbox, "REVIEW_STATE_PATH", state_path)

    policy = inbox.list_inbox(status="all")["auto_reject_review_policy"]

    assert policy["cadence"] == "weekly"
    assert policy["review_weekday"] == "Monday"
    assert policy["pattern_count"] == 2
    assert policy["persisted_pattern_count"] == 1
    assert policy["rejected_review_count"] == 1


def test_review_candidate_writes_state_and_audit(tmp_path, monkeypatch):
    source = tmp_path / "candidates.jsonl"
    _write_jsonl(source, [{"id": "a1", "claim": "修正したい記憶候補"}])
    state_path = tmp_path / "state.json"
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(inbox, "SOURCE_PATHS", {"test_source": source})
    monkeypatch.setattr(inbox, "REVIEW_STATE_PATH", state_path)
    monkeypatch.setattr(inbox, "REVIEW_AUDIT_PATH", audit_path)

    item = inbox.review_candidate(
        "test_source__a1",
        decision="revised",
        note="表現を直す",
        edited_claim="修正後の記憶候補",
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert item["status"] == "revised"
    assert state["reviews"]["test_source__a1"]["edited_claim"] == "修正後の記憶候補"
    assert json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])["status"] == "revised"


def test_review_candidate_concurrent_reviews_do_not_lose_each_other(tmp_path, monkeypatch):
    source = tmp_path / "candidates.jsonl"
    _write_jsonl(
        source,
        [
            {"id": "a1", "claim": "並行レビュー対象A"},
            {"id": "a2", "claim": "並行レビュー対象B"},
        ],
    )
    state_path = tmp_path / "state.json"
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(inbox, "SOURCE_PATHS", {"test_source": source})
    monkeypatch.setattr(inbox, "REVIEW_STATE_PATH", state_path)
    monkeypatch.setattr(inbox, "REVIEW_AUDIT_PATH", audit_path)

    original_save = inbox.save_review_state

    def slow_save(state, path=None):
        # Widen the read-modify-write window so a race would be caught if the
        # critical section in review_candidate() were not locked.
        time.sleep(0.05)
        return original_save(state, path)

    monkeypatch.setattr(inbox, "save_review_state", slow_save)

    errors: list[BaseException] = []

    def run(inbox_id: str, decision: str) -> None:
        try:
            inbox.review_candidate(inbox_id, decision=decision)
        except BaseException as exc:  # pragma: no cover - surfaced via errors list
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=("test_source__a1", "adopted")),
        threading.Thread(target=run, args=("test_source__a2", "rejected")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert set(state["reviews"].keys()) == {"test_source__a1", "test_source__a2"}
    assert state["reviews"]["test_source__a1"]["status"] == "adopted"
    assert state["reviews"]["test_source__a2"]["status"] == "rejected"


def test_revised_requires_edited_claim(tmp_path, monkeypatch):
    source = tmp_path / "candidates.jsonl"
    _write_jsonl(source, [{"id": "a1", "claim": "修正したい記憶候補"}])
    monkeypatch.setattr(inbox, "SOURCE_PATHS", {"test_source": source})
    monkeypatch.setattr(inbox, "REVIEW_STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(inbox, "REVIEW_AUDIT_PATH", tmp_path / "audit.jsonl")

    with pytest.raises(ValueError):
        inbox.review_candidate("test_source__a1", decision="revised", edited_claim="")
