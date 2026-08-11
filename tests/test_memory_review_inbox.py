import json
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


def test_revised_requires_edited_claim(tmp_path, monkeypatch):
    source = tmp_path / "candidates.jsonl"
    _write_jsonl(source, [{"id": "a1", "claim": "修正したい記憶候補"}])
    monkeypatch.setattr(inbox, "SOURCE_PATHS", {"test_source": source})
    monkeypatch.setattr(inbox, "REVIEW_STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(inbox, "REVIEW_AUDIT_PATH", tmp_path / "audit.jsonl")

    with pytest.raises(ValueError):
        inbox.review_candidate("test_source__a1", decision="revised", edited_claim="")
