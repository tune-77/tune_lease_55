import json
from datetime import datetime, timedelta

import api.chat_memory as chat_memory
import api.feedback_pattern_loop as feedback_loop
import api.shion_memory_decay as memory_decay
import api.shion_relationship as relationship


class _UserCursor:
    def execute(self, *_args, **_kwargs):
        return None

    def fetchall(self):
        return [{"user_id": "user-1"}]


class _UserConnection:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return _UserCursor()


def test_chat_summary_internal_failure_is_not_success(monkeypatch):
    monkeypatch.setattr(chat_memory, "init_chat_messages_table", lambda: None)
    monkeypatch.setattr(chat_memory, "get_connection", lambda: _UserConnection())
    monkeypatch.setattr(chat_memory, "get_message_count", lambda _user_id: 30)
    monkeypatch.setattr(chat_memory, "_cached_message_count", lambda _user_id: 0)
    monkeypatch.setattr(
        chat_memory,
        "get_summary",
        lambda _user_id, **_kwargs: (_ for _ in ()).throw(RuntimeError("Gemini timeout")),
    )

    result = chat_memory.refresh_stale_chat_summaries()

    assert result["status"] == "error"
    assert result["failed"] == 1
    assert result["refreshed"] == 0


def test_inactivity_decay_is_idempotent_for_same_day(tmp_path, monkeypatch):
    state_path = tmp_path / "relationship.json"
    current = datetime(2026, 10, 5, 4, 5)
    state_path.write_text(
        json.dumps(
            {
                "score": 7.0,
                "last_interaction": (current - timedelta(days=5)).isoformat(),
                "trend": "stable",
                "delta_history": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(relationship, "_STATE_PATH", state_path)

    first = relationship.apply_inactivity_decay(now=current)
    second = relationship.apply_inactivity_decay(now=current)

    # REV-597: 5日の沈黙で中立（6.5）へ戻る分と、その日1日分のペナルティ（0.15）だけ（以前は超過日数×0.15）
    assert first["score"] == 6.758
    assert second["score"] == first["score"]
    assert second["delta_history"] == [-0.15]
    assert second["last_inactivity_decay_date"] == "2026-10-05"


def test_inactivity_decay_fails_closed_on_corrupt_state(tmp_path, monkeypatch):
    state_path = tmp_path / "relationship.json"
    state_path.write_text("truncated", encoding="utf-8")
    monkeypatch.setattr(relationship, "_STATE_PATH", state_path)

    try:
        relationship.apply_inactivity_decay(now=datetime(2026, 10, 5, 4, 5))
    except RuntimeError as exc:
        assert "unreadable" in str(exc)
    else:
        raise AssertionError("corrupt state must stop the decay job")

    assert state_path.read_text(encoding="utf-8") == "truncated"


def test_memory_decay_writes_at_most_one_snapshot_per_day(tmp_path, monkeypatch):
    index_path = tmp_path / "index.json"
    freshness_path = tmp_path / "freshness.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "id": "memory-1",
                        "content": "memory",
                        "confidence": 0.8,
                        "created_at": "2026-10-01T00:00:00",
                        "status": "active",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(memory_decay, "_INDEX_PATH", index_path)
    monkeypatch.setattr(memory_decay, "_FRESHNESS_PATH", freshness_path)
    monkeypatch.setattr(memory_decay, "_USAGE_LOG_PATH", tmp_path / "missing-usage.jsonl")

    first = memory_decay.run_memory_decay_batch()
    second = memory_decay.run_memory_decay_batch()

    assert first["status"] == "ok"
    assert second["status"] == "skipped_duplicate"
    assert len(freshness_path.read_text(encoding="utf-8").splitlines()) == 1


def test_pdca_evaluation_is_not_duplicated_on_same_day(tmp_path, monkeypatch):
    improvement_path = tmp_path / "improvement.jsonl"
    feedback_path = tmp_path / "feedback.jsonl"
    pdca_path = tmp_path / "pdca.jsonl"
    improvement_path.write_text(
        json.dumps(
            {
                "title": "proposal",
                "ts": "2026-10-01T00:00:00+00:00",
                "proposed_by": "shion",
                "surface": "shion_self_proposal",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    feedback_path.write_text(
        json.dumps({"ts": "2026-10-02T00:00:00+00:00", "rating": "negative"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(feedback_loop, "_IMPROVEMENT_LOG_PATH", improvement_path)
    monkeypatch.setattr(feedback_loop, "_FEEDBACK_PATH", feedback_path)
    monkeypatch.setattr(feedback_loop, "_PDCA_LOG_PATH", pdca_path)

    first = feedback_loop.evaluate_proposal_impact()
    second = feedback_loop.evaluate_proposal_impact()

    assert first["evaluated"] == 1
    assert second["evaluated"] == 0
    assert len(pdca_path.read_text(encoding="utf-8").splitlines()) == 1
