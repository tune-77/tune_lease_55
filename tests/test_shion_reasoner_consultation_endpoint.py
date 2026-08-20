from fastapi import BackgroundTasks

from api import shion_reasoner_consultation_queue as queue
from api.routers import shion_tasks


def test_reasoner_consultation_endpoint_round_trip(tmp_path, monkeypatch):
    queue_path = tmp_path / "reasoner_queue.jsonl"
    monkeypatch.setattr(queue, "QUEUE_PATH", queue_path)
    monkeypatch.setattr(queue, "_log_action_ledger", lambda item: None)

    created = shion_tasks.post_shion_reasoner_consultation(
        shion_tasks.ReasonerConsultationRequest(
            target="claude",
            purpose="fallback",
            title="Codex停止時の第二意見",
            question="匿名化した論点だけで第二意見をほしい。",
            shion_hypothesis="Codexが止まっているためClaudeに回す。",
            priority="high",
        ),
        BackgroundTasks(),
    )

    assert created["status"] == "queued"
    queue_id = created["item"]["id"]
    listed = shion_tasks.get_shion_reasoner_consultations(status="open")
    assert listed["count"] == 1
    assert listed["items"][0]["id"] == queue_id

    updated = shion_tasks.patch_shion_reasoner_consultation(
        queue_id,
        shion_tasks.ReasonerConsultationStatusRequest(
            status="answered",
            answered_by="claude",
            answer_summary="相談結果を要約済み。",
            note="第二意見を確認",
        ),
        BackgroundTasks(),
    )

    assert updated["item"]["status"] == "answered"
    assert updated["item"]["answered_by"] == "claude"
    assert shion_tasks.get_shion_reasoner_consultations(status="open")["count"] == 0
    assert shion_tasks.get_shion_reasoner_consultations(status="answered")["items"][0]["id"] == queue_id
