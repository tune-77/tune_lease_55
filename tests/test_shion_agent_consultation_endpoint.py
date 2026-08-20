from fastapi import BackgroundTasks

from api import shion_agent_consultation_queue as queue
from api.routers import shion_tasks


def test_agent_consultation_endpoint_round_trip(tmp_path, monkeypatch):
    queue_path = tmp_path / "queue.jsonl"
    monkeypatch.setattr(queue, "QUEUE_PATH", queue_path)
    monkeypatch.setattr(queue, "_log_action_ledger", lambda item: None)

    created = shion_tasks.post_shion_agent_consultation(
        shion_tasks.AgentConsultationRequest(
            agent="judgment-asset-auditor",
            title="判断資産監査",
            reason="候補保全が断線していないか確認したい。",
            priority="high",
        ),
        BackgroundTasks(),
    )

    assert created["status"] == "queued"
    queue_id = created["item"]["id"]
    listed = shion_tasks.get_shion_agent_consultations(status="open")
    assert listed["count"] == 1
    assert listed["items"][0]["id"] == queue_id

    updated = shion_tasks.patch_shion_agent_consultation(
        queue_id,
        shion_tasks.AgentConsultationStatusRequest(status="done", note="監査完了"),
        BackgroundTasks(),
    )

    assert updated["item"]["status"] == "done"
    assert shion_tasks.get_shion_agent_consultations(status="open")["count"] == 0
    assert shion_tasks.get_shion_agent_consultations(status="done")["items"][0]["id"] == queue_id
