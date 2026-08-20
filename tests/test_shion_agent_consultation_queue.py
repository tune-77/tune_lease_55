from pathlib import Path

import pytest

from api import shion_agent_consultation_queue as queue


def test_request_consultation_appends_queue_item(tmp_path: Path):
    path = tmp_path / "queue.jsonl"

    item = queue.request_consultation(
        agent="judgment-asset-auditor",
        title="判断資産候補が消える疑い",
        reason="candidate stateに人間シグナルがあるのに候補JSONLへ残らない疑いがある。",
        observed_signal="ready_for_promotion が滞留",
        suggested_scope="data/autoresearch_judgment_asset_candidate_state.json",
        priority="high",
        path=path,
    )

    assert item["id"].startswith("agentq_")
    assert item["agent"] == "judgment-asset-auditor"
    assert item["priority"] == "high"
    assert item["status"] == "open"
    assert item["safety_policy"] == "queue_only_no_agent_execution_no_code_change"
    assert queue.list_consultations(path=path)[0]["id"] == item["id"]


def test_set_consultation_status_replays_event_log(tmp_path: Path):
    path = tmp_path / "queue.jsonl"
    item = queue.request_consultation(
        agent="ledger-consistency-auditor",
        title="REV重複疑い",
        reason="REV番号が複数台帳でずれている疑い。",
        path=path,
    )

    updated = queue.set_consultation_status(item["id"], "in_review", note="Codex確認中", path=path)

    assert updated["status"] == "in_review"
    assert updated["resolution_note"] == "Codex確認中"
    assert queue.list_consultations(status="open", path=path) == []
    assert queue.list_consultations(status="in_review", path=path)[0]["id"] == item["id"]


def test_rejects_unknown_agent(tmp_path: Path):
    with pytest.raises(ValueError):
        queue.request_consultation(
            agent="unknown-agent",
            title="x",
            reason="y",
            path=tmp_path / "queue.jsonl",
        )


def test_prompt_block_exposes_pending_items_without_execution(tmp_path: Path):
    path = tmp_path / "queue.jsonl"
    queue.request_consultation(
        agent="judgment-asset-auditor",
        title="候補保全監査",
        reason="保全経路を確認したい。",
        path=path,
    )

    block = queue.build_agent_consultation_prompt_block(path=path)

    assert "AGENT相談キュー" in block
    assert "直接実行しない" in block
    assert "judgment-asset-auditor" in block
    assert "候補保全監査" in block
