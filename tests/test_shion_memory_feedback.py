import json

import pytest
from fastapi import HTTPException

from api.routers.shion_memory_feedback import (
    ShionMemoryFeedbackRequest,
    append_shion_memory_feedback,
)


def test_append_shion_memory_feedback_writes_usage_log(tmp_path):
    log = tmp_path / "shion_memory_usage_log.jsonl"
    entry = append_shion_memory_feedback(
        ShionMemoryFeedbackRequest(
            memory_ids=["mem_a", "mem_a", "mem_b"],
            outcome="helped",
            route="case_screening",
            surface="next_chat_rag",
            question="この案件の確認点は？",
            response_hash="abc123",
            comment="効いた",
            user_id="u1",
        ),
        usage_log=log,
    )

    assert entry["memory_feedback"] == "helped"
    assert entry["refs"] == ["mem_a", "mem_b"]
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert row["source"] == "explicit_memory_feedback"
    assert row["route"] == "case_screening"
    assert row["refs"] == ["mem_a", "mem_b"]


def test_append_shion_memory_feedback_requires_memory_ids(tmp_path):
    with pytest.raises(HTTPException):
        append_shion_memory_feedback(
            ShionMemoryFeedbackRequest(memory_ids=[], outcome="neutral"),
            usage_log=tmp_path / "usage.jsonl",
        )
