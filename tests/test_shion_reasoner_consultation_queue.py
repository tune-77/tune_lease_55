from pathlib import Path

import pytest

from api import shion_reasoner_consultation_queue as queue


def test_request_reasoner_consultation_appends_sanitized_queue_item(tmp_path: Path):
    path = tmp_path / "reasoner_queue.jsonl"

    item = queue.request_reasoner_consultation(
        target="claude",
        purpose="fallback",
        title="Codex停止時の第二意見",
        question="株式会社テスト を匿名化し、売上 123456789 円を見てよいか",
        shion_hypothesis="Codexが止まっているため、要約だけでClaudeに第二意見を聞くべき。",
        context_summary="電話 090-1234-5678 とメール test@example.com は含めない。",
        priority="high",
        path=path,
    )

    assert item["id"].startswith("reasonq_")
    assert item["target"] == "claude"
    assert item["purpose"] == "fallback"
    assert item["priority"] == "high"
    assert item["status"] == "open"
    assert item["safety_policy"] == "queue_only_safe_summary_no_raw_customer_data_no_code_execution"
    assert "[COMPANY]" in item["question"]
    assert "[LARGE_NUMBER]" in item["question"]
    assert "[PHONE]" in item["context_summary"]
    assert "[EMAIL]" in item["context_summary"]
    assert queue.list_reasoner_consultations(path=path)[0]["id"] == item["id"]


def test_set_reasoner_consultation_status_replays_event_log(tmp_path: Path):
    path = tmp_path / "reasoner_queue.jsonl"
    item = queue.request_reasoner_consultation(
        target="gemini",
        purpose="hypothesis_check",
        title="仮説比較",
        question="審査観点の優先順位を比較したい。",
        shion_hypothesis="物件保全より資金繰りを先に見るべき。",
        path=path,
    )

    updated = queue.set_reasoner_consultation_status(
        item["id"],
        "answered",
        note="回答を要約済み",
        answered_by="gemini",
        answer_summary="資金繰りを先に見つつ、物件保全は条件設計で補う。",
        path=path,
    )

    assert updated["status"] == "answered"
    assert updated["answered_by"] == "gemini"
    assert "資金繰り" in updated["answer_summary"]
    assert queue.list_reasoner_consultations(status="open", path=path) == []
    assert queue.list_reasoner_consultations(status="answered", path=path)[0]["id"] == item["id"]


def test_requires_question_and_hypothesis(tmp_path: Path):
    with pytest.raises(ValueError):
        queue.request_reasoner_consultation(
            target="claude",
            purpose="fallback",
            title="x",
            question="",
            shion_hypothesis="",
            path=tmp_path / "reasoner_queue.jsonl",
        )


def test_reasoner_prompt_block_exposes_pending_items_without_execution(tmp_path: Path):
    path = tmp_path / "reasoner_queue.jsonl"
    queue.request_reasoner_consultation(
        target="claude",
        purpose="long_context_review",
        title="長文レビュー",
        question="AGENT作業プロトコルの抜けを見たい。",
        shion_hypothesis="Codexが止まった時の代替相談先が必要。",
        path=path,
    )

    block = queue.build_reasoner_consultation_prompt_block(path=path)

    assert "外部推論相談キュー" in block
    assert "直接暴走実行しない" in block
    assert "claude" in block
    assert "長文レビュー" in block
