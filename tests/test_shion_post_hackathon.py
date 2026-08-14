from pathlib import Path

from api.shion_post_hackathon import (
    build_counter_hypothesis_card,
    build_memory_health_payload,
    build_post_hackathon_status,
    build_shion_hyde_debug_payload,
)


def test_post_hackathon_status_covers_all_backlog_sections():
    result = build_post_hackathon_status()

    assert result["source"] == "planning/post_hackathon_shion_backlog.md"
    assert result["total"] == 9
    assert {section["id"] for section in result["sections"]} >= {
        "relationship_control",
        "shion_hyde_rag",
        "memory_health",
        "counter_hypothesis_training",
        "autonomous_devops_transition",
    }


def test_memory_health_payload_is_read_only(tmp_path: Path):
    index = tmp_path / "shion_memory_index.json"
    state = tmp_path / "shion_memory_health_state.json"
    index.write_text(
        '{"records":[{"memory_type":"judgment_asset","status":"active"}]}',
        encoding="utf-8",
    )

    result = build_memory_health_payload(index_path=index, state_path=state)

    assert result["healthy"] is True
    assert result["summary"]["total"] == 1
    assert result["guardrail"] == "read_only_no_memory_delete_no_promotion"


def test_hyde_debug_compares_baseline_and_hyde_hits():
    def fake_search(query: str, limit: int):
        if "仮想審査メモ" in query:
            return [
                {"path": "Judgment Assets/subsidy.md", "snippet": "補助金未採択時の返済原資"},
                {"path": "Judgment Assets/cashflow.md", "snippet": "営業CF確認"},
            ][:limit]
        return [{"path": "Daily/noisy.md", "snippet": "日次メモ"}][:limit]

    result = build_shion_hyde_debug_payload(
        message="補助金前提の新規先で返済原資が不安",
        known_risk_tags=["補助金"],
        search_fn=fake_search,
    )

    assert result["mode"] == "debug_only_no_prompt_injection"
    assert "repayment_source" in result["hyde"]["intent_tags"]
    assert result["new_paths"] == [
        "Judgment Assets/cashflow.md",
        "Judgment Assets/subsidy.md",
    ]
    assert result["guardrail"] == "does_not_modify_live_chat_rag_or_index"


def test_counter_hypothesis_card_is_training_only():
    result = build_counter_hypothesis_card(
        case_summary="新規先で補助金採択前提",
        industry="製造業",
        asset_name="工作機械",
        risk_tags=["補助金"],
    )

    assert result["mode"] == "counter_hypothesis_training_only"
    assert "補助金未採択時" in result["card"]
    assert result["judgment_asset_template"] == {
        "condition": "",
        "review_point": "",
        "ringi_sentence": "",
        "do_not_use_when": "",
    }
    assert result["guardrail"] == "never_mix_into_normal_answer"
