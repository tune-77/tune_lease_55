from api.chat_debug_metadata import (
    append_chat_debug_metadata,
    append_retrieval_debug_payloads,
    user_personal_memory_debug_payload,
    vertex_answer_public_payload,
    vertex_search_public_payload,
)
from api.chat_prompt_blocks import append_optional_block, block_with_spacing, join_prompt_blocks
from api.chat_response_pipeline import (
    build_chat_response_payload,
    build_prompt_feedback_snapshot,
    obsidian_daily_intelligence_payload,
    vertex_retrieval_response_extra,
)


def test_vertex_public_payloads_keep_stable_shape_and_ref_limits():
    search = {"used": True, "status": "ok", "refs": ["a", "b", "c"], "summary": "summary"}
    answer = {
        "used": True,
        "status": "ok",
        "grounding_score": 0.887,
        "grounding_score_source": "citation_coverage_fallback",
        "low_support_claim_count": 0,
        "support_count": 25,
        "refs": ["x", "y"],
        "answer_text": "answer",
        "grounding_supports": [{"claim_text": "claim"}],
    }

    assert vertex_search_public_payload(search, ref_limit=2) == {
        "used": True,
        "status": "ok",
        "refs": ["a", "b"],
    }
    assert vertex_answer_public_payload(answer)["grounding_score"] == 0.887

    memory_debug = append_retrieval_debug_payloads({}, vertex_ai_search=search, vertex_answer_api=answer)

    assert memory_debug["vertex_ai_search"]["summary_preview"] == "summary"
    assert memory_debug["vertex_answer_api"]["answer_preview"] == "answer"
    assert memory_debug["vertex_answer_api"]["grounding_supports"] == [{"claim_text": "claim"}]


def test_user_personal_memory_debug_payload_is_small_and_stable():
    payload = user_personal_memory_debug_payload({"block": "x", "refs": list(range(10)), "line_count": 7})

    assert payload == {"used": True, "refs": [0, 1, 2, 3, 4, 5], "line_count": 7}


def test_append_chat_debug_metadata_adds_user_and_retrieval_debug():
    memory_debug = append_chat_debug_metadata(
        {},
        user_personal_memory={"block": "x", "refs": list(range(10)), "line_count": 7},
        vertex_ai_search={"used": True, "status": "ok", "refs": ["a"], "summary": "summary"},
        vertex_answer_api={"used": True, "status": "ok", "refs": ["x"], "answer_text": "answer"},
    )

    assert memory_debug["user_personal_memory"] == {"used": True, "refs": [0, 1, 2, 3, 4, 5], "line_count": 7}
    assert memory_debug["vertex_ai_search"]["summary_preview"] == "summary"
    assert memory_debug["vertex_answer_api"]["answer_preview"] == "answer"


def test_prompt_block_helpers_preserve_existing_concat_behavior():
    assert join_prompt_blocks(["root", "", "\n\nblock"]) == "root\n\nblock"
    assert block_with_spacing("pdca") == "\n\npdca"
    assert append_optional_block("root", "") == "root"
    assert append_optional_block("root", "pdca") == "root\n\npdca"


def test_response_pipeline_helpers_build_chat_payload_and_feedback_snapshot():
    snapshot = build_prompt_feedback_snapshot(
        "system",
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        "next",
    )
    assert snapshot == "system\n\nuser: hello\nassistant: hi\n\nuser: next"

    payload = build_chat_response_payload(
        reply="ok",
        total_messages=2,
        lease_news_focus={},
        lease_news_brief={},
        lease_news_actions={},
        long_input_mode=False,
        context_mode="screening",
        response_mode="shion",
        external_research={"used": False},
        personal_memory_capture={"captured": False},
        language_material_capture={"captured": False},
        response_impact_prediction={"captured": False},
        judgment_asset_capture={"captured": False},
        obsidian_daily_intelligence=obsidian_daily_intelligence_payload(used=True),
        knowledge_refs=[{"ref": "k"}],
        extra={"vertex_ai_search": {"used": True}},
    )

    assert payload["reply"] == "ok"
    assert payload["knowledge_refs"] == [{"ref": "k"}]
    assert payload["obsidian_daily_intelligence"]["used"] is True
    assert payload["vertex_ai_search"] == {"used": True}


def test_vertex_retrieval_response_extra_keeps_public_response_shape():
    extra = vertex_retrieval_response_extra(
        vertex_ai_search={"used": True, "status": "ok", "refs": ["a"]},
        vertex_answer_api={
            "used": True,
            "status": "ok",
            "grounding_score": 0.5,
            "grounding_score_source": "fallback",
            "refs": ["x"],
        },
    )

    assert extra["vertex_ai_search"] == {"used": True, "status": "ok", "refs": ["a"]}
    assert extra["vertex_answer_api"] == {
        "used": True,
        "status": "ok",
        "grounding_score": 0.5,
        "grounding_score_source": "fallback",
        "low_support_claim_count": 0,
        "support_count": 0,
        "refs": ["x"],
    }
