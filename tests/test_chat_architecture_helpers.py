from api.chat_debug_metadata import (
    append_chat_debug_metadata,
    append_retrieval_debug_payloads,
    chat_memory_debug_payload,
    dialogue_shared_memory_public_payload,
    relationship_loop_engineering_payload,
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


def test_chat_memory_debug_payload_keeps_core_shape_and_limits():
    payload = chat_memory_debug_payload(
        category="rag",
        context_mode="screening",
        knowledge_refs=[str(i) for i in range(20)],
        memory_recall={"route": "local", "refs": list(range(20)), "impact_hints": list(range(20))},
        pdca_block="pdca",
        judgment_learning_used=True,
        rag_context="rag",
        db_context="db",
        obsidian_daily_used=True,
        identity_memory={"block": "identity", "refs": list(range(20)), "layers": {"identity": True}},
        continuity_hook={
            "used": True,
            "route": "lease",
            "hook": "hook",
            "reason": "reason",
            "human_response_feedback": {"positive_count": 2},
        },
        delta_awareness={"used": True, "delta": "changed"},
        memory_to_judgment={"used": True, "directive": "use it", "memory_refs": list(range(20))},
        memory_expression={"used": True, "memory_refs": 2, "knowledge_refs": 3},
        reflection_gate={"used": True, "mode": "check", "checklist": list(range(20))},
        grey_judgment_memory={"used": True, "query_terms": list(range(20)), "refs": list(range(20))},
        world_proxy={
            "used": True,
            "level": "L1_inference_time_guidance",
            "functions": list(range(20)),
            "grounding": {"knowledge_refs": 12},
        },
    )

    assert payload["category"] == "rag"
    assert payload["knowledge_refs"] == [str(i) for i in range(12)]
    assert payload["memory_recall"]["refs"] == list(range(12))
    assert payload["memory_recall"]["impact_hints"] == list(range(8))
    assert payload["identity_memory"]["layers"]["identity"] is True
    assert payload["pdca_applied"] is True
    assert payload["rag_context_used"] is True
    assert payload["db_context_used"] is True
    assert payload["relationship_loop_engineering"]["closed_loop"] is True
    assert payload["world_proxy"]["functions"] == list(range(8))
    assert payload["world_proxy"]["grounding"] == {"knowledge_refs": 12}


def test_relationship_loop_engineering_payload_summarizes_loop_evidence():
    payload = relationship_loop_engineering_payload(
        continuity_hook={
            "route": "lease",
            "hook": "前回の続き",
            "human_response_feedback": {"positive_count": 3, "negative_count": 1},
        },
        delta_awareness={"delta": "changed"},
        memory_to_judgment={"directive": "審査へ戻す"},
        reflection_gate={"used": True, "mode": "check"},
    )

    assert payload["name"] == "Relationship Loop Engineering"
    assert payload["loop"][0]["evidence"] == {"positive_count": 3, "negative_count": 1}
    assert payload["loop"][4]["evidence"] == {"directive": "審査へ戻す"}
    assert payload["closed_loop"] is True


def test_dialogue_shared_memory_public_payload_keeps_compact_shape():
    payload = dialogue_shared_memory_public_payload(
        {
            "identity_memory": {
                "block": "identity",
                "refs": list(range(20)),
                "layers": {"identity": True, "judgment": False, "recent": True},
            },
            "experience_loop": {"used": True},
            "memory_recall": {"route": "local", "refs": list(range(20)), "practical_scene": {"scene": "dialogue"}},
            "continuity_hook": {"used": True, "route": "lease", "reason": "続き"},
            "delta_awareness": {"used": True, "delta": "changed"},
            "memory_to_judgment": {"used": True, "route": "lease", "directive": "審査へ戻す"},
            "memory_expression": {"used": True, "route": "lease", "memory_refs": 2, "knowledge_refs": 3, "grey_refs": 4},
            "reflection_gate": {"used": True, "mode": "check"},
            "grey_judgment_memory": {"used": True, "refs": list(range(20))},
            "world_proxy": {
                "used": True,
                "level": "L1_inference_time_guidance",
                "functions": list(range(20)),
            },
        }
    )

    assert payload["identity_memory"] == {
        "used": True,
        "refs": list(range(8)),
        "layers": {"identity": True, "judgment": False, "recent": True},
    }
    assert payload["memory_recall"]["refs"] == list(range(8))
    assert payload["memory_recall"]["practical_scene"] == {"scene": "dialogue"}
    assert payload["grey_judgment_memory"]["refs"] == list(range(8))
    assert payload["world_proxy"]["functions"] == list(range(8))


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
