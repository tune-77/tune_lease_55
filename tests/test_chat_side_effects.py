from api.chat_side_effects import (
    chat_exchange_metadata,
    compact_identity_memory_payload,
    compact_memory_recall_payload,
    memory_usage_extra,
    prompt_feedback_extra,
    should_auto_save_chat,
)


def test_compact_payloads_keep_existing_shape_and_limits():
    memory_recall = {
        "route": "hybrid",
        "refs": list(range(12)),
        "practical_scene": {"scene": "review"},
        "ignored": True,
    }
    identity_memory = {
        "block": "identity",
        "refs": list("abcdefghij"),
        "layers": {"tone": 1},
        "ignored": True,
    }

    assert compact_memory_recall_payload(memory_recall) == {
        "route": "hybrid",
        "refs": list(range(8)),
        "practical_scene": {"scene": "review"},
    }
    assert compact_identity_memory_payload(identity_memory) == {
        "used": True,
        "refs": list("abcdefgh"),
        "layers": {"tone": 1},
    }


def test_chat_exchange_metadata_matches_general_and_rag_shapes():
    assert chat_exchange_metadata(context_mode="casual") == {"context_mode": "casual"}

    metadata = chat_exchange_metadata(
        context_mode="screening",
        knowledge_ref_count=3,
        improvement_mode=True,
        include_improvement_mode=True,
        vertex_ai_search={"used": True, "status": "ok", "refs": ["a", "b"]},
        vertex_answer_api={
            "used": True,
            "status": "ok",
            "grounding_score": 0.887,
            "grounding_score_source": "citation_coverage_fallback",
            "low_support_claim_count": 0,
            "support_count": 2,
            "refs": ["x"],
        },
    )

    assert metadata == {
        "context_mode": "screening",
        "knowledge_refs": 3,
        "improvement_mode": True,
        "vertex_ai_search": {"used": True, "status": "ok", "refs": ["a", "b"]},
        "vertex_answer_api": {
            "used": True,
            "status": "ok",
            "grounding_score": 0.887,
            "grounding_score_source": "citation_coverage_fallback",
            "low_support_claim_count": 0,
            "support_count": 2,
            "refs": ["x"],
        },
    }


def test_prompt_feedback_extra_matches_general_and_rag_shapes():
    assert prompt_feedback_extra(user_id="u1", intent="", category="general") == {
        "user_id": "u1",
        "intent": "",
        "category": "general",
    }

    extra = prompt_feedback_extra(
        user_id="u1",
        intent="ask",
        category="rag",
        improvement_mode=False,
        memory_recall={"route": "local", "refs": ["r"], "practical_scene": None},
        continuity_hook={"used": True},
        delta_awareness={"changed": False},
        memory_to_judgment={"used": True},
        memory_expression={"tone": "plain"},
        reflection_gate={"allowed": True},
        grey_judgment_memory={"used": False},
        world_proxy={"used": True},
    )

    assert extra["improvement_mode"] is False
    assert extra["memory_recall"] == {"route": "local", "refs": ["r"], "practical_scene": {}}
    assert extra["grey_judgment_memory"] == {"used": False}
    assert extra["world_proxy"] == {"used": True}


def test_memory_usage_extra_keeps_general_memory_and_rag_vertex_shapes():
    general = memory_usage_extra(
        user_id="u1",
        category="general",
        memory_recall={"route": "local", "refs": ["r"], "practical_scene": {"p": 1}},
        identity_memory={"block": "x", "refs": ["i"], "layers": {"l": 1}},
        continuity_hook={"used": True},
        grey_judgment_memory={"used": True},
        world_proxy={"used": True, "functions": ["memory_experience_proxy"]},
    )

    assert general["memory_recall"]["route"] == "local"
    assert general["identity_memory"] == {"used": True, "refs": ["i"], "layers": {"l": 1}}
    assert general["continuity_hook"] == {"used": True}
    assert general["grey_judgment_memory"] == {"used": True}
    assert general["world_proxy"] == {"used": True, "functions": ["memory_experience_proxy"]}

    rag = memory_usage_extra(
        user_id="u1",
        category="rag",
        improvement_mode=True,
        vertex_ai_search={"used": False, "status": "disabled", "refs": []},
        vertex_answer_api={"used": False, "status": "disabled", "refs": []},
        identity_memory={"block": "", "refs": [], "layers": {}},
    )

    assert rag["improvement_mode"] is True
    assert rag["vertex_ai_search"] == {"used": False, "status": "disabled", "refs": []}
    assert rag["vertex_answer_api"]["support_count"] == 0
    assert "memory_recall" not in rag


def test_memory_usage_extra_includes_estimated_user_emotion_when_present():
    without_emotion = memory_usage_extra(user_id="u1", category="rag")
    assert "estimated_user_emotion" not in without_emotion

    with_emotion = memory_usage_extra(
        user_id="u1",
        category="rag",
        estimated_user_emotion="疲れている",
    )
    assert with_emotion["estimated_user_emotion"] == "疲れている"


def test_should_auto_save_chat_skips_improvement_messages():
    assert should_auto_save_chat(improvement_mode=False) is True
    assert should_auto_save_chat(improvement_mode=True) is False
