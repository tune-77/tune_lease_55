from api.chat_continuity_prompts import (
    build_continuity_hook_prompt_block,
    build_delta_awareness_prompt_block,
    build_memory_expression_prompt_block,
    build_memory_to_judgment_prompt_block,
    relationship_signal_route,
)


def test_relationship_signal_route_classifies_key_routes():
    assert relationship_signal_route("同じ紫苑として覚えてる？") == "relationship_ux"
    assert relationship_signal_route("Cloud Run と Cloudflare の差は？") == "environment_continuity"
    assert relationship_signal_route("リース稟議の条件付き承認") == "lease_judgment"
    assert relationship_signal_route("実装してテスト") == "implementation"
    assert relationship_signal_route("おはよう") == "default"


def test_build_continuity_hook_prompt_block_includes_feedback_lines():
    block, payload = build_continuity_hook_prompt_block(
        "リース稟議を見て",
        human_feedback_summary=lambda route: {
            "route": route,
            "positive_starts": ["ここは稟議で使えます"],
            "negative_starts": ["もちろんです"],
            "recent_comments": ["一般論が薄い"],
        },
    )

    assert payload["route"] == "lease_judgment"
    assert "Human Response Feedback" in block
    assert "もちろんです" in block


def test_build_delta_awareness_requires_explicit_continuation():
    history = [{"role": "user", "content": "リース稟議を見て"}]

    skipped_block, skipped = build_delta_awareness_prompt_block("別件で改善できる？", history)
    used_block, used = build_delta_awareness_prompt_block("その続きで改善できる？", history)

    assert skipped_block == ""
    assert skipped["used"] is False
    assert used["used"] is True
    assert "Delta Awareness" in used_block


def test_build_memory_to_judgment_and_expression_blocks():
    m2j_block, m2j = build_memory_to_judgment_prompt_block(
        "この条件付き承認をどう見る？",
        memory_recall={"refs": ["r1"]},
        continuity_hook={"route": "lease_judgment"},
    )
    expr_block, expr = build_memory_expression_prompt_block(
        "この条件付き承認をどう見る？",
        grey_judgment={"refs": [{"case_id": "demo"}]},
        continuity_hook={"route": "lease_judgment"},
    )

    assert m2j["route"] == "lease_judgment"
    assert "稟議で使える判断軸" in m2j_block
    assert expr["grey_refs"] == 1
    assert "数字は足りるが、通すなら条件を残す" in expr_block
