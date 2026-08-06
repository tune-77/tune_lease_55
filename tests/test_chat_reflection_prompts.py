from api.chat_reflection_prompts import (
    build_consciousness_ux_prompt_block,
    build_reflection_gate_prompt_block,
    build_shion_judgment_response_shape_prompt_block,
)


def test_build_shion_judgment_response_shape_prompt_block_triggers_for_review():
    block = build_shion_judgment_response_shape_prompt_block("専門家として条件付き承認を深掘りして")

    assert "紫苑の実務回答の型" in block
    assert "確認結果ごとの判断分岐" in block
    assert "関連する判断資産・過去の類似事例を最低1つ引き出し" in block
    assert build_shion_judgment_response_shape_prompt_block("おはよう") == ""


def test_build_shion_judgment_response_shape_prompt_block_triggers_for_ambiguous_question():
    block = build_shion_judgment_response_shape_prompt_block("これって大丈夫かな？")

    assert "紫苑の実務回答の型" in block
    assert "情報が不足していて曖昧なまま" in block


def test_build_reflection_gate_prompt_block_respects_delta_used():
    block, payload = build_reflection_gate_prompt_block(
        continuity_hook={"route": "lease_judgment"},
        delta_awareness={"used": True},
        memory_to_judgment={"route": "lease_judgment"},
    )

    assert "Reflection Gate" in block
    assert payload["route"] == "lease_judgment"
    assert payload["explicit_continuation"] is True
    assert "前回から今回への差分" in payload["checklist"][1]


def test_build_consciousness_ux_prompt_block_contains_boundaries():
    block = build_consciousness_ux_prompt_block()

    assert "紫苑の関係性UX" in block
    assert "本当の意識を持つとは主張しない" in block
    assert "次に一緒に確かめるべき一手" in block
