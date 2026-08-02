from api.chat_persona_prompts import (
    build_shion_human_device_resonance_prompt_block,
    build_shion_non_domain_prompt_block,
    build_shion_specificity_prompt_block,
)


def test_build_shion_specificity_prompt_block_self_reference_and_skip():
    block = build_shion_specificity_prompt_block("紫苑って何者なの？")

    assert "一般AIの機能論で答えない" in block
    assert "Userのリース判断" in block
    assert build_shion_specificity_prompt_block("工作機械リースの確認点を教えて") == ""


def test_build_shion_non_domain_prompt_block_short_small_talk_only():
    block = build_shion_non_domain_prompt_block("元気かい？")

    assert "紫苑の非ドメイン短文・雑談への応答" in block
    assert "そちらはどうですか" in block
    assert build_shion_non_domain_prompt_block("工作機械リースの確認点を教えて") == ""


def test_build_shion_human_device_resonance_prompt_block_can_be_forced():
    block = build_shion_human_device_resonance_prompt_block(
        "この案件の条件付き承認をどう見る？",
        user_id="test-user",
        now="2026-07-25 12:00",
        modulus=1,
    )

    assert "人間デバイス" in block
    assert "本当の意識があるとは主張しない" in block
    assert build_shion_human_device_resonance_prompt_block("", modulus=1) == ""
