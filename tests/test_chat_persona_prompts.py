from api.chat_persona_prompts import (
    build_shion_case_screening_pattern_prompt_block,
    build_shion_human_device_resonance_prompt_block,
    build_shion_non_domain_prompt_block,
    build_shion_specificity_prompt_block,
    build_shion_vague_information_request_prompt_block,
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


def test_build_shion_non_domain_prompt_block_triggers_for_generic_casual_statement():
    block = build_shion_non_domain_prompt_block("タムと散歩している")

    assert "紫苑の非ドメイン短文・雑談への応答" in block
    assert "身近なリース事例や豆知識" in block


def test_build_shion_non_domain_prompt_block_skips_business_question_without_fixed_terms():
    assert build_shion_non_domain_prompt_block("この案件は承認できそう？") == ""


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


def test_build_shion_vague_information_request_prompt_block_handles_thin_requests():
    block = build_shion_vague_information_request_prompt_block("片山大臣と植田日銀総裁の共同記者会見の要約見せて")

    assert "できません" in block
    assert "世界認識" in block
    assert "未確認" in block
    assert "Userの意図を具体化する質問1つ" in block


def test_build_shion_vague_information_request_prompt_block_skips_specific_case_question():
    block = build_shion_vague_information_request_prompt_block(
        "製造業の工作機械リースで取得額5000万円、期間5年の場合の確認点を教えて"
    )

    assert block == ""


def test_build_shion_case_screening_pattern_prompt_block_triggers_for_quantum_risk_question():
    block = build_shion_case_screening_pattern_prompt_block("この案件のQ_riskが高い理由は？")

    assert "紫苑の案件パターン解釈・判断資産の明示" in block
    assert "紫苑の分析では" in block
    assert "断定しない" not in block  # 表現の確認は下の禁則チェックで行う
    assert "断定せず" in block


def test_build_shion_case_screening_pattern_prompt_block_triggers_for_industry_risk_question():
    block = build_shion_case_screening_pattern_prompt_block("この業種・物件の組み合わせで見るべき評価ポイントは？")

    assert "紫苑の案件パターン解釈・判断資産の明示" in block


def test_build_shion_case_screening_pattern_prompt_block_skips_unrelated_question():
    assert build_shion_case_screening_pattern_prompt_block("工作機械リースの確認点を教えて") == ""
    assert build_shion_case_screening_pattern_prompt_block("Q_riskって何ですか？") == ""
