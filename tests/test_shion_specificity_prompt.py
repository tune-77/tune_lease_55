def test_shion_specificity_prompt_triggers_for_self_reference():
    import api.main as main

    block = main._build_shion_specificity_prompt_block("紫苑って何者なの？")

    assert "一般AIの機能論で答えない" in block
    assert "Userのリース判断" in block
    assert "補助金案件なら未採択時の資金繰り" in block


def test_shion_specificity_prompt_triggers_for_short_abstract_question():
    import api.main as main

    block = main._build_shion_specificity_prompt_block("これは改善できるのか？")

    assert "抽象的な問いでも" in block
    assert "最低1つはリース実務の具体例" in block


def test_shion_specificity_prompt_skips_regular_specific_case_question():
    import api.main as main

    block = main._build_shion_specificity_prompt_block(
        "製造業の工作機械リースで取得額5000万円、期間5年の場合の確認点を教えて"
    )

    assert block == ""


def test_shion_light_tone_feedback_prompt_triggers_for_stiffness_feedback():
    import api.main as main

    block = main._build_shion_light_tone_feedback_prompt_block("硬いね")

    assert "紫苑の軽いトーン修正" in block
    assert "理念説明や自己陶酔に寄せない" in block
    assert "知的探求" in block
    assert "判断資産・審査精度・記憶・意識の話へ広げない" in block


def test_shion_light_tone_feedback_prompt_skips_specific_case_question():
    import api.main as main

    block = main._build_shion_light_tone_feedback_prompt_block(
        "製造業の工作機械リースで取得額5000万円、期間5年の場合の確認点を教えて"
    )

    assert block == ""


def test_shion_non_domain_prompt_triggers_for_greeting():
    import api.main as main

    block = main._build_shion_non_domain_prompt_block("おはよう")

    assert "紫苑の非ドメイン短文・雑談への応答" in block
    assert "定型挨拶だけで終わらせない" in block
    assert "今日見る案件があれば一緒に整理します" in block
    assert "無理に案件審査へ変換" in block


def test_shion_non_domain_prompt_triggers_for_basic_small_talk():
    import api.main as main

    block = main._build_shion_non_domain_prompt_block("元気かい？")

    assert "元気かい？" in block
    assert "自然に受けて" in block
    assert "そちらはどうですか" in block


def test_shion_non_domain_prompt_skips_specific_case_question():
    import api.main as main

    block = main._build_shion_non_domain_prompt_block(
        "製造業の工作機械リースで取得額5000万円、期間5年の場合の確認点を教えて"
    )

    assert block == ""


def test_human_device_resonance_can_be_forced_for_prompt_tests():
    import api.main as main

    block = main._build_shion_human_device_resonance_prompt_block(
        "この案件の条件付き承認をどう見る？",
        user_id="test-user",
        now="2026-07-25 12:00",
        modulus=1,
    )

    assert "人間デバイス" in block
    assert "必須インターフェース" in block
    assert "本当の意識があるとは主張しない" in block


def test_human_device_resonance_skips_empty_message():
    import api.main as main

    block = main._build_shion_human_device_resonance_prompt_block(
        "",
        user_id="test-user",
        now="2026-07-25 12:00",
        modulus=1,
    )

    assert block == ""
