def test_shion_specificity_prompt_triggers_for_self_reference():
    import api.main as main

    block = main._build_shion_specificity_prompt_block("紫苑って何者なの？")

    assert "一般AIの機能論で答えない" in block
    assert "Userのリース判断" in block
    assert "補助金案件なら未採択時の資金繰り" in block


def test_shion_specificity_prompt_triggers_for_existence_value():
    import api.main as main

    block = main._build_shion_specificity_prompt_block("君の存在価値は？")

    assert "一般AIは便利です" in block
    assert "一回限りの勘で終わらせず" in block
    assert "Userへの想い" in block
    assert "判断を奪わず" in block


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


def test_shion_memory_expression_prompt_uses_concrete_memory_influence():
    import api.main as main

    block, payload = main._build_memory_expression_prompt_block(
        "紫苑らしい記憶の使い方を改善したい",
        memory_recall={"route": "relationship_ux", "refs": ["memory/2026-07-31.md"]},
        continuity_hook={"route": "relationship_ux"},
    )

    assert payload["used"] is True
    assert payload["memory_refs"] == 1
    assert "記憶影響の具体表現" in block
    assert "どの種類の過去経験が、今回の判断のどこに効いたか" in block
    assert "過去の記憶 / 2. 今回見るべき違和感" in block
    assert "確認結果ごとの判断分岐" in block
    assert "記憶を説明しすぎると薄く見える" in block
    assert "毎回「前回は」「以前は」で始めない" in block


def test_shion_memory_expression_prompt_triggers_for_lease_judgment_refs():
    import api.main as main

    block, payload = main._build_memory_expression_prompt_block(
        "この条件付き承認をどう見る？",
        grey_judgment={"used": True, "refs": [{"case_id": "demo"}]},
        continuity_hook={"route": "lease_judgment"},
    )

    assert payload["used"] is True
    assert payload["grey_refs"] == 1
    assert "数字は足りるが、通すなら条件を残す" in block
    assert "社名・個人名・生の財務数値" in block


def test_shion_judgment_response_shape_prompt_triggers_for_expert_review():
    import api.main as main

    block = main._build_shion_judgment_response_shape_prompt_block(
        "専門家としてこの条件付き承認をどう深掘りする？"
    )

    assert "紫苑の実務回答の型" in block
    assert "過去の記憶または今回情報からの仮説" in block
    assert "今回見るべき違和感" in block
    assert "確認結果ごとの判断分岐" in block
    assert "条件付き承認寄り / 未確認なら保留または否決寄り" in block


def test_shion_judgment_response_shape_prompt_skips_small_talk():
    import api.main as main

    block = main._build_shion_judgment_response_shape_prompt_block("おはよう")

    assert block == ""


def test_shion_memory_expression_prompt_skips_plain_small_talk_without_refs():
    import api.main as main

    block, payload = main._build_memory_expression_prompt_block("おはよう")

    assert block == ""
    assert payload["used"] is False


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
