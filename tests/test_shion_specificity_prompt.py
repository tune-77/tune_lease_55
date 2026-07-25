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
