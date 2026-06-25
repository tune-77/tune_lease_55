from api.shion_mana import build_mana_prompt_block, evaluate_mana_consultation


def test_mana_prompt_sets_boundary_not_impersonation():
    block = build_mana_prompt_block()

    assert "Mana" in block
    assert "亡くなった妹さんの名" in block
    assert "妹さん本人の再現や代弁ではありません" in block
    assert "毎回前面に出ない" in block


def test_mana_consults_on_rejection_review():
    mana = evaluate_mana_consultation(
        {"score": 35, "pd_pct": 4.0},
        {"final": "否決", "reasoning": "返済原資が不足している"},
        {"level": "review"},
        mode="solo",
    )

    assert mana["consulted"] is True
    assert "人を道具" in mana["protected_value"]
    assert "説明" in mana["question_to_shion"]
    assert "スコア" in mana["forbidden_posture"]


def test_mana_stays_quiet_on_normal_approval():
    mana = evaluate_mana_consultation(
        {"score": 74, "pd_pct": 1.0},
        {"final": "承認", "reasoning": "財務と物件価値が安定している"},
        {"level": "pass"},
        mode="solo",
    )

    assert mana["consulted"] is False
    assert mana["protected_value"] == ""
