from api.shion_conscience import build_conscience_prompt_block, evaluate_conscience


def test_conscience_prompt_defines_shion_conscience():
    block = build_conscience_prompt_block()

    assert "良心の紫苑" in block
    assert "結論を甘くする役ではありません" in block
    assert "ユーザーに迎合せず" in block


def test_conscience_review_for_rejection():
    check = evaluate_conscience(
        {"company_name": "テスト工業", "score": 32, "pd_pct": 6.2, "lease_amount": 80},
        {"final": "否決", "reasoning": "資金繰りが弱く返済原資の説明が不足している", "conditions": []},
    )

    assert check["triggered"] is True
    assert check["level"] == "review"
    assert check["action"] == "表現修正または追加確認"
    assert any("否決理由" in c for c in check["cautions"])
    assert any("数字外の事情" in c for c in check["cautions"])


def test_conscience_pass_when_low_impact_approval():
    check = evaluate_conscience(
        {"company_name": "テスト商事", "score": 72, "pd_pct": 1.1, "lease_amount": 10},
        {"final": "承認", "reasoning": "財務・物件ともに安定している", "conditions": []},
    )

    assert check["triggered"] is False
    assert check["level"] == "pass"
    assert check["action"] == "記録のみ"
