from api.gunshi_gemini import build_strategy_cards


def _params() -> dict:
    return {
        "score": 56.5,
        "industry_cat": "製造業",
        "company_name": "本件",
        "asset_name": "工作機械",
        "equity_ratio": 5,
        "op_profit": 100,
        "nenshu": 10000,
    }


def test_strategy_cards_use_yanami_body_copy():
    cards = build_strategy_cards(_params(), ["返済原資を明確化する"], 0.4, 0.55, humor_style="yanami")

    assert "私が泣く" in cards["today_moves"][0]
    assert cards["risk_cards"][0].startswith("【ここで刺される】")
    assert "雑には止められない" in cards["ringi_lines"][-1]
    assert "つん子" in cards["disclaimer"]


def test_strategy_cards_use_yukikaze_body_copy():
    cards = build_strategy_cards(_params(), ["返済原資を明確化する"], 0.4, 0.55, humor_style="yukikaze")

    assert cards["today_moves"][0].startswith("PILOT TASK 1:")
    assert cards["risk_cards"][0].startswith("JAM SIGNATURE:")
    assert cards["competitor_moves"][0].startswith("COUNTER-VECTOR:")
    assert cards["ringi_lines"][-1].startswith("RINGI LOG:")
    assert "YUKIKAZE" in cards["disclaimer"]
