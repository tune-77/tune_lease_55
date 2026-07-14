import importlib


screening = importlib.import_module("api.multi_agent_screening")


def test_arbiter_explanation_drops_unknown_truncated_condition_code():
    reasoning, conditions = screening._render_arbiter_explanation(
        {
            "final": "条件付承認",
            "reason_codes": ["finance", "asset_value_"],
            "condition_codes": ["guarantee", "asset_value_"],
        }
    )

    assert "財務指標" in reasoning
    assert "asset_value_" not in reasoning
    assert conditions == ["必要に応じて保証、担保、追加保全を検討する"]


def test_arbiter_explanation_uses_manual_review_when_conditions_are_truncated():
    _, conditions = screening._render_arbiter_explanation(
        {
            "final": "条件付承認",
            "reason_codes": ["finance"],
            "condition_codes": ["asset_value_"],
        }
    )

    assert conditions == ["AI応答または根拠が不完全なため、人手で審査根拠を再確認する"]


def test_persona_explanation_drops_unknown_truncated_codes():
    reasons, extras = screening._render_persona_explanation(
        {
            "opinion": "承認",
            "reason_codes": ["finance", "cashflow_"],
            "opportunity_codes": ["growth", "productivity_"],
        },
        "optimist",
    )

    assert reasons == ["財務指標と返済原資を中心に評価する"]
    assert extras == ["投資により売上成長や受注拡大が見込める"]


def test_trim_generated_sentence_prefers_complete_sentence():
    text = (
        "返済原資と物件保全を確認できる場合は条件付き承認の余地がある。"
        "ただし追加資料の提出と資金繰り確認を条件にする必要があり、"
        "ここから先は画面に出すには長すぎる説明が続く。"
    )

    trimmed = screening._trim_generated_sentence(text, limit=45)

    assert trimmed == "返済原資と物件保全を確認できる場合は条件付き承認の余地がある。"


def test_arbiter_token_budget_default_is_large_enough_for_json_retry():
    assert screening._ARBITER_MAX_TOKENS >= 2048
