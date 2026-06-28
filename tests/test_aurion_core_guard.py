from api.aurion_core_guard import build_aurion_core_guard


def test_qrisk_is_discipline_signal_not_score_deduction():
    core = build_aurion_core_guard(
        {"competitor": "競合なし", "main_bank": "非メイン先", "asset_name": "工作機械"},
        {
            "score": 76,
            "score_base": 76,
            "score_borrower": 74,
            "asset_score": 70,
            "quantum_risk": 42,
            "credit_quantum_strong_warning": False,
        },
    )

    assert core["mode"] == "discipline_not_deduction"
    assert core["math_discipline"]["score_should_not_be_auto_deducted"] is True
    assert core["math_discipline"]["q_risk_role"] == "discovery_signal"
    assert core["severity"] == "caution"
    assert any(flag["key"] == "high_score_high_qrisk" for flag in core["discipline_flags"])
    assert core["emotion_synapse"]["vigilance"] >= 70
    assert "減点ではなく" in core["shion_ux_message"]


def test_strong_warning_stops_condition_escape():
    core = build_aurion_core_guard(
        {"competitor": "競合あり", "main_bank": "メイン先", "asset_name": ""},
        {
            "score": 72,
            "score_borrower": 40,
            "asset_score": 82,
            "quantum_risk": 60,
            "credit_quantum_strong_warning": True,
            "mahalanobis_score": 75,
            "umap_anomaly_score": 76,
        },
    )

    assert core["severity"] == "stop"
    keys = {flag["key"] for flag in core["discipline_flags"]}
    assert "credit_quantum_strong_warning" in keys
    assert "asset_borrower_gap" in keys
    assert "pricing_competition" in keys
    assert any("承認可否を急がず" in action for action in core["next_actions"])
