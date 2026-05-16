from mobile_app import api as mobile_api
from mobile_app.api import _build_mobile_chart_payload


def test_mobile_chart_payload_builds_radar_and_factors():
    response_payload = {
        "score": 82.0,
        "probability": 0.61,
        "base_rate": 2.15,
        "recommended_rate": 2.45,
        "streamlit": {
            "score_borrower": 73.4,
        },
    }
    sc_result = {
        "score_contributions": [
            {"feature": "op_profit", "label_ja": "営業利益", "contribution": 12.3},
            {"feature": "rent", "label_ja": "賃借料", "contribution": -4.5},
            {"feature": "intercept", "label_ja": "切片", "contribution": 1.0},
        ]
    }
    q_risk_result = {"score": 18.0}
    stealth_result = {"score": 6.0}

    charts = _build_mobile_chart_payload(response_payload, sc_result, q_risk_result, stealth_result)

    assert len(charts["radar"]) == 6
    assert charts["radar"][0]["label"] == "総合スコア"
    assert charts["factors"][0]["label"] == "営業利益"
    assert charts["factors"][1]["label"] == "賃借料"
    assert charts["summary"]["q_risk"] == 18.0
    assert charts["summary"]["competitor_pressure"] == 6.0


def test_visual_insights_payload_builds_bubble_only(monkeypatch):
    sample_cases = [
        {
            "final_status": "成約",
            "score": 80,
            "final_rate": 3.2,
            "base_rate_at_time": 2.6,
            "winning_spread": 0.6,
            "industry_major": "D 建設業",
            "industry_sub": "建設",
            "sales_dept": "宇都宮営業部",
            "inputs": {"acquisition_cost": 12},
        },
        {
            "final_status": "失注",
            "score": 60,
            "final_rate": 2.9,
            "base_rate_at_time": 2.5,
            "winning_spread": 0.4,
            "industry_major": "D 建設業",
            "industry_sub": "建設",
            "sales_dept": "宇都宮営業部",
            "inputs": {"acquisition_cost": 10},
        },
        {
            "final_status": "成約",
            "score": 70,
            "final_rate": 3.1,
            "base_rate_at_time": 2.7,
            "winning_spread": 0.4,
            "industry_major": "E 製造業",
            "industry_sub": "製造",
            "sales_dept": "小山営業部",
            "inputs": {"acquisition_cost": 14},
        },
    ]
    monkeypatch.setattr(mobile_api, "load_past_cases", lambda: sample_cases)

    payload = mobile_api._build_visual_insights_payload(current_case={"score": 88, "spread_pred": 0.31})

    assert payload["meta"]["case_count"] == 3
    assert len(payload["bubble"]["points"]) == 3
    assert "heatmap" not in payload
    assert "sankey" not in payload
    assert payload["bubble"]["current"]["score"] == 88


def test_chat_analysis_graphs_include_industry_gaps_and_computed_values():
    input_data = {
        "industry": "D 建設業",
        "gross_profit": 32,
        "op_profit": 5,
        "ord_profit": 4,
        "net_income": 2.5,
        "dep_expense": 1.2,
        "depreciation": 0.8,
        "nenshu": 100,
        "bank_credit": 18,
        "lease_credit": 12,
        "machines": 9,
        "other_assets": 11,
        "net_assets": 40,
        "total_assets": 120,
        "rent": 4,
        "rent_expense": 3,
        "acquisition_cost": 10,
        "contracts": 2,
        "lease_term": 60,
        "proposed_rate": 3.2,
        "competitor_rate": 2.8,
        "lease_asset_score": 88,
        "q_weighted": 76,
    }
    score_result = {
        "base_rate": 2.15,
        "recommended_rate": 2.45,
        "spread_pred": 0.31,
        "score": 82,
        "probability": 0.61,
        "quantum_risk": 18,
        "aurion": {
            "q_risk": {"score": 12},
            "competitor_pressure": {"score": 7},
        },
    }

    payload = mobile_api._build_chat_analysis_graphs("営業利益と業種平均との差を見たい", input_data, score_result)

    assert payload["industry"]["benchmark_found"] is True
    assert any(row["label"] == "営業利益率" for row in payload["ratios"])
    assert any(row["label"] == "自己資本比率" for row in payload["ratios"])
    assert any(item["label"] == "提案金利" for item in payload["computed"])
    assert any(item["label"] == "AURION Q_risk" and item["value"] == 12 for item in payload["computed"])
    assert any(item["label"] == "総合スコア" and item["value"] == 82 for item in payload["computed"])
