import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOBILE_APP = ROOT / "mobile_app"
if str(MOBILE_APP) not in sys.path:
    sys.path.insert(0, str(MOBILE_APP))

from advisor_strategy import _extract_gemini_json, build_gemini_strategy_advice, build_strategy_advice


def test_advisor_high_qrisk_adds_integrity_condition():
    result = {
        "score": 66,
        "judgment": "条件付",
        "recommended_rate": 3.22,
        "base_rate": 2.65,
        "spread_pred": 0.57,
        "quantum_risk": 76.08,
        "streamlit": {
            "credit_risk_group_score": 0.4,
            "credit_risk_group_level": "normal",
        },
        "aurion": {
            "q_risk": {"score": 0, "level": "ok", "patterns": [], "pattern_details": []},
            "competitor_pressure": {"score": 47, "level": "caution", "patterns": ["COMP"], "pattern_details": []},
        },
    }
    case = {
        "industry": "D 建設業",
        "customer_type": "既存先",
        "main_bank": "メイン先",
        "competitor": "競合あり",
        "competitor_rate": 2.8,
        "nenshu": 100,
        "op_profit": 5,
        "acquisition_cost": 5,
    }

    advice = build_strategy_advice(result, case)

    assert advice["decision"]["stance"] == "条件付き前向き"
    assert advice["metrics"]["q_risk"] == 76.08
    assert any("Q_risk" in x for x in advice["risk_points"])
    assert any("決算書原本" in x for x in advice["recommended_conditions"])
    assert any("金利だけ" in x for x in advice["sales_talk"])


def test_advisor_credit_and_qrisk_high_is_strong_warning():
    result = {
        "score": 74,
        "judgment": "承認",
        "quantum_risk": 82,
        "streamlit": {
            "credit_risk_group_score": 88,
            "credit_risk_group_level": "high",
        },
        "aurion": {},
    }

    advice = build_strategy_advice(result, {"customer_type": "既存先"})

    assert advice["decision"]["stance"] == "強警戒・条件厳格化"
    assert advice["decision"]["confidence"] <= 0.52
    assert any("信用リスク群" in x for x in advice["risk_points"])


def test_gemini_advisor_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = {
        "score": 66,
        "judgment": "条件付",
        "quantum_risk": 76,
        "streamlit": {"credit_risk_group_score": 0.4, "credit_risk_group_level": "normal"},
        "aurion": {},
    }

    advice = build_gemini_strategy_advice(result, {"customer_type": "既存先"})

    assert advice["source"] == "rule_fallback"
    assert "GEMINI_API_KEY" in advice["llm_error"]
    assert advice["summary"]


def test_extract_gemini_json_reads_candidate_parts():
    class Part:
        text = '{"summary":"ok","risk_points":["x"]}'

    class Content:
        parts = [Part()]

    class Candidate:
        content = Content()

    class Response:
        text = ""
        parsed = None
        candidates = [Candidate()]

    assert _extract_gemini_json(Response()) == {"summary": "ok", "risk_points": ["x"]}
