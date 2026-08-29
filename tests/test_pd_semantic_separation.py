from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_score_result_does_not_publish_hybrid_probability_as_pd():
    source = (ROOT / "components" / "score_calculation.py").read_text(encoding="utf-8")

    assert '"pd_percent": None' in source
    assert '"model_risk_probability_percent": _model_risk_probability_percent' in source
    assert '"pd_percent": round(' not in source


def test_user_facing_model_probability_is_explicitly_not_pd():
    streamlit_source = (ROOT / "components" / "analysis_results.py").read_text(encoding="utf-8")
    next_source = (
        ROOT / "frontend" / "src" / "components" / "analysis" / "AdvancedAnalysis.tsx"
    ).read_text(encoding="utf-8")

    assert "PDではありません" in streamlit_source
    assert "PDではありません" in next_source
