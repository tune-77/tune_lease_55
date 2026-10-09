"""実際の審査経路で、適用ルールと実効加減点を確認する（外部モデルは固定）。"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scoring_core as sc
from api.schemas import ScoringResponse


@pytest.fixture
def fixed_scoring(monkeypatch):
    monkeypatch.setattr(sc, "get_effective_coeffs", lambda _: {"intercept": 0})
    monkeypatch.setattr(sc, "_load_benchmarks", lambda: {})
    monkeypatch.setattr(sc, "_resolve_benchmark", lambda *_: {"lease_cost_ratio": 1.0})
    monkeypatch.setattr(sc, "_load_capex_lease_data", lambda: {})
    monkeypatch.setattr(sc, "build_estat_context", lambda **_: None)
    monkeypatch.setattr(sc, "_load_lgb_qual_bundle", lambda: None)
    monkeypatch.setattr(sc, "generate_default_warnings", lambda _: [])
    monkeypatch.setattr(sc, "generate_asset_warnings", lambda *_: ([], []))
    monkeypatch.setenv("ENABLE_SYNC_SCORING_DIAGNOSTICS", "0")
    monkeypatch.setitem(sys.modules, "scoring.predict_one", SimpleNamespace(predict_one=lambda **_: {"ai_prob": 0.5}, map_industry_major_to_scoring=lambda x: x))
    monkeypatch.setitem(sys.modules, "credit_risk_detector", SimpleNamespace(detect_credit_risk_group=lambda _: {"flag": False}))
    monkeypatch.setitem(sys.modules, "quantum_analysis_module", SimpleNamespace(compute_simple_q_risk=lambda _: {"quantum_risk": 0}))
    return {"nenshu": 1000, "total_assets": 100, "net_assets": 20, "intuition": 3}


@pytest.mark.parametrize("changes, expected_score, adjustments", [
    ({}, 50.0, {}),
    ({"net_assets": -20}, 40.0, {"SCORING-EQUITY": -10.0}),
    ({"net_assets": -200}, 20.0, {"SCORING-EQUITY": -30.0}),
    ({"rent_expense": 18}, 48.5, {"SCORING-LEASE-RATIO": -1.5}),
    ({"rent_expense": 40}, 47.0, {"SCORING-LEASE-RATIO": -3.0}),
    ({"rent_expense": 25}, 50.0, {}),
    ({"intuition": 5}, 53.0, {"SCORING-INTUITION": 3.0}),
    ({"intuition": 1}, 47.0, {"SCORING-INTUITION": -3.0}),
    ({"company_no": "900303"}, 35.0, {"SCORING-DEMO-CAP": -15.0}),
])
def test_applied_ids_and_existing_scores(fixed_scoring, changes, expected_score, adjustments):
    result = sc.run_full_api_scoring({**fixed_scoring, **changes})
    assert result["borrower_model"] == "rf"
    assert result["score"] == expected_score
    assert result["hantei"] == "要審議"
    reasons = result["judgment_reasons"]
    assert {r["asset_id"]: r["score_delta"] for r in reasons if r["effect"] == "adjustment"} == adjustments
    assert result["score_borrower"] + sum(r["score_delta"] for r in reasons) == pytest.approx(expected_score)
    assert any(r["asset_id"] == "SCORING-APPROVAL-LINE" for r in reasons)
    assert all(r["asset_version"] == 1 and r["summary"] and r["applied_reason"] for r in reasons)
    # APIのシリアライズでもID・版・審査時点の説明が残る。
    response = ScoringResponse.model_validate(result).model_dump()
    assert response["judgment_reasons"] == reasons


def test_clipping_records_effective_delta(fixed_scoring, monkeypatch):
    monkeypatch.setitem(sys.modules, "scoring.predict_one", SimpleNamespace(predict_one=lambda **_: {"ai_prob": 0.01}, map_industry_major_to_scoring=lambda x: x))
    result = sc.run_quick_scoring({**fixed_scoring, "intuition": 5})
    assert result["score"] == 100
    reason = next(r for r in result["judgment_reasons"] if r["asset_id"] == "SCORING-INTUITION")
    assert reason["score_delta"] == 1.0


def test_warning_gates_do_not_change_score(fixed_scoring, monkeypatch):
    monkeypatch.setattr(sc, "generate_default_warnings", lambda _: ["財務警告"])
    monkeypatch.setattr(sc, "generate_asset_warnings", lambda *_: (["物件警告"], ["換金性高"]))
    monkeypatch.setitem(sys.modules, "scoring.predict_one", SimpleNamespace(predict_one=lambda **_: {"ai_prob": 0.1}, map_industry_major_to_scoring=lambda x: x))
    monkeypatch.setitem(sys.modules, "credit_risk_detector", SimpleNamespace(detect_credit_risk_group=lambda _: {"flag": True, "level": "high", "reasons": ["信用警告"]}))
    monkeypatch.setitem(sys.modules, "quantum_analysis_module", SimpleNamespace(compute_simple_q_risk=lambda _: {"quantum_risk": 60}))
    result = sc.run_quick_scoring(fixed_scoring)
    assert result["score"] == 90
    assert result["hantei"] == "要審議"
    assert result["score_based_hantei"] == "承認圏内"
    reasons = {r["asset_id"]: r for r in result["judgment_reasons"]}
    for asset_id in ["SCORING-REVIEW-PATTERN", "SCORING-REVIEW-CREDIT", "SCORING-REVIEW-QUANTUM", "SCORING-ASSET-RISK", "SCORING-ASSET-STRENGTH"]:
        assert reasons[asset_id]["score_delta"] == 0


def test_asset_update_preserves_old_snapshot(fixed_scoring, monkeypatch, tmp_path):
    assets = json.loads((Path(sc._SCRIPT_DIR) / "static_data/scoring_judgment_assets.json").read_text())
    path = tmp_path / "static_data/scoring_judgment_assets.json"
    path.parent.mkdir()
    path.write_text(json.dumps(assets))
    monkeypatch.setattr(sc, "_SCRIPT_DIR", str(tmp_path))
    original = sc.run_quick_scoring(fixed_scoring)["judgment_reasons"][-1]
    assets["SCORING-APPROVAL-LINE"].update(asset_version=2, summary="変更後")
    path.write_text(json.dumps(assets))
    updated = sc.run_quick_scoring(fixed_scoring)["judgment_reasons"][-1]
    assert original["asset_version"] == 1
    assert original["summary"] != updated["summary"]
    assert updated["asset_version"] == 2
    path.unlink()
    assert sc.run_quick_scoring(fixed_scoring)["score"] == 50


@pytest.mark.parametrize("endpoint", ["calculate_score", "calculate_score_full"])
def test_api_and_case_save_keep_basis(fixed_scoring, monkeypatch, tmp_path, endpoint):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("USE_LEGACY_STREAMLIT_FULL_SCORE", "0")
    import api.main as main
    import api.aurion_core_guard as guard
    import data_cases
    from fastapi import BackgroundTasks
    from api.schemas import ScoringRequest

    result = sc.run_quick_scoring({**fixed_scoring, "net_assets": -20})
    monkeypatch.setattr(main, "run_quick_scoring", lambda _: result.copy())
    monkeypatch.setattr(main, "run_full_api_scoring", lambda _: result.copy())
    for name in ["_build_rate_proposal", "_build_data_source_summary", "_build_screening_context_notes", "_build_approval_comment_draft", "_build_financial_consistency_risk", "_build_bayes_reverse_strategy"]:
        monkeypatch.setattr(main, name, lambda *_: {})
    monkeypatch.setattr(main, "_build_conditional_approval_actions", lambda *_: [])
    monkeypatch.setattr(main, "_record_scoring_memory_usage", lambda *_: None)
    monkeypatch.setattr(guard, "build_aurion_core_guard", lambda *_: {})
    saved = []

    def save_case(case):
        saved.append(case)
        return "case-test"

    monkeypatch.setattr(data_cases, "save_case_log", save_case)
    response = getattr(main, endpoint)(ScoringRequest(), BackgroundTasks()).model_dump()
    assert response["judgment_reasons"] == result["judgment_reasons"]
    if endpoint == "calculate_score_full":
        assert response["case_id"] == "case-test"
        assert saved[0]["result"]["judgment_reasons"] == result["judgment_reasons"]
