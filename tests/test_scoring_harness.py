import sys
from types import SimpleNamespace

from scripts.run_scoring_harness import SAMPLE_CASES, run_harness, validate_scoring_result


def test_validate_scoring_result_accepts_minimal_valid_result():
    result = {
        "score": 72.0,
        "score_base": 72.0,
        "score_borrower": 72.0,
        "approval_line": 71,
        "user_op_margin": 5.0,
        "user_equity_ratio": 30.0,
        "bench_op_margin": 3.0,
        "bench_equity_ratio": 20.0,
        "asset_score": 60.0,
        "hantei": "承認圏内",
        "comparison": "ok",
        "score_contributions": [],
        "asset_score_warnings": [],
        "credit_risk_warnings": [],
        "asset_warnings": [],
        "default_warnings": [],
    }

    assert validate_scoring_result(result) == []


def test_validate_scoring_result_rejects_hantei_mismatch():
    result = {
        "score": 50.0,
        "score_base": 50.0,
        "score_borrower": 50.0,
        "approval_line": 71,
        "user_op_margin": 5.0,
        "user_equity_ratio": 30.0,
        "bench_op_margin": 3.0,
        "bench_equity_ratio": 20.0,
        "asset_score": 60.0,
        "hantei": "承認圏内",
        "comparison": "ok",
        "score_contributions": [],
        "asset_score_warnings": [],
        "credit_risk_warnings": [],
        "asset_warnings": [],
        "default_warnings": [],
    }

    failures = validate_scoring_result(result)

    assert any("hantei mismatch" in failure for failure in failures)


def test_validate_scoring_result_accepts_risk_review_override():
    result = {
        "score": 90.0,
        "score_base": 90.0,
        "score_borrower": 95.0,
        "approval_line": 71,
        "user_op_margin": -2.0,
        "user_equity_ratio": -10.0,
        "bench_op_margin": 3.0,
        "bench_equity_ratio": 20.0,
        "asset_score": 45.0,
        "hantei": "要審議",
        "risk_review_required": True,
        "comparison": "債務超過のため要審議",
        "score_contributions": [],
        "asset_score_warnings": [],
        "credit_risk_warnings": [],
        "asset_warnings": [],
        "default_warnings": ["高リスク財務パターンとの類似警告"],
    }

    assert validate_scoring_result(result) == []


def test_scoring_harness_smoke_cases_pass(monkeypatch):
    def fake_run_quick_scoring(inputs):
        risk_review_required = float(inputs.get("net_assets", 0) or 0) < 0
        return {
            "score": 72.0,
            "score_base": 72.0,
            "score_borrower": 72.0,
            "approval_line": 71,
            "user_op_margin": 5.0,
            "user_equity_ratio": 30.0,
            "bench_op_margin": 3.0,
            "bench_equity_ratio": 20.0,
            "asset_score": inputs.get("asset_score", 50),
            "hantei": "要審議" if risk_review_required else "承認圏内",
            "risk_review_required": risk_review_required,
            "comparison": "ok",
            "score_contributions": [],
            "asset_score_warnings": [],
            "credit_risk_warnings": [],
            "asset_warnings": [],
            "default_warnings": [],
            "credit_risk_group_level": "normal",
        }

    monkeypatch.setitem(sys.modules, "scoring_core", SimpleNamespace(run_quick_scoring=fake_run_quick_scoring))

    report = run_harness(SAMPLE_CASES)

    assert report["total"] == len(SAMPLE_CASES)
    assert report["failed"] == 0
    assert report["passed"] == len(SAMPLE_CASES)
