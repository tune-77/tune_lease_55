from __future__ import annotations

from report_generator import generate_full_report_from_res


def _session_state() -> dict:
    return {
        "rep_company": "テスト製作所",
        "last_submitted_inputs": {
            "nenshu": 1000,
            "rieki": 20,
            "acquisition_cost": 100,
            "lease_term": 60,
            "bank_credit": 200,
        },
    }


def test_report_does_not_fabricate_pd_from_score_when_pd_missing():
    report = generate_full_report_from_res(
        {"score": 56.5, "industry_major": "E 製造業"},
        _session_state(),
    )

    assert "デフォルト確率（PD）: 未算出" in report
    assert "スコアからの推定表示は行いません" in report
    assert "8.00%" not in report
    assert "PDは未算出です" in report


def test_report_uses_pd_percent_only_when_present():
    report = generate_full_report_from_res(
        {"score": 56.5, "industry_major": "E 製造業", "pd_percent": 3.25},
        _session_state(),
    )

    assert "デフォルト確率（PD）: 3.25%" in report
    assert "未算出" not in report
