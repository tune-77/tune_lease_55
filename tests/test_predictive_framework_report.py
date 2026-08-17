"""予測的フレームワーク観測レポート（読み取り専用集計）のテスト。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_predictive_framework_report",
    REPO_ROOT / "scripts" / "build_predictive_framework_report.py",
)
report_module = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(report_module)


def _event(
    *,
    case_id: str,
    risk_level: str,
    contracted: bool = False,
    delinquency: bool = False,
    prediction_source: str = "snapshot",
    severity: str = "medium",
    error_type: str = "general_prediction_error",
) -> dict:
    return {
        "event_id": f"pe-{case_id}",
        "case_id": case_id,
        "prediction": {
            "risk_level": risk_level,
            "main_concern": "自己資本比率",
            "recommended_action": "条件付き承認",
            "prediction_source": prediction_source,
        },
        "outcome": {"contracted": contracted, "delinquency": delinquency},
        "prediction_error": {"type": error_type, "severity": severity},
    }


def _candidate(*, candidate_id: str, belief: str, error_type: str, case_id: str = "case-x") -> dict:
    return {
        "candidate_id": candidate_id,
        "case_id": case_id,
        "target_belief": belief,
        "error_type": error_type,
        "severity": "medium",
        "status": "ready_for_review",
        "suggested_update": "懸念を条件付きに弱める。",
    }


def test_coverage_separates_snapshot_from_reconstruction():
    coverage = report_module.build_prediction_coverage(
        [
            _event(case_id="a", risk_level="high"),
            _event(case_id="b", risk_level="low", prediction_source="reconstructed_at_result"),
            _event(case_id="c", risk_level="low", prediction_source="reconstructed_at_result"),
        ]
    )

    assert coverage["observed_events"] == 3
    assert coverage["from_snapshot"] == 1
    assert coverage["coverage_rate"] == 0.333
    assert coverage["trustworthy"] is True


def test_coverage_is_untrustworthy_without_any_snapshot():
    coverage = report_module.build_prediction_coverage(
        [_event(case_id="a", risk_level="high", prediction_source="reconstructed_at_result")]
    )

    assert coverage["trustworthy"] is False
    assert coverage["coverage_rate"] == 0.0


def test_calibration_counts_outcomes_per_risk_band():
    calibration = report_module.build_calibration(
        [
            _event(case_id="a", risk_level="high", delinquency=True),
            _event(case_id="b", risk_level="high", contracted=True),
            _event(case_id="c", risk_level="low", contracted=True),
        ]
    )

    by_risk = {row["risk_level"]: row for row in calibration}
    assert by_risk["high"]["events"] == 2
    assert by_risk["high"]["delinquency_rate"] == 0.5
    assert by_risk["low"]["contract_rate"] == 1.0
    # RISK_ORDER の順で並ぶ（high が先）
    assert [row["risk_level"] for row in calibration] == ["high", "low"]


def _band(risk_level: str, count: int, *, delinquent: int = 0) -> list[dict]:
    return [
        _event(case_id=f"{risk_level}-{i}", risk_level=risk_level, delinquency=i < delinquent, contracted=i >= delinquent)
        for i in range(count)
    ]


def test_warns_when_risk_order_is_inverted():
    calibration = report_module.build_calibration(
        _band("high", 5) + _band("low", 5, delinquent=2),
    )

    warnings = report_module.build_calibration_warnings(calibration)

    assert len(warnings) == 1
    assert warnings[0]["kind"] == "risk_order_inverted"


def test_no_warning_when_risk_order_holds():
    calibration = report_module.build_calibration(
        _band("high", 5, delinquent=2) + _band("low", 5),
    )

    assert report_module.build_calibration_warnings(calibration) == []


def test_no_warning_while_no_delinquency_has_been_observed():
    """延滞0件どうしの比較（0.0 <= 0.0）で警告が出続けないこと。"""
    calibration = report_module.build_calibration(_band("high", 20) + _band("low", 20))

    assert report_module.build_calibration_warnings(calibration) == []


def test_no_warning_on_small_samples():
    calibration = report_module.build_calibration(
        _band("high", 2) + _band("low", 2, delinquent=1),
    )

    assert report_module.build_calibration_warnings(calibration) == []


def test_repeat_beliefs_flag_only_unreviewed_repeats():
    candidates = [
        _candidate(candidate_id="PEC-1", belief="JA-001", error_type="risk_overestimated"),
        _candidate(candidate_id="PEC-2", belief="JA-001", error_type="risk_overestimated"),
        _candidate(candidate_id="PEC-3", belief="JA-001", error_type="risk_overestimated"),
        _candidate(candidate_id="PEC-4", belief="JA-002", error_type="confirmation_gap"),
    ]

    rows = report_module.build_repeat_beliefs(candidates, [])
    top = rows[0]
    assert top["target_belief"] == "JA-001"
    assert top["occurrences"] == 3
    assert top["alertable"] is True

    # 全件レビュー済みなら先回り対象から外れる
    reviews = [{"candidate_id": f"PEC-{i}", "decision": "adopted"} for i in (1, 2, 3)]
    reviewed_rows = report_module.build_repeat_beliefs(candidates, reviews)
    assert reviewed_rows[0]["alertable"] is False
    assert reviewed_rows[0]["unreviewed"] == 0


def test_below_threshold_repeats_are_not_alertable():
    candidates = [
        _candidate(candidate_id="PEC-1", belief="JA-009", error_type="confirmation_gap"),
        _candidate(candidate_id="PEC-2", belief="JA-009", error_type="confirmation_gap"),
    ]

    rows = report_module.build_repeat_beliefs(candidates, [])

    assert rows[0]["occurrences"] == 2
    assert rows[0]["alertable"] is False


def test_review_rate_counts_only_reviewed_candidates():
    candidates = [
        _candidate(candidate_id="PEC-1", belief="JA-001", error_type="risk_overestimated"),
        _candidate(candidate_id="PEC-2", belief="JA-002", error_type="confirmation_gap"),
    ]
    reviews = [{"candidate_id": "PEC-1", "decision": "adopted"}]

    summary = report_module.build_candidate_review_rate(candidates, reviews)

    assert summary["total_candidates"] == 2
    assert summary["reviewed"] == 1
    assert summary["rate"] == 0.5
    assert summary["by_decision"] == {"adopted": 1}


def test_daily_focus_calls_out_missing_snapshots_first():
    report = report_module.build_predictive_framework_report(
        snapshot_rows=[],
        event_rows=[_event(case_id="a", risk_level="high", prediction_source="reconstructed_at_result")],
        candidate_rows=[],
        review_rows=[],
    )

    assert "事前予測が1件も記録されていない" in report["daily_review_focus"][0]


def test_daily_focus_is_quiet_when_nothing_is_wrong():
    report = report_module.build_predictive_framework_report(
        snapshot_rows=[{"case_id": "a", "prediction": {"risk_level": "high"}}],
        event_rows=[
            _event(case_id="a", risk_level="high", delinquency=True),
            _event(case_id="b", risk_level="low", contracted=True),
        ],
        candidate_rows=[],
        review_rows=[],
    )

    assert report["daily_review_focus"] == ["予測の並びと候補レビューに未処理の異常なし。今日は何もしない。"]


def test_report_is_renderable_and_empty_safe():
    report = report_module.build_predictive_framework_report(
        snapshot_rows=[],
        event_rows=[],
        candidate_rows=[],
        review_rows=[],
    )
    markdown = report_module.render_markdown(report)

    assert report["policy"] == "read_only_no_automatic_scoring_or_rag_change"
    assert "予測的フレームワーク観測レポート" in markdown
    assert "観測イベントなし" in markdown
