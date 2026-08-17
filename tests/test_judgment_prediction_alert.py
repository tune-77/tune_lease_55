"""判断領域の先回り提示（予測が繰り返し外れている前提の通知）のテスト。"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from api.shion_proactive_alert import check_judgment_prediction_alert


def _write_report(tmp_path, *, generated_at=None, trustworthy=True, repeat_beliefs=None, warnings=None):
    path = tmp_path / "predictive_framework_latest.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": generated_at or datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                "prediction_coverage": {"trustworthy": trustworthy, "coverage_rate": 0.8},
                "repeat_beliefs": repeat_beliefs if repeat_beliefs is not None else [],
                "calibration_warnings": warnings if warnings is not None else [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_alerts_on_repeated_unreviewed_miss(tmp_path):
    path = _write_report(
        tmp_path,
        repeat_beliefs=[
            {"target_belief": "JA-001", "error_type": "risk_overestimated", "occurrences": 3, "alertable": True},
            {"target_belief": "JA-002", "error_type": "confirmation_gap", "occurrences": 1, "alertable": False},
        ],
    )

    alert = check_judgment_prediction_alert(report_path=path)

    assert alert["has_alert"] is True
    assert "JA-001" in alert["message"]
    assert "JA-002" not in alert["message"]
    assert len(alert["items"]) == 1
    assert alert["policy"] == "read_only_human_review_required"


def test_stays_silent_when_nothing_repeats(tmp_path):
    path = _write_report(tmp_path, repeat_beliefs=[{"target_belief": "JA-001", "occurrences": 1, "alertable": False}])

    alert = check_judgment_prediction_alert(report_path=path)

    assert alert["has_alert"] is False
    assert alert["reason"] == "no_repeat_miss"
    assert alert["message"] is None


def test_stays_silent_when_coverage_is_not_trustworthy(tmp_path):
    """事前予測が0件の集計を根拠に警告しない（逆算を実績のように見せないため）。"""
    path = _write_report(
        tmp_path,
        trustworthy=False,
        repeat_beliefs=[{"target_belief": "JA-001", "occurrences": 5, "alertable": True}],
    )

    alert = check_judgment_prediction_alert(report_path=path)

    assert alert["has_alert"] is False
    assert alert["reason"] == "prediction_coverage_not_trustworthy"


def test_stays_silent_when_report_is_stale(tmp_path):
    stale = (datetime.now(tz=timezone.utc) - timedelta(days=30)).isoformat(timespec="seconds")
    path = _write_report(
        tmp_path,
        generated_at=stale,
        repeat_beliefs=[{"target_belief": "JA-001", "occurrences": 5, "alertable": True}],
    )

    alert = check_judgment_prediction_alert(report_path=path)

    assert alert["has_alert"] is False
    assert alert["reason"] == "report_stale"


def test_stays_silent_when_report_is_missing(tmp_path):
    alert = check_judgment_prediction_alert(report_path=tmp_path / "missing.json")

    assert alert["has_alert"] is False
    assert alert["reason"] == "report_unavailable"


def test_includes_calibration_warning(tmp_path):
    path = _write_report(
        tmp_path,
        warnings=[{"kind": "risk_order_inverted", "message": "高リスク予測の延滞率が低リスク予測を上回っていない。"}],
    )

    alert = check_judgment_prediction_alert(report_path=path)

    assert alert["has_alert"] is True
    assert "高リスク予測" in alert["message"]


def test_limit_caps_reported_items(tmp_path):
    path = _write_report(
        tmp_path,
        repeat_beliefs=[
            {"target_belief": f"JA-{i:03d}", "error_type": "confirmation_gap", "occurrences": 4, "alertable": True}
            for i in range(5)
        ],
    )

    alert = check_judgment_prediction_alert(report_path=path, limit=2)

    assert len(alert["items"]) == 2
