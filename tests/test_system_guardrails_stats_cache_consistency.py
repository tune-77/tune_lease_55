import data_cases
from system_guardrails import _audit_stats_cache_consistency


def test_no_issue_when_no_cache_persisted_yet(monkeypatch):
    monkeypatch.setattr(data_cases, "load_dashboard_stats_cache", lambda: None)
    monkeypatch.setattr(data_cases, "load_department_stats_cache", lambda: None)

    issues = _audit_stats_cache_consistency()

    assert issues == []


def test_no_issue_when_cache_matches_live_values(monkeypatch):
    monkeypatch.setattr(
        data_cases, "load_dashboard_stats_cache", lambda: {"analysis": {"closed_count": 42}}
    )
    monkeypatch.setattr(
        data_cases, "build_dashboard_stats_cache", lambda: {"analysis": {"closed_count": 42}}
    )
    monkeypatch.setattr(
        data_cases,
        "load_department_stats_cache",
        lambda: {"overall": {"total_count": 10}},
    )
    monkeypatch.setattr(
        data_cases,
        "build_department_stats_cache",
        lambda: {"overall": {"total_count": 10}},
    )

    issues = _audit_stats_cache_consistency()

    assert issues == []


def test_dashboard_cache_drift_is_flagged_as_error(monkeypatch):
    monkeypatch.setattr(
        data_cases, "load_dashboard_stats_cache", lambda: {"analysis": {"closed_count": 40}}
    )
    monkeypatch.setattr(
        data_cases, "build_dashboard_stats_cache", lambda: {"analysis": {"closed_count": 45}}
    )
    monkeypatch.setattr(data_cases, "load_department_stats_cache", lambda: None)

    issues = _audit_stats_cache_consistency()

    assert len(issues) == 1
    assert issues[0]["severity"] == "error"
    assert issues[0]["kind"] == "stats_cache_drift"
    assert "40" in issues[0]["message"] and "45" in issues[0]["message"]


def test_department_cache_drift_is_flagged_as_error(monkeypatch):
    monkeypatch.setattr(data_cases, "load_dashboard_stats_cache", lambda: None)
    monkeypatch.setattr(
        data_cases,
        "load_department_stats_cache",
        lambda: {"overall": {"total_count": 10}},
    )
    monkeypatch.setattr(
        data_cases,
        "build_department_stats_cache",
        lambda: {"overall": {"total_count": 13}},
    )

    issues = _audit_stats_cache_consistency()

    assert len(issues) == 1
    assert issues[0]["severity"] == "error"
    assert issues[0]["kind"] == "stats_cache_drift"
