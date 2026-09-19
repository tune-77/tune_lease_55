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


def test_degenerate_none_cache_against_live_value_is_flagged(monkeypatch):
    """成約5件未満で run_contract_driver_analysis() が None を返した結果が
    closed_count=None として固着したケース（2026-09-19 の実障害）。"""
    monkeypatch.setattr(
        data_cases, "load_dashboard_stats_cache", lambda: {"analysis": {"closed_count": None}}
    )
    monkeypatch.setattr(
        data_cases, "build_dashboard_stats_cache", lambda: {"analysis": {"closed_count": 1174}}
    )
    monkeypatch.setattr(data_cases, "load_department_stats_cache", lambda: None)

    issues = _audit_stats_cache_consistency()

    assert len(issues) == 1
    assert issues[0]["severity"] == "error"
    assert issues[0]["kind"] == "stats_cache_drift"
    assert "1174" in issues[0]["message"]


def test_live_none_against_cached_value_is_flagged(monkeypatch):
    """逆方向。キャッシュに値があるのにDB側が集計不能に転じた＝データ欠損の警報。"""
    monkeypatch.setattr(
        data_cases, "load_dashboard_stats_cache", lambda: {"analysis": {"closed_count": 1174}}
    )
    monkeypatch.setattr(
        data_cases, "build_dashboard_stats_cache", lambda: {"analysis": {"closed_count": None}}
    )
    monkeypatch.setattr(data_cases, "load_department_stats_cache", lambda: None)

    issues = _audit_stats_cache_consistency()

    assert len(issues) == 1
    assert issues[0]["kind"] == "stats_cache_drift"


def test_both_none_is_not_drift(monkeypatch):
    """成約が本当に5件未満の新規DB。キャッシュも実値も None で一致するので警報しない。"""
    monkeypatch.setattr(
        data_cases, "load_dashboard_stats_cache", lambda: {"analysis": {"closed_count": None}}
    )
    monkeypatch.setattr(
        data_cases, "build_dashboard_stats_cache", lambda: {"analysis": {"closed_count": None}}
    )
    monkeypatch.setattr(data_cases, "load_department_stats_cache", lambda: None)

    issues = _audit_stats_cache_consistency()

    assert issues == []
