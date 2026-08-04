from api import shion_proactive_alert


def test_no_alert_when_errors_below_threshold(monkeypatch):
    monkeypatch.setattr(
        "lease_intelligence_tools.get_recent_errors",
        lambda hours, limit: {"total_error_lines": 3, "patterns": []},
    )

    result = shion_proactive_alert.check_shion_proactive_alerts()

    assert result["has_alert"] is False
    assert result["message"] is None
    assert result["total_error_lines"] == 3


def test_alert_when_errors_spike(monkeypatch):
    monkeypatch.setattr(
        "lease_intelligence_tools.get_recent_errors",
        lambda hours, limit: {
            "total_error_lines": 12,
            "patterns": [{"pattern": "ConnectionError", "count": 8, "sample": "..."}],
        },
    )

    result = shion_proactive_alert.check_shion_proactive_alerts()

    assert result["has_alert"] is True
    assert "ConnectionError" in result["message"]
    assert "8件" in result["message"]
    assert result["total_error_lines"] == 12
