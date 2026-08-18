import scoring_anomaly_monitor as monitor
from scoring_anomaly_monitor import detect_scoring_anomalies, record_scoring_anomalies


def _base_result(**overrides):
    result = {
        "score": 75.0,
        "score_borrower": 72.0,
        "hantei": "承認圏内",
        "quantum_risk": 10.0,
        "credit_quantum_strong_warning": False,
        "credit_risk_group_score": 0.0,
    }
    result.update(overrides)
    return result


def test_clean_result_has_no_findings():
    findings = detect_scoring_anomalies(_base_result())

    assert findings == []


def test_missing_score_is_undefined_state_error():
    findings = detect_scoring_anomalies(_base_result(score=None))

    assert any(f.kind == "undefined_state" and f.level == "error" for f in findings)


def test_unexpected_hantei_value_is_undefined_state_error():
    findings = detect_scoring_anomalies(_base_result(hantei="不明"))

    assert any(f.kind == "undefined_state" and f.level == "error" for f in findings)


def test_large_score_gap_is_error():
    findings = detect_scoring_anomalies(_base_result(score=90.0, score_borrower=50.0))

    assert any(f.kind == "score_gap" and f.level == "error" for f in findings)


def test_moderate_score_gap_is_warning_only():
    findings = detect_scoring_anomalies(_base_result(score=80.0, score_borrower=58.0))

    assert any(f.kind == "score_gap" and f.level == "warning" for f in findings)
    assert not any(f.kind == "score_gap" and f.level == "error" for f in findings)


def test_quantum_risk_strong_warning_is_error():
    findings = detect_scoring_anomalies(_base_result(quantum_risk=65.0))

    assert any(f.kind == "quantum_risk" and f.level == "error" for f in findings)


def test_quantum_risk_attention_is_warning_only():
    findings = detect_scoring_anomalies(_base_result(quantum_risk=40.0))

    assert any(f.kind == "quantum_risk" and f.level == "warning" for f in findings)
    assert not any(f.kind == "quantum_risk" and f.level == "error" for f in findings)


def test_credit_quantum_strong_warning_flag_is_error():
    findings = detect_scoring_anomalies(_base_result(credit_quantum_strong_warning=True))

    assert any(f.kind == "credit_quantum_strong_warning" and f.level == "error" for f in findings)


def test_record_scoring_anomalies_logs_and_alerts_once_then_cools_down(monkeypatch):
    logged = []
    alerted = []

    monkeypatch.setattr(monitor, "log_warning", lambda msg, context="": logged.append((context, msg)))
    monkeypatch.setattr("slack_notify.push_notification", lambda text: alerted.append(text))
    monitor._last_alert_at.clear()

    result = _base_result(quantum_risk=65.0)

    record_scoring_anomalies(result, case_label="case-1")
    record_scoring_anomalies(result, case_label="case-1")

    assert len(logged) == 2  # 検知のたびに即時ログは毎回残す
    assert len(alerted) == 1  # クールダウン中はSlack通知を重複させない


def test_record_scoring_anomalies_returns_empty_for_clean_result():
    findings = record_scoring_anomalies(_base_result(), case_label="case-2")

    assert findings == []
