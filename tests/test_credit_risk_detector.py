from credit_risk_detector import detect_credit_risk_group


def test_excluded_grade_is_detected_without_db():
    result = detect_credit_risk_group({"grade": "9"}, db_path="/no/such/db.sqlite")

    assert result["available"] is True
    assert result["flag"] is True
    assert result["level"] == "excluded_grade"
    assert result["score"] == 100.0


def test_missing_training_db_returns_unavailable_for_normal_grade():
    result = detect_credit_risk_group({"grade": "4-6"}, db_path="/no/such/db.sqlite")

    assert result["available"] is False
    assert result["flag"] is False
    assert result["level"] == "unavailable"
