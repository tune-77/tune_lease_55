"""
P1-002: POST /predict API — warnings[] / rule_check_status フィールド追加
SPEC: specs/phase1/P1-002-api-warnings-field.md
"""
import sys
import os
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mobile_app"))

SPEC_ID = "P1-002"
PHASE = 1

_MINIMAL_REQUEST = {
    "gross_profit": 100,
    "op_profit": 20,
    "ord_profit": 18,
    "net_income": 12,
    "dep_expense": 5,
    "nenshu": 500,
    "acquisition_cost": 10,
    "lease_credit": 5,
    "bank_credit": 50,
    "machines": 30,
    "lease_term": 60,
    "contracts": 3,
    "customer_type": "既存先",
}


@pytest.fixture(scope="module")
def client():
    import api
    api.app.config["TESTING"] = True
    with api.app.test_client() as c:
        yield c


def _post(client, body):
    return client.post("/predict", json=body)


# AC-201: warnings キーが常にレスポンスに含まれる
def test_201_warnings_key_present(client):
    rv = _post(client, _MINIMAL_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    assert "warnings" in data
    assert isinstance(data["warnings"], list)


# AC-202: rule_check_status キーが常にレスポンスに含まれる
def test_202_rule_check_status_key_present(client):
    rv = _post(client, _MINIMAL_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    assert "rule_check_status" in data
    assert isinstance(data["rule_check_status"], str)


# AC-203: 既存フィールドが変化しない
def test_203_existing_fields_unchanged(client):
    rv = _post(client, _MINIMAL_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    for field in ("score", "probability", "judgment", "spread_pred",
                  "base_rate", "recommended_rate", "rate_range"):
        assert field in data, f"既存フィールド {field!r} が消えている"

    assert isinstance(data["score"], int)
    assert isinstance(data["probability"], float)
    assert isinstance(data["judgment"], str)
    assert isinstance(data["rate_range"], dict)
    assert "low" in data["rate_range"]
    assert "high" in data["rate_range"]


# AC-204: 電子計算機 72ヶ月 → TERM_EXCEEDS_LEGAL_LIFE, high_risk
def test_204_term_exceeds_legal_life(client):
    body = {**_MINIMAL_REQUEST, "asset_type": "電子計算機", "lease_term": 72}
    rv = _post(client, body)
    assert rv.status_code == 200
    data = rv.get_json()
    codes = [w["code"] for w in data["warnings"]]
    assert "TERM_EXCEEDS_LEGAL_LIFE" in codes
    assert data["rule_check_status"] == "high_risk"


# AC-205: 工作機械 60ヶ月 + 保険付保済み → warnings=[], ok
def test_205_all_checks_pass(client):
    body = {
        **_MINIMAL_REQUEST,
        "asset_type": "工作機械",
        "lease_term": 60,
        "insurance_applicable": "付保済",
    }
    rv = _post(client, body)
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["warnings"] == []
    assert data["rule_check_status"] == "ok"


# AC-206: check_lease_rules が例外 → 200, score 正常, skipped, warnings=[]
def test_206_rule_check_exception_fallback(client):
    with patch("lease_rule_checks.check_lease_rules", side_effect=RuntimeError("mock error")):
        rv = _post(client, _MINIMAL_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["rule_check_status"] == "skipped"
    assert data["warnings"] == []
    assert isinstance(data["score"], int)
    assert 0 <= data["score"] <= 100


# AC-207: asset_type なし → 200, 後方互換
def test_207_no_asset_type_backward_compat(client):
    body = {k: v for k, v in _MINIMAL_REQUEST.items() if k != "asset_type"}
    rv = _post(client, body)
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["warnings"] == []
    for field in ("score", "probability", "judgment"):
        assert field in data


# AC-208: is_re_lease=true + re_lease_insurance="未付保" → RE_LEASE_INSURANCE_NOT_COVERED
def test_208_re_lease_insurance_warning(client):
    body = {
        **_MINIMAL_REQUEST,
        "is_re_lease": True,
        "re_lease_insurance": "未付保",
    }
    rv = _post(client, body)
    assert rv.status_code == 200
    data = rv.get_json()
    codes = [w["code"] for w in data["warnings"]]
    assert "RE_LEASE_INSURANCE_NOT_COVERED" in codes
