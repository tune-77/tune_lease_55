"""
P2-002: POST /predict — aurion.q_risk フィールド追加
SPEC: specs/phase2/P2-002-api-aurion-field.md
AC-501〜AC-508 に対応するテスト。
"""
import sys
import os
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mobile_app"))

SPEC_ID = "P2-002"
PHASE = 2

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
    "depreciation": 3,
}

# AC-502: 粗利率120%（矛盾あり）
_CONTRADICTED_REQUEST = {
    **_MINIMAL_REQUEST,
    "gross_profit": 120,
    "nenshu": 100,
    "op_profit": 10,
    "net_income": 5,
}

# AC-503: 正常財務データ（矛盾なし）
_NORMAL_REQUEST = {
    **_MINIMAL_REQUEST,
    "gross_profit": 30,
    "nenshu": 100,
    "op_profit": 10,
    "net_income": 5,
    "lease_credit": 25,
    "bank_credit": 20,
    "acquisition_cost": 8,
    "machines": 0,
    "depreciation": 3,
}

# AC-508: FIN-CONTRADICT-004（リース残高/年商超過）
_LC_EXCEED_REQUEST = {
    **_MINIMAL_REQUEST,
    "gross_profit": 30,
    "nenshu": 100,
    "op_profit": 10,
    "net_income": 5,
    "lease_credit": 60,
    "bank_credit": 10,
    "acquisition_cost": 8,
    "machines": 0,
    "depreciation": 3,
}


@pytest.fixture(scope="module")
def client():
    import api
    api.app.config["TESTING"] = True
    with api.app.test_client() as c:
        yield c


def _post(client, body):
    return client.post("/predict", json=body)


# ── AC-501: aurion.q_risk フィールドが存在する ────────────────────────────
def test_501_aurion_field_present(client):
    rv = _post(client, _MINIMAL_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    assert "aurion" in data, "aurion キーが存在しない"
    assert "q_risk" in data["aurion"], "aurion.q_risk キーが存在しない"
    q = data["aurion"]["q_risk"]
    assert "score" in q
    assert "level" in q
    assert "patterns" in q
    assert "pattern_details" in q


# ── AC-502: 粗利率120%矛盾 → score > 0, FIN-CONTRADICT-001 ───────────────
def test_502_contradicted_finance_score_positive(client):
    rv = _post(client, _CONTRADICTED_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    q = data["aurion"]["q_risk"]
    assert q["score"] > 0, f"score が 0 のまま: {q}"
    assert "FIN-CONTRADICT-001" in q["patterns"], f"FIN-CONTRADICT-001 が検知されなかった: {q['patterns']}"


def test_502b_fullwidth_numbers_are_normalized_for_q_risk(client):
    body = {
        **_CONTRADICTED_REQUEST,
        "gross_profit": "１２０",
        "nenshu": "１００",
        "op_profit": "１０",
    }
    rv = _post(client, body)
    assert rv.status_code == 200
    q = rv.get_json()["aurion"]["q_risk"]
    assert q["score"] > 0
    assert "FIN-CONTRADICT-001" in q["patterns"]


# ── AC-503: 正常財務データ → level == "ok", patterns == [] ───────────────
def test_503_normal_finance_level_ok(client):
    rv = _post(client, _NORMAL_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    q = data["aurion"]["q_risk"]
    assert q["level"] == "ok", f"level が ok でない: {q['level']}"
    assert q["patterns"] == [], f"patterns が空でない: {q['patterns']}"


# ── AC-504: 既存フィールドが aurion 追加後も変化しない ───────────────────
def test_504_existing_fields_unchanged(client):
    rv = _post(client, _MINIMAL_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    for field in ("score", "judgment", "sys_score_b", "recommended_rate"):
        assert field in data, f"既存フィールド {field!r} が消えている"
    assert isinstance(data["score"], int)
    assert isinstance(data["judgment"], str)
    assert isinstance(data["sys_score_b"], float)
    assert isinstance(data["recommended_rate"], float)


# ── AC-505: aurion モジュール未ロード時も 200 で成功 ─────────────────────
def test_505_fallback_when_aurion_not_loaded(client):
    import api
    with patch.object(api, "_detect_q_risk", None):
        rv = _post(client, _MINIMAL_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    q = data["aurion"]["q_risk"]
    assert q["score"] == 0, f"score が 0 でない: {q['score']}"
    assert q["level"] == "ok", f"level が ok でない: {q['level']}"


# ── AC-506: GET /health に aurion_module_loaded フィールドが含まれる ──────
def test_506_health_aurion_module_loaded(client):
    rv = client.get("/health")
    assert rv.status_code == 200
    data = rv.get_json()
    assert "aurion_module_loaded" in data, "aurion_module_loaded キーが /health に存在しない"
    assert isinstance(data["aurion_module_loaded"], bool)


# ── AC-507: caution レベル時に [api.aurion] ログが出力される ─────────────
def test_507_log_output_on_caution(client, capsys):
    rv = _post(client, _CONTRADICTED_REQUEST)
    assert rv.status_code == 200
    captured = capsys.readouterr()
    assert "[api.aurion]" in captured.out, (
        f"[api.aurion] プレフィックスのログが出力されなかった。"
        f"stdout: {captured.out!r}"
    )


# ── AC-508: FIN-CONTRADICT-004 の pattern_details に values が含まれる ────
def test_508_pattern_details_values(client):
    rv = _post(client, _LC_EXCEED_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    q = data["aurion"]["q_risk"]
    assert "FIN-CONTRADICT-004" in q["patterns"], (
        f"FIN-CONTRADICT-004 が検知されなかった: {q['patterns']}"
    )
    detail = next(
        (d for d in q["pattern_details"] if d["code"] == "FIN-CONTRADICT-004"),
        None,
    )
    assert detail is not None, "FIN-CONTRADICT-004 の pattern_details が存在しない"
    values = detail.get("values", {})
    assert "lease_credit" in values, f"values に lease_credit がない: {values}"
    assert "nenshu" in values, f"values に nenshu がない: {values}"
    assert "ratio" in values, f"values に ratio がない: {values}"
