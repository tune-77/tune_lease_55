"""
P3-002: POST /predict — aurion.competitor_pressure フィールド追加
SPEC: specs/phase3/P3-002-api-competitor-pressure-field.md
AC-801〜AC-808 に対応するテスト。
"""
import sys
import os
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mobile_app"))

SPEC_ID = "P3-002"
PHASE = 3

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

# AC-806: competitor=0 かつ spread が低い → COMP-STEALTH-001 → level=caution
_STEALTH_CAUTION_REQUEST = {
    **_MINIMAL_REQUEST,
    "competitor": "競合なし",
    "gross_profit": 30,
    "nenshu": 1000,
    "op_profit": 10,
    "net_income": 5,
    "machines": 0,
    "depreciation": 3,
    "lease_credit": 5,
    "bank_credit": 10,
    "acquisition_cost": 5,
}

# AC-804: spread が範囲外になりうる入力（ns=0 で spread_pred→中央値2.5）
_SPREAD_CLIP_REQUEST = {
    **_MINIMAL_REQUEST,
    "gross_profit": 0,
    "op_profit": 0,
    "ord_profit": 0,
    "net_income": 0,
    "nenshu": 0,
    "machines": 0,
    "bank_credit": 0,
    "lease_credit": 0,
    "dep_expense": 0,
    "depreciation": 0,
    "acquisition_cost": 0,
}


@pytest.fixture(scope="module")
def client():
    import api
    api.app.config["TESTING"] = True
    with api.app.test_client() as c:
        yield c


def _post(client, body):
    return client.post("/predict", json=body)


# ── AC-801: aurion.competitor_pressure フィールドが存在する ──────────────
def test_801_competitor_pressure_field_present(client):
    rv = _post(client, _MINIMAL_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    assert "aurion" in data, "aurion キーが存在しない"
    assert "competitor_pressure" in data["aurion"], "aurion.competitor_pressure キーが存在しない"
    cp = data["aurion"]["competitor_pressure"]
    assert "score" in cp, "competitor_pressure.score がない"
    assert "level" in cp, "competitor_pressure.level がない"
    assert "patterns" in cp, "competitor_pressure.patterns がない"
    assert "pattern_details" in cp, "competitor_pressure.pattern_details がない"


# ── AC-802: aurion.q_risk フィールドが変化していない ─────────────────────
def test_802_q_risk_field_unchanged(client):
    rv = _post(client, _MINIMAL_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    assert "aurion" in data
    assert "q_risk" in data["aurion"], "aurion.q_risk キーが消えている"
    q = data["aurion"]["q_risk"]
    assert "score" in q
    assert "level" in q
    assert "patterns" in q
    assert "pattern_details" in q
    assert isinstance(q["score"], int)
    assert isinstance(q["level"], str)
    assert isinstance(q["patterns"], list)
    assert isinstance(q["pattern_details"], list)


# ── AC-803: stealth_competitor 未ロード時はフォールバック値 ───────────────
def test_803_fallback_when_stealth_not_loaded(client):
    import api
    with patch.object(api, "_stealth_loaded", False):
        rv = _post(client, _MINIMAL_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    cp = data["aurion"]["competitor_pressure"]
    assert cp["score"] == 0, f"score が 0 でない: {cp['score']}"
    assert cp["level"] == "ok", f"level が ok でない: {cp['level']}"
    assert cp["patterns"] == [], f"patterns が空でない: {cp['patterns']}"
    assert cp["pattern_details"] == [], f"pattern_details が空でない: {cp['pattern_details']}"


# ── AC-804: spread_pred がクリップ後の値で渡される ────────────────────────
def test_804_spread_pred_clipped(client):
    """
    spread_pred は max(-0.5, min(6.0, raw)) でクリップされる。
    モデル未ロード時は中央値 2.5 が使われる。いずれも -0.5〜6.0 の範囲内に収まる。
    """
    rv = _post(client, _SPREAD_CLIP_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    spread = data.get("spread_pred", None)
    assert spread is not None, "spread_pred がレスポンスにない"
    assert -0.5 <= spread <= 6.0, f"spread_pred={spread} がクリップ範囲外"
    # competitor_pressure が正常に返っている（クリップ済み値が渡されたことの間接確認）
    assert "competitor_pressure" in data["aurion"]


# ── AC-805: GET /health に stealth_competitor_module_loaded が含まれる ────
def test_805_health_stealth_module_loaded(client):
    rv = client.get("/health")
    assert rv.status_code == 200
    data = rv.get_json()
    assert "stealth_competitor_module_loaded" in data, (
        "stealth_competitor_module_loaded キーが /health に存在しない"
    )
    assert isinstance(data["stealth_competitor_module_loaded"], bool)
    assert data["stealth_competitor_module_loaded"] is True


# ── AC-806: caution/high_risk 時に [api.aurion.stealth] ログが出力される ──
def test_806_log_output_on_caution(client, capsys):
    """
    spread_pred が低くなる入力で COMP-STEALTH-001 が発火し、
    [api.aurion.stealth] プレフィックスのログが出力されることを確認する。
    spread_predictor が未ロードの場合、spread_pred は中央値 2.5 になり COMP-STEALTH-001 は
    発火しない（1.5 超）。その場合は stealth_competitor 内の stealth ログは出ない。
    本テストは spread が低い場合をモックで模倣する。
    """
    import api

    _STEALTH_CAUTION = {"score": 35, "level": "caution", "patterns": ["COMP-STEALTH-001"], "pattern_details": []}
    with patch("api.detect_stealth_competitor", return_value=_STEALTH_CAUTION):
        rv = _post(client, _STEALTH_CAUTION_REQUEST)
    assert rv.status_code == 200
    captured = capsys.readouterr()
    assert "[api.aurion.stealth]" in captured.out, (
        f"[api.aurion.stealth] プレフィックスのログが出力されなかった。"
        f"stdout: {captured.out!r}"
    )


# ── AC-807: 既存スコア（RF/LGBM）の出力値が変化しない ────────────────────
def test_807_existing_scores_unchanged(client):
    rv = _post(client, _MINIMAL_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    for field in ("score", "probability", "judgment", "spread_pred"):
        assert field in data, f"既存フィールド {field!r} が消えている"
    assert isinstance(data["score"], int)
    assert isinstance(data["probability"], float)
    assert isinstance(data["judgment"], str)
    assert isinstance(data["spread_pred"], float)
    assert data["judgment"] in ("承認", "条件付", "否認")


# ── AC-808: competitor_pressure.score が 0〜100 の範囲内 ──────────────────
def test_808_score_range(client):
    rv = _post(client, _MINIMAL_REQUEST)
    assert rv.status_code == 200
    data = rv.get_json()
    score = data["aurion"]["competitor_pressure"]["score"]
    assert isinstance(score, int), f"score が int でない: {type(score)}"
    assert 0 <= score <= 100, f"score={score} が 0〜100 の範囲外"
