"""/api/future-simulation の疎通と、数値予測の記録先分離のテスト。"""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import prediction_snapshot
from api.prediction_snapshot import load_prediction_snapshot, record_numeric_actual, record_numeric_forecast
from api.routers.analytics import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_returns_percentile_bands_in_thousand_yen():
    response = _client().post(
        "/api/future-simulation",
        json={"sales": 100_000, "op_profit": 5_000, "years": 5, "n_simulations": 500},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["unit"] == "千円"
    assert payload["years"] == [0, 1, 2, 3, 4, 5]
    assert len(payload["sales_percentiles"]["50"]) == 6
    assert payload["recorded"]["status"] == "skipped"


def test_rejects_non_positive_sales():
    response = _client().post("/api/future-simulation", json={"sales": 0})

    assert response.status_code == 200
    assert response.json() == {"available": False, "reason": "sales_must_be_positive"}


def test_caps_simulation_count_to_protect_the_worker():
    response = _client().post(
        "/api/future-simulation",
        json={"sales": 100_000, "n_simulations": 10_000_000},
    )

    assert response.status_code == 422


def test_numeric_forecast_is_recorded_apart_from_judgment_prediction(tmp_path, monkeypatch):
    """数値予測が判断予測スナップショットを上書きしないこと。"""
    snapshots = tmp_path / "prediction_snapshots.jsonl"
    forecasts = tmp_path / "prediction_numeric_forecasts.jsonl"
    monkeypatch.setattr(prediction_snapshot, "SNAPSHOTS_PATH", snapshots)
    monkeypatch.setattr(prediction_snapshot, "NUMERIC_FORECASTS_PATH", forecasts)

    prediction_snapshot.record_prediction_snapshot(
        case_id="case-1",
        inputs={"industry_sub": "金属加工"},
        result={"score": 52.0, "hantei": "要審議", "main_concern": "自己資本比率が低い"},
        final_status="未登録",
    )
    recorded = record_numeric_forecast(
        case_id="case-1",
        forecast={"method": "gbm", "unit": "千円", "years": [0, 1], "deficit_prob": 0.2},
        assumptions={"drift": 0.01},
    )

    assert recorded["status"] == "recorded"
    # 判断予測は数値予測に上書きされない
    snapshot = load_prediction_snapshot("case-1")
    assert snapshot is not None
    assert snapshot["event_type"] == "prediction_snapshot"
    assert snapshot["prediction"]["main_concern"] == "自己資本比率が低い"

    entry = json.loads(forecasts.read_text(encoding="utf-8").strip())
    assert entry["event_type"] == "numeric_forecast_snapshot"
    # 実績が戻るまでは採点できないことを明示する
    assert entry["calibratable"] is False
    assert entry["calibration_blocker"] == "future_actuals_not_collected"


def test_numeric_forecast_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_snapshot, "NUMERIC_FORECASTS_PATH", tmp_path / "forecasts.jsonl")

    assert record_numeric_forecast(case_id="", forecast={"method": "gbm"})["status"] == "skipped"
    assert record_numeric_forecast(case_id="case-1", forecast={})["status"] == "skipped"


def test_numeric_actual_is_recorded_for_later_shadow_scoring(tmp_path):
    actuals = tmp_path / "prediction_numeric_actuals.jsonl"

    recorded = record_numeric_actual(
        case_id="case-1",
        observed_year=1,
        sales=112_000,
        op_profit=6_500,
        observed_at="2027-03-31",
        actuals_path=actuals,
    )

    assert recorded["status"] == "recorded"
    entry = json.loads(actuals.read_text(encoding="utf-8").strip())
    assert entry["event_type"] == "numeric_forecast_actual"
    assert entry["status"] == "shadow_only"
    assert entry["unit"] == "千円"
    assert entry["actual"] == {"sales": 112_000.0, "op_profit": 6_500.0}
    assert record_numeric_actual(
        case_id="case-1",
        observed_year=1,
        sales=float("nan"),
        actuals_path=actuals,
    ) == {"status": "skipped", "reason": "actual_values_invalid"}


def test_numeric_actual_api_validates_and_records(tmp_path, monkeypatch):
    actuals = tmp_path / "prediction_numeric_actuals.jsonl"
    monkeypatch.setattr(prediction_snapshot, "NUMERIC_ACTUALS_PATH", actuals)

    invalid = _client().post(
        "/api/future-simulation/actuals",
        json={"case_id": "case-1", "observed_year": 1},
    )
    assert invalid.status_code == 422

    response = _client().post(
        "/api/future-simulation/actuals",
        json={
            "case_id": "case-1",
            "observed_year": 1,
            "sales": 112_000,
            "op_profit": 6_500,
            "observed_at": "2027-03-31",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"
