"""/api/future-simulation の疎通と、数値予測の記録先分離のテスト。"""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import prediction_snapshot
from api.prediction_snapshot import load_prediction_snapshot, record_numeric_forecast
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
    # 3〜5年後の実績を持っていないため採点できないことを明示する
    assert entry["calibratable"] is False
    assert entry["calibration_blocker"] == "future_actuals_not_collected"


def test_numeric_forecast_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_snapshot, "NUMERIC_FORECASTS_PATH", tmp_path / "forecasts.jsonl")

    assert record_numeric_forecast(case_id="", forecast={"method": "gbm"})["status"] == "skipped"
    assert record_numeric_forecast(case_id="case-1", forecast={})["status"] == "skipped"
