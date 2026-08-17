"""将来財務シミュレーション計算部（UI非依存）のテスト。"""
from __future__ import annotations

import pytest

from future_simulation_core import run_business_simulation, simulation_to_json


def test_returns_none_for_non_positive_sales():
    assert run_business_simulation(0.0, 100.0, 0.01, 0.15) is None
    assert run_business_simulation(-500.0, 100.0, 0.01, 0.15) is None


def test_simulation_shape_matches_requested_years():
    result = run_business_simulation(100_000.0, 5_000.0, 0.01, 0.15, years=5, n_simulations=500)

    assert result is not None
    assert list(result["years"]) == [0, 1, 2, 3, 4, 5]
    # 0期目を含むので years+1 点
    assert len(result["sales_percentiles"][50]) == 6
    assert len(result["op_percentiles"][50]) == 6
    assert result["method"] == "gbm"


def test_first_period_starts_at_current_sales():
    result = run_business_simulation(100_000.0, 5_000.0, 0.01, 0.15, years=3, n_simulations=500)

    assert result is not None
    assert result["sales_percentiles"][50][0] == pytest.approx(100_000.0)


def test_percentile_bands_are_ordered():
    result = run_business_simulation(100_000.0, 5_000.0, 0.02, 0.2, years=5, n_simulations=1000)

    assert result is not None
    for index in range(6):
        assert result["sales_percentiles"][10][index] <= result["sales_percentiles"][50][index]
        assert result["sales_percentiles"][50][index] <= result["sales_percentiles"][90][index]


def test_higher_volatility_widens_the_band():
    calm = run_business_simulation(100_000.0, 5_000.0, 0.01, 0.05, years=5, n_simulations=2000)
    wild = run_business_simulation(100_000.0, 5_000.0, 0.01, 0.35, years=5, n_simulations=2000)

    assert calm is not None and wild is not None
    calm_width = calm["sales_percentiles"][90][-1] - calm["sales_percentiles"][10][-1]
    wild_width = wild["sales_percentiles"][90][-1] - wild["sales_percentiles"][10][-1]
    assert wild_width > calm_width


def test_deficit_probability_is_a_probability():
    result = run_business_simulation(100_000.0, -2_000.0, 0.0, 0.2, years=5, n_simulations=1000)

    assert result is not None
    assert 0.0 <= result["deficit_prob"] <= 1.0


def test_json_conversion_has_no_numpy_types():
    result = run_business_simulation(100_000.0, 5_000.0, 0.01, 0.15, years=3, n_simulations=500)
    payload = simulation_to_json(result)

    assert payload is not None
    assert payload["unit"] == "千円"
    assert all(isinstance(year, int) for year in payload["years"])
    assert all(isinstance(value, float) for value in payload["sales_percentiles"]["50"])
    assert isinstance(payload["deficit_prob"], float)
    # パーセンタイルのキーは JSON 用に文字列化されている
    assert set(payload["sales_percentiles"]) == {"10", "25", "50", "75", "90"}


def test_json_conversion_passes_through_none():
    assert simulation_to_json(None) is None
