"""将来財務シミュレーションの計算部（UI非依存）。

`future_simulation.py` は Streamlit と Plotly をモジュール先頭で import するため、
FastAPI プロセスからは読み込めなかった。計算だけをここへ切り出し、Streamlit 側は
このモジュールを再利用する（描画と入力UIは future_simulation.py に残す）。

単位は千円で統一する（`CLAUDE.md` の数値単位を参照。フロントの百万円は
`toThousandYenPayload()` で千円へ変換済みの値が入ってくる前提）。
"""
from __future__ import annotations

from typing import Any

import numpy as np

# TimesFM（オプション）。未導入環境でも GBM で動く。
try:
    from timesfm_engine import forecast_financial_paths as _tfm_paths, TIMESFM_AVAILABLE as _TIMESFM_AVAILABLE
except ImportError:
    _tfm_paths = None
    _TIMESFM_AVAILABLE = False

PERCENTILES = [10, 25, 50, 75, 90]

# 限界利益率の下限。業種別データを持たないため、営業利益率+10%と30%の高い方を使う。
_MIN_CONTRIBUTION_MARGIN_RATIO = 0.30


def run_business_simulation(
    current_sales: float,
    current_op_profit: float,
    drift: float,
    volatility: float,
    years: int = 5,
    n_simulations: int = 10000,
    use_timesfm: bool = False,
    sales_history: list | None = None,
) -> dict | None:
    """GBM または TimesFM で将来の売上高と営業利益をシミュレーションする。

    Args:
        current_sales: 現在の売上高（千円）
        current_op_profit: 現在の営業利益（千円）
        drift: 年間期待成長率（例: 0.02）
        volatility: 年間ボラティリティ（例: 0.10）
        years: シミュレーション期間（年）
        n_simulations: シミュレーション回数
        use_timesfm: True のとき TimesFM で売上パスを生成（未導入時は GBM へフォールバック）
        sales_history: TimesFM 用の過去売上時系列

    Returns:
        パーセンタイル・赤字確率などを含む dict。売上が0以下なら None。
    """
    if current_sales <= 0:
        return None

    # 簡易的な固定費・変動費の算出（限界利益率と固定費を推計）
    current_op_margin = current_op_profit / current_sales
    contribution_margin_ratio = max(_MIN_CONTRIBUTION_MARGIN_RATIO, current_op_margin + 0.10)

    # 固定費 = 売上高 × 限界利益率 - 営業利益
    fixed_costs = current_sales * contribution_margin_ratio - current_op_profit

    n_periods = years  # 年単位

    if use_timesfm and _TIMESFM_AVAILABLE and _tfm_paths is not None:
        historical = sales_history if sales_history else [current_sales]
        sales_paths = _tfm_paths(
            historical_values=historical,
            n_periods=n_periods,
            n_paths=n_simulations,
            fallback_mu=drift,
            fallback_sigma=volatility,
            dt=1.0,
        )
        method = "timesfm"
    else:
        dt = 1.0  # 年単位
        np.random.seed(42)
        Z = np.random.normal(0, 1, size=(n_simulations, years))
        sales_paths = np.zeros((n_simulations, years + 1))
        sales_paths[:, 0] = current_sales
        for t in range(1, years + 1):
            sales_paths[:, t] = sales_paths[:, t - 1] * np.exp(
                (drift - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * Z[:, t - 1]
            )
        method = "gbm"

    op_profit_paths = sales_paths * contribution_margin_ratio - fixed_costs

    sales_percentiles = {p: np.percentile(sales_paths, p, axis=0) for p in PERCENTILES}
    op_percentiles = {p: np.percentile(op_profit_paths, p, axis=0) for p in PERCENTILES}

    final_op_profits = op_profit_paths[:, -1]
    deficit_prob = float(np.mean(final_op_profits < 0))

    return {
        "years": np.arange(0, years + 1),
        "sales_percentiles": sales_percentiles,
        "op_percentiles": op_percentiles,
        "deficit_prob": deficit_prob,
        "final_op_median": op_percentiles[50][-1],
        "final_op_worst10": op_percentiles[10][-1],
        "method": method,
    }


def simulation_to_json(sim_data: dict | None) -> dict[str, Any] | None:
    """numpy 配列を JSON で返せる素の list / float へ落とす。"""
    if not sim_data:
        return None
    return {
        "years": [int(year) for year in sim_data["years"]],
        "sales_percentiles": {
            str(p): [float(value) for value in values] for p, values in sim_data["sales_percentiles"].items()
        },
        "op_percentiles": {
            str(p): [float(value) for value in values] for p, values in sim_data["op_percentiles"].items()
        },
        "deficit_prob": float(sim_data["deficit_prob"]),
        "final_op_median": float(sim_data["final_op_median"]),
        "final_op_worst10": float(sim_data["final_op_worst10"]),
        "method": str(sim_data["method"]),
        "unit": "千円",
    }
