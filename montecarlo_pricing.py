"""
montecarlo_pricing.py
モンテカルロ法を用いたリース利回り（プライシング）最適化エンジン
"""
import numpy as np
import pandas as pd
from base_rate_master import get_base_rate_by_term

def simulate_optimal_yield(
    pd_percent: float,
    lease_term_months: int = 60,
    lgd_percent: float = 80.0,
    n_trials: int = 10000,
    historical_winning_rate: float | None = None,
    competitor_rate: float | None = None,
    has_competitor: bool = False
) -> dict:
    """
    モンテカルロシミュレーションを実行し、期待収益を最大化する最適な提案利回りを算出する。
    【修正】リース市場の現実的な上限金利（相場の天井）を考慮し、非現実的な高金利を排除する。
    """
    base_yield = get_base_rate_by_term(lease_term_months=lease_term_months)
    if base_yield is None:
        base_yield = 2.66 

    pd_rate = np.clip(pd_percent / 100.0, 0.0, 1.0)
    lgd_rate = np.clip(lgd_percent / 100.0, 0.0, 1.0)

    # 1. 黄金のプライシング基準点 (r_mid) の決定
    if competitor_rate and competitor_rate > 0:
        r_mid = competitor_rate
        k = 2.0
    elif has_competitor:
        base_target = historical_winning_rate if (historical_winning_rate and historical_winning_rate > 0) else (base_yield + 2.5)
        r_mid = max(base_yield + 0.2, base_target - 0.5)
        k = 1.5
    else:
        r_mid = historical_winning_rate if (historical_winning_rate and historical_winning_rate > 0) else (base_yield + 2.5)
        k = 1.2

    # 【実務制約】日本のリース市場における現実的な上限金利（スプレッド +5.0% 程度）
    market_ceiling = base_yield + 5.0
    
    # テスト範囲を現実的な天井までに絞る
    test_yields = np.arange(base_yield + 0.1, market_ceiling + 0.1, 0.1)
    
    best_yield = base_yield + 2.0
    max_expected_profit = -999.0
    best_success_prob = 0.0
    yield_curves = []

    for r in test_yields:
        # A. 顧客の成約確率
        success_prob = 1.0 / (1.0 + np.exp(k * (r - r_mid)))
        
        # B. モンテカルロ試行
        defaults = np.random.rand(n_trials) < pd_rate
        scenario_profits = np.where(
            defaults,
            -lgd_rate * 5,  # 損失のインパクトを実務に即して緩和
            (r - base_yield)
        )
        
        expected_profit = success_prob * np.mean(scenario_profits)
        
        yield_curves.append({
            "yield": np.round(r, 2),
            "success_prob": np.round(success_prob * 100.0, 2),
            "expected_profit": np.round(expected_profit, 4)
        })
        
        if expected_profit > max_expected_profit:
            max_expected_profit = expected_profit
            best_yield = r
            best_success_prob = success_prob

    # 判定ステータス
    status = "🟢 積極提案"
    if pd_percent > 25.0:
        # 倒産リスクが過大な場合、金利を上げても回収不能なため「否決推奨」寄りの警告
        status = "🔴 慎重（リスク過大・保証検討）"
        # 金利も上限にへばりつかないように実績付近に戻す
        best_yield = min(r_mid + 1.0, market_ceiling)
    elif best_success_prob < 0.30:
        status = "🟡 競合注意 (成約率低め)"
    return {
        "recommended_yield": np.round(best_yield, 2),
        "base_yield": np.round(base_yield, 2),
        "success_prob": np.round(best_success_prob * 100.0, 2),
        "expected_profit_index": np.round(max_expected_profit, 4),
        "yield_curves": yield_curves,
        "status": status
    }
