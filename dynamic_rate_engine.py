"""
動的金利提案エンジン MVP（Phase1）

【概要】
LightGBM が算出した PD（倒産確率）と Monte Carlo プライシングを統合し、
スコア・競合・業種・期間を考慮した多因子ダイナミックプライシングを行う。

【失注率削減の仕組み】
- PD を正しく反映したリスクプレミアムを計算し、過剰な金利設定を防止
- スコアが高い案件では積極的に低スプレッドを提示して成約率を改善
- 競合金利を意識したスプレッド上限制約を設ける
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RateProposal:
    """動的金利提案の結果"""
    recommended_rate: float
    spread: float
    base_rate: float
    success_prob: float
    pd_percent: float
    lgd_percent: float
    expected_spread: float
    risk_premium: float
    status: str
    confidence: str  # "high" / "medium" / "low"
    pd_source: str   # "lgbm" / "score_fallback"
    scenarios: list[dict] = field(default_factory=list)
    mc_yield_curves: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def pd_from_score(score: float) -> float:
    """スコアから年間倒産確率（小数）を経験則で推定するフォールバック。

    実務的な目安:
        score >= 80 → PD ≈ 1%
        score == 60 → PD ≈ 5%
        score == 40 → PD ≈ 15%
        score <= 25 → PD ≈ 30%
    """
    x = (score - 60.0) * (-0.08)
    return float(np.clip(1.0 / (1.0 + math.exp(-x)), 0.005, 0.40))


def compute_risk_premium(pd_percent: float, lgd_percent: float = 80.0, lease_term_months: int = 60) -> float:
    """PD・LGD・期間から最低限必要なリスクプレミアム（年率%）を算出する。

    簡易式: RP = PD_annual × LGD × √(term/60)
    長期ほど累積リスクが増加するため期間係数で調整する。
    """
    pd_annual = pd_percent / 100.0
    lgd = lgd_percent / 100.0
    term_adj = math.sqrt(max(lease_term_months, 12) / 60.0)
    rp = pd_annual * lgd * term_adj * 100.0
    return round(rp, 3)


def _sigmoid_win_prob(rate: float, r_mid: float, k: float = 1.5) -> float:
    """シグモイド曲線による成約確率推定。rate > r_mid で確率低下。"""
    return float(np.clip(1.0 / (1.0 + np.exp(k * (rate - r_mid))), 0.01, 0.99))


def compute_dynamic_rate_proposal(
    score: float,
    base_rate: float,
    pd_percent: Optional[float] = None,
    competitor_rate: float = 0.0,
    lease_term_months: int = 60,
    lgd_percent: float = 80.0,
    n_trials: int = 10000,
) -> RateProposal:
    """
    動的金利提案エンジン本体。

    Args:
        score: 審査スコア（0–100）
        base_rate: 基準金利（%）
        pd_percent: LightGBM 推定倒産確率（%）。None / 負値の場合はスコアから推定
        competitor_rate: 競合他社の提示金利（%）。0 なら競合なし
        lease_term_months: リース期間（月）
        lgd_percent: Loss Given Default（%）
        n_trials: Monte Carlo 試行回数

    Returns:
        RateProposal dataclass
    """
    # ── 1. PD確定 ──────────────────────────────────────────────────────────────
    if pd_percent is not None and pd_percent >= 0:
        pd_source = "lgbm"
    else:
        pd_percent = pd_from_score(score) * 100.0
        pd_source = "score_fallback"

    # ── 2. リスクプレミアム ─────────────────────────────────────────────────────
    rp = compute_risk_premium(pd_percent, lgd_percent, lease_term_months)

    # ── 3. Monte Carlo 最適プライシング ────────────────────────────────────────
    from montecarlo_pricing import simulate_optimal_yield

    has_competitor = competitor_rate > 0.0
    mc = simulate_optimal_yield(
        pd_percent=pd_percent,
        lease_term_months=lease_term_months,
        lgd_percent=lgd_percent,
        n_trials=n_trials,
        historical_winning_rate=None,
        competitor_rate=competitor_rate if has_competitor else None,
        has_competitor=has_competitor,
    )

    mc_rate: float = float(mc["recommended_yield"])
    mc_success: float = float(mc["success_prob"])
    mc_status: str = mc["status"]

    # ── 4. スコアベース補正 ────────────────────────────────────────────────────
    # スコア80点基準。高スコアほど低金利を提示できる（最大 -0.30%, 最小 +0.50%）
    score_adj = float(np.clip((80.0 - float(score)) * 0.01, -0.30, 0.50))

    # ── 5. 最終レート確定 ──────────────────────────────────────────────────────
    final_rate = mc_rate + score_adj
    min_rate = base_rate + max(rp, 0.05)          # リスクプレミアム下限
    max_rate = base_rate + 6.0                     # 市場上限
    final_rate = float(np.clip(final_rate, min_rate, max_rate))
    spread = round(final_rate - base_rate, 4)

    # ── 6. シナリオ生成 ────────────────────────────────────────────────────────
    T_offset = 0.15
    conservative_spread = float(np.clip(spread - T_offset, max(rp * 0.8, 0.01), 6.0))
    aggressive_spread   = float(np.clip(spread + T_offset, max(rp * 0.8, 0.01), 6.0))

    r_mid = competitor_rate if has_competitor else (base_rate + 2.5)
    k = 2.0 if has_competitor else 1.2

    scenarios: list[dict] = []
    for label, emoji, sp, desc in [
        ("守り",   "🛡️", conservative_spread,
         "成約確率優先。競合優位を取りに行く場合。"),
        ("推奨",   "⚖️", spread,
         f"期待収益最大点（PD={pd_percent:.1f}% / RP={rp:.3f}%込み）。"),
        ("強気",   "⚔️", aggressive_spread,
         "利幅優先。スコアが高く交渉優位な場合。"),
    ]:
        rate_val = round(base_rate + sp, 4)
        wp = _sigmoid_win_prob(rate_val, r_mid, k) if label != "推奨" else round(mc_success / 100.0, 4)
        scenarios.append({
            "label":            label,
            "emoji":            emoji,
            "spread":           round(sp, 4),
            "rate":             rate_val,
            "win_prob":         round(wp, 4),
            "expected_profit":  round(wp * sp, 4),
            "description":      desc,
        })

    # ── 7. 信頼度 ──────────────────────────────────────────────────────────────
    if pd_source == "lgbm" and float(score) >= 60:
        confidence = "high"
    elif pd_source == "lgbm" or float(score) >= 50:
        confidence = "medium"
    else:
        confidence = "low"

    notes: list[str] = []
    if pd_source == "score_fallback":
        notes.append("PD は LightGBM 未利用のためスコアから推定（精度低め）")
    if float(score) < 40:
        notes.append("スコア低（<40）: リスクプレミアムにより金利が高め")
    if has_competitor:
        notes.append(f"競合金利 {competitor_rate:.2f}% を考慮して算出")
    if pd_percent > 20:
        notes.append(f"PD={pd_percent:.1f}% 過大リスク: 保証人・担保検討を推奨")

    return RateProposal(
        recommended_rate=round(final_rate, 4),
        spread=spread,
        base_rate=base_rate,
        success_prob=round(mc_success / 100.0, 4),
        pd_percent=round(pd_percent, 2),
        lgd_percent=lgd_percent,
        expected_spread=round(spread * mc_success / 100.0, 4),
        risk_premium=rp,
        status=mc_status,
        confidence=confidence,
        pd_source=pd_source,
        scenarios=scenarios,
        mc_yield_curves=mc.get("yield_curves", []),
        notes=notes,
    )
