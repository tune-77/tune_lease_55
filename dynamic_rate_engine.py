"""
dynamic_rate_engine.py
動的金利提案エンジン MVP（Phase1）

スプレッド = 基準スプレッド + PD補正 + 業種補正 + スコア補正 + 競合補正
各補正はクランプ済みで逆ザヤを防止。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# デフォルトPD基準値（業界全体の平均的な倒産確率）
_BASELINE_PD_PCT = 5.0
# PD 1%変化あたりのスプレッド調整幅（%）
_PD_RATE_PER_PCT = 0.025
# スコア1点変化あたりのスプレッド調整幅（%）
_SCORE_RATE_PER_PT = 0.003
# スコアの基準点（この値でスコア補正がゼロ）
_SCORE_BASELINE = 60.0


def _industry_spread_adjustment(industry_sub: str, df: pd.DataFrame) -> float:
    """
    業種別の成約スプレッド平均と全体平均の差を業種補正として返す。
    データが不足している場合は 0.0 を返す。
    """
    if df.empty or "spread" not in df.columns or "industry_sub" not in df.columns:
        return 0.0

    won = df[df.get("won", pd.Series(dtype=int)) == 1] if "won" in df.columns else df
    if won.empty:
        return 0.0

    overall_mean = won["spread"].mean()
    industry_df = won[won["industry_sub"] == industry_sub]
    if len(industry_df) < 3:
        return 0.0

    industry_mean = industry_df["spread"].mean()
    adj = industry_mean - overall_mean
    return float(np.clip(adj, -0.20, 0.20))


def compute_dynamic_spread(
    score: float,
    pd_percent: float,
    industry_sub: str,
    base_rate: float,
    competitor_rate: float,
    historical_df: pd.DataFrame,
    base_optimal_spread: float,
    lease_term_months: int = 60,
) -> dict:
    """
    PD・スコア・業種・競合を加味した動的スプレッドを算出する。

    Parameters
    ----------
    score              : 借手スコア（0-100）
    pd_percent         : LightGBM等で推定された倒産確率（%）
    industry_sub       : 業種（細分類）
    base_rate          : 基準金利（%）
    competitor_rate    : 競合他社金利（0なら競合なし）
    historical_df      : 過去案件DataFrame（_load_rate_dataframe() の戻り値）
    base_optimal_spread: suggest_rate() で算出した基準スプレッド
    lease_term_months  : リース期間（ヶ月）

    Returns
    -------
    dict with keys:
        dynamic_spread      : 動的スプレッド（%）
        dynamic_rate        : 動的適用金利（%）
        pd_adjustment       : PD補正値
        score_adjustment    : スコア補正値
        industry_adjustment : 業種補正値
        competitor_adjustment: 競合補正値
        win_rate_estimate   : 予測成約確率（0–1）
        win_rate_improvement: 基準スプレッドとの成約確率差（percentage points）
        recommendation      : 要約テキスト
    """
    # ── 1. PD補正: 業界基準(5%)超 → スプレッド増、未満 → 減（成約率狙い） ──
    pd_adj = float(np.clip(
        (pd_percent - _BASELINE_PD_PCT) * _PD_RATE_PER_PCT,
        -0.30, 0.50
    ))

    # ── 2. スコア補正: 高スコア借手は優遇（スプレッド引き下げ） ──────────
    score_adj = float(np.clip(
        -(score - _SCORE_BASELINE) * _SCORE_RATE_PER_PT,
        -0.20, 0.20
    ))

    # ── 3. 業種補正: 業種別歴史実績スプレッドからの乖離 ────────────────
    ind_adj = _industry_spread_adjustment(industry_sub, historical_df)

    # ── 4. 動的スプレッド（補正前） ───────────────────────────────────────
    raw_spread = base_optimal_spread + pd_adj + score_adj + ind_adj

    # ── 5. 競合補正: 競合金利がある場合は競合比 -0.1% 以内に収める ────
    comp_adj = 0.0
    if competitor_rate > 0 and base_rate > 0:
        comp_spread = competitor_rate - base_rate
        if raw_spread > comp_spread + 0.05:
            comp_adj = (comp_spread + 0.05) - raw_spread
            comp_adj = float(np.clip(comp_adj, -0.50, 0.0))

    dynamic_spread = float(np.clip(raw_spread + comp_adj, 0.05, 5.0))
    dynamic_rate = base_rate + dynamic_spread

    # ── 6. 成約確率推定（ロジスティック近似） ────────────────────────────
    # 競合スプレッドを基準に相対位置から確率推定
    if competitor_rate > 0 and base_rate > 0:
        comp_spread = competitor_rate - base_rate
        gap = comp_spread - dynamic_spread  # 正 → 競合より低い（有利）
        win_rate_est = float(1.0 / (1.0 + np.exp(-3.0 * gap)))
    else:
        # 競合なし: スコアと過去データから推定
        win_rate_est = float(np.clip(0.50 + (score - _SCORE_BASELINE) * 0.006, 0.30, 0.90))

    # ── 7. 基準スプレッドとの成約確率差（改善効果） ─────────────────────
    if competitor_rate > 0 and base_rate > 0:
        comp_spread_v = competitor_rate - base_rate
        gap_base = comp_spread_v - base_optimal_spread
        base_win_rate = float(1.0 / (1.0 + np.exp(-3.0 * gap_base)))
    else:
        base_win_rate = win_rate_est
    win_rate_improvement = win_rate_est - base_win_rate

    # ── 8. 推奨テキスト ──────────────────────────────────────────────────
    parts = []
    if pd_adj > 0.01:
        parts.append(f"リスクプレミアム +{pd_adj:.2f}%（PD {pd_percent:.1f}%）")
    elif pd_adj < -0.01:
        parts.append(f"優良先割引 {pd_adj:.2f}%（PD {pd_percent:.1f}%低リスク）")
    if ind_adj > 0.01:
        parts.append(f"業種割増 +{ind_adj:.2f}%")
    elif ind_adj < -0.01:
        parts.append(f"業種割引 {ind_adj:.2f}%")
    if score_adj < -0.01:
        parts.append(f"高スコア割引 {score_adj:.2f}%")
    if comp_adj < -0.01:
        parts.append(f"競合対応調整 {comp_adj:.2f}%")

    recommendation = "、".join(parts) if parts else "標準スプレッド適用"

    return {
        "dynamic_spread": round(dynamic_spread, 3),
        "dynamic_rate": round(dynamic_rate, 3),
        "pd_adjustment": round(pd_adj, 3),
        "score_adjustment": round(score_adj, 3),
        "industry_adjustment": round(ind_adj, 3),
        "competitor_adjustment": round(comp_adj, 3),
        "win_rate_estimate": round(win_rate_est, 3),
        "win_rate_improvement": round(win_rate_improvement, 3),
        "recommendation": recommendation,
    }


# ── PD推定・リスクプレミアム・シナリオ提案（Phase1 MVP） ──────────────────────
# スコアからPDを推定するモデルが無い案件向けのフォールバックと、
# 3シナリオ（守り/推奨/強気）の金利提案を提供する。

# スコア0点でのPD上限／スコア100点でのPD下限（クランプ）
_PD_SCORE_MAX = 0.40
_PD_SCORE_MIN = 0.005
# スコア→PDの減衰係数（大きいほどスコアに敏感）
_PD_SCORE_DECAY = 0.032
# デフォルトLGD（倒産時損失率）
_DEFAULT_LGD_PCT = 45.0
# スプレッドの絶対上限（逆ザヤ・暴利防止）
_RATE_MAX_SPREAD = 6.0
# この PD（%）以上は高リスクとして注意メッセージを出す
_HIGH_PD_WARNING_PCT = 15.0
# シナリオ名とスプレッド倍率（守り＜推奨＜強気の順を保証）
_SCENARIO_SPREAD_MULTIPLIERS: dict[str, float] = {"守り": 0.7, "推奨": 1.0, "強気": 1.3}


def pd_from_score(score: float) -> float:
    """借手スコア（0-100）から倒産確率PD（0.0-1.0の割合）を推定する。

    LightGBM等の実測PDが無い案件向けの簡易フォールバック。
    スコアが高いほど指数関数的にPDが下がり、[0.005, 0.40] にクランプする。
    """
    s = max(0.0, min(100.0, float(score)))
    pd_value = _PD_SCORE_MAX * float(np.exp(-_PD_SCORE_DECAY * s))
    return float(np.clip(pd_value, _PD_SCORE_MIN, _PD_SCORE_MAX))


def compute_risk_premium(
    pd_percent: float,
    lgd_percent: float = _DEFAULT_LGD_PCT,
    lease_term_months: int = 60,
) -> float:
    """PD・LGD・リース期間からリスクプレミアム（%）を算出する。

    RP = PD × LGD × (期間/60ヶ月) を基準スプレッドへの上乗せ幅（%）として返す。
    60ヶ月を基準期間とし、長期ほどリスク晒露が増えるものとして線形に増加させる。
    """
    pd_ratio = max(0.0, float(pd_percent)) / 100.0
    lgd_ratio = max(0.0, float(lgd_percent)) / 100.0
    term_factor = max(1, int(lease_term_months)) / 60.0
    return round(pd_ratio * lgd_ratio * term_factor * 100.0, 6)


@dataclass
class RateProposal:
    """動的金利提案の結果（推奨シナリオ + 守り/推奨/強気の3シナリオ）。"""

    spread: float
    recommended_rate: float
    base_rate: float
    pd_percent: float
    pd_source: str
    scenarios: list[dict] = field(default_factory=list)
    success_prob: float = 0.0
    notes: list[str] = field(default_factory=list)


def _scenario_win_prob(scenario_spread: float, reference_spread: float) -> float:
    """基準スプレッドとの差から成約確率を推定する（ロジスティック近似）。

    compute_dynamic_spread() の win_rate_estimate と同じ式・係数を用いる。
    """
    gap = reference_spread - scenario_spread
    return float(1.0 / (1.0 + np.exp(-3.0 * gap)))


def compute_dynamic_rate_proposal(
    score: float,
    base_rate: float,
    pd_percent: float | None = None,
    competitor_rate: float = 0.0,
    lease_term_months: int = 60,
    lgd_percent: float = _DEFAULT_LGD_PCT,
) -> RateProposal:
    """PD・スコア・競合を加味した金利提案を3シナリオ（守り/推奨/強気）付きで返す。

    実測PD（LightGBM等）が無い/不正な値の場合はスコアからのフォールバックに切り替える。
    """
    notes: list[str] = []
    if pd_percent is None or pd_percent < 0:
        pd_source = "score_fallback"
        pd_percent = pd_from_score(score) * 100.0
    else:
        pd_source = "lgbm"

    risk_premium = compute_risk_premium(
        pd_percent, lgd_percent=lgd_percent, lease_term_months=lease_term_months
    )
    score_adj = (_SCORE_BASELINE - score) * _SCORE_RATE_PER_PT
    spread = max(risk_premium + score_adj, risk_premium * 0.8)

    # 競合金利がある場合、リスクプレミアムの8割を下限に競合へ寄せる
    if competitor_rate > 0:
        competitor_gap = (base_rate + spread) - competitor_rate
        if competitor_gap > 0:
            spread = max(risk_premium * 0.8, spread - min(competitor_gap, spread) * 0.2)

    spread = float(np.clip(spread, 0.01, _RATE_MAX_SPREAD))
    recommended_rate = round(base_rate + spread, 4)

    scenarios: list[dict] = []
    for label, multiplier in _SCENARIO_SPREAD_MULTIPLIERS.items():
        scenario_spread = min(spread * multiplier, _RATE_MAX_SPREAD)
        win_prob = _scenario_win_prob(scenario_spread, spread)
        scenarios.append({
            "label": label,
            "spread": round(scenario_spread, 4),
            "recommended_rate": round(base_rate + scenario_spread, 4),
            "win_prob": round(win_prob, 4),
        })
    success_prob = scenarios[1]["win_prob"]

    if pd_percent >= _HIGH_PD_WARNING_PCT:
        notes.append(
            f"PD={pd_percent:.1f}% は高リスク水準のため、追加担保・保証・期間短縮等の条件強化を検討する。"
        )

    return RateProposal(
        spread=spread,
        recommended_rate=recommended_rate,
        base_rate=base_rate,
        pd_percent=pd_percent,
        pd_source=pd_source,
        scenarios=scenarios,
        success_prob=success_prob,
        notes=notes,
    )
