"""
動的金利提案エンジン MVP
Phase1 最優先・失注率-15%想定

スコア帯・競合状況・ブレークイーン閾値を組み合わせた競争的金利提案を生成する。
Streamlit非依存の純計算モジュール（UI / API / Slack Bot から共通利用可）。

【アルゴリズム概要】
1. スコアからPD（デフォルト確率）を推定
2. ブレークイーン = PD × LGD × (1 + cushion) → これを最低採算スプレッドとする
3. スコア帯に応じた基準スプレッドを算出（優良: -0.10%、高リスク: +0.30%）
4. 競合金利がある場合は競合対応戦略を決定
5. ボルツマン温度で3シナリオ（守り/推奨/強気）を生成
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ── スコア帯定義 ──────────────────────────────────────────────────────────────
# score >= min の最初のティアが適用される（上から評価）
_SCORE_TIERS: list[dict] = [
    {
        "label": "優良",
        "color": "#16a34a",
        "min": 85,
        "spread_adj": -0.10,
        "action": "compete_premium",
    },
    {
        "label": "良好",
        "color": "#2563eb",
        "min": 71,
        "spread_adj": 0.00,
        "action": "compete",
    },
    {
        "label": "要検討",
        "color": "#d97706",
        "min": 60,
        "spread_adj": +0.15,
        "action": "caution",
    },
    {
        "label": "高リスク",
        "color": "#dc2626",
        "min": 0,
        "spread_adj": +0.30,
        "action": "pass",
    },
]

# 日本リース業界標準パラメータ
_DEFAULT_LGD = 0.60      # Loss Given Default: 回収不能率 60%
_DEFAULT_CUSHION = 0.30  # 安全バッファ 30%
_DEFAULT_BASE_SPREAD = 2.00  # 基準スプレッド（過去データなし時の初期値）


@dataclass
class RateScenario:
    label: str
    emoji: str
    rate: float
    spread: float
    win_prob_est: float
    description: str
    color: str


@dataclass
class RateProposal:
    """動的金利提案エンジンの出力"""
    # 推奨金利
    optimal_rate: float
    optimal_spread: float

    # ブレークイーン（最低採算ライン）
    break_even_rate: float
    break_even_spread: float

    # スコア情報
    score: float
    pd_est: float           # 推定デフォルト確率
    score_tier: str
    score_tier_color: str

    # 競合状況
    has_competitor: bool
    competitor_rate: float
    competitive_action: str  # compete_premium / compete / caution / pass

    # 3シナリオ
    scenarios: list[RateScenario] = field(default_factory=list)

    # アクション説明
    reasoning: str = ""
    action_summary: str = ""

    # メタ情報
    base_rate: float = 0.0
    data_count: int = 0
    method: str = "dynamic_engine_v1"


# ── 内部ユーティリティ ────────────────────────────────────────────────────────

def _score_to_pd(score: float) -> float:
    """スコアから推定デフォルト確率（PD）を計算する。
    score=71(承認ライン) → PD≈5%、score=50 → PD≈18%、score=85 → PD≈2%
    """
    z = -(score - 55.0) / 13.0
    if z > 500:
        return 1.0
    if z < -500:
        return 0.0
    return round(1.0 / (1.0 + math.exp(-z)), 4)


def _get_score_tier(score: float) -> dict:
    for tier in _SCORE_TIERS:
        if score >= tier["min"]:
            return tier
    return _SCORE_TIERS[-1]


def _win_prob_estimate(
    proposed_rate: float,
    competitor_rate: float,
    base_rate: float,
    score: float,
) -> float:
    """成約確率を簡易推定する。
    競合金利との差とスコアから計算。
    """
    if competitor_rate <= 0:
        return float(np.clip(0.50 + (score - 50.0) / 120.0, 0.30, 0.92))

    rate_diff = proposed_rate - competitor_rate  # 正 = 我々が高い
    # ロジスティック曲線: rate_diff=0 → 50%, -0.2% → 70%, +0.2% → 30%
    base_prob = 1.0 / (1.0 + math.exp(rate_diff * 8.0))
    # スコアボーナス: 優良顧客は少し高くても選ばれやすい（最大+15%）
    score_bonus = max(0.0, (score - 71.0) / 100.0 * 0.15)
    return float(np.clip(base_prob + score_bonus, 0.05, 0.95))


def _boltzmann_temperature(n: int) -> float:
    """データ量に応じたボルツマン温度（多いほど収束）"""
    return max(0.05, 0.25 / math.log(n + 2))


# ── メイン計算 ────────────────────────────────────────────────────────────────

def compute_proposal(
    score: float,
    base_rate: float,
    competitor_rate: float = 0.0,
    term_months: int = 60,
    past_spreads: list[float] | None = None,
) -> RateProposal:
    """
    動的金利提案を計算する（Streamlit非依存）。

    Args:
        score:            審査スコア (0-100)
        base_rate:        基準金利 (%)
        competitor_rate:  競合提示金利 (%, 0=不明)
        term_months:      リース期間（月）
        past_spreads:     過去成約案件のスプレッドリスト（あれば精度向上）

    Returns:
        RateProposal
    """
    score = float(np.clip(score, 0.0, 100.0))
    base_rate = max(0.0, float(base_rate))
    competitor_rate = max(0.0, float(competitor_rate))
    n = len(past_spreads) if past_spreads else 0

    tier = _get_score_tier(score)
    pd_est = _score_to_pd(score)

    # ── ブレークイーン計算 ──────────────────────────────────────────────────
    break_even_spread = pd_est * _DEFAULT_LGD * (1.0 + _DEFAULT_CUSHION)
    break_even_spread = round(max(break_even_spread, 0.05), 3)
    break_even_rate = round(base_rate + break_even_spread, 2)

    # ── 基準スプレッド ────────────────────────────────────────────────────
    if past_spreads and n >= 3:
        hist_base = float(np.percentile(past_spreads, 50))  # 中央値
    else:
        hist_base = _DEFAULT_BASE_SPREAD

    base_spread = hist_base + tier["spread_adj"]
    # ブレークイーン + 最低利幅 5bp を確保
    base_spread = round(max(base_spread, break_even_spread + 0.05), 3)

    # ── 競合対応ロジック ───────────────────────────────────────────────────
    has_competitor = competitor_rate > 0.0
    action = tier["action"]
    reasoning = ""
    action_summary = ""
    optimal_spread: float

    if has_competitor:
        comp_spread = competitor_rate - base_rate
        rate_gap = (base_rate + base_spread) - competitor_rate  # 正 = 我々が高い

        if action == "pass":
            # 高リスク → ブレークイーン以下には絶対に下げない
            optimal_spread = base_spread
            action_summary = "⛔ 競合対応不可：信用リスクにより採算ラインを下回ります"
            reasoning = (
                f"スコア {score:.0f}（高リスク帯）：推定PD {pd_est:.1%} のため"
                f"ブレークイーン {break_even_rate:.2f}% 未満への提案は推奨しません。"
            )

        elif action == "compete_premium":
            # 優良顧客 → 競合比 +0.1% 以内でも取れる可能性あり
            target_spread = max(break_even_spread + 0.05, comp_spread + 0.10)
            if target_spread > comp_spread + 0.30:
                target_spread = comp_spread + 0.20
            optimal_spread = round(target_spread, 3)
            if rate_gap > 0.3:
                action_summary = "🏆 優良顧客：競合比 +0.1% 以内で対抗可能"
            else:
                action_summary = "💪 優良顧客：競合より有利な条件"
            reasoning = (
                f"競合金利 {competitor_rate:.2f}% に対し当社基準 {base_rate + base_spread:.2f}%"
                f"（差: {rate_gap:+.2f}%）。優良顧客のため小幅値引きで対応可能。"
            )

        elif action == "compete":
            # 良好 → 競合金利以下に設定
            target_spread = max(break_even_spread + 0.05, comp_spread - 0.05)
            optimal_spread = round(target_spread, 3)
            if rate_gap > 0:
                action_summary = "✅ 競合対応：競合金利 −0.05% で勝負できます"
            else:
                action_summary = "✅ 良好顧客：現状でも成約圏内"
            reasoning = (
                f"競合金利 {competitor_rate:.2f}%。良好スコアのため価格対応可。"
                f"推奨スプレッド +{round(target_spread, 2):.2f}%（競合差: {rate_gap:+.2f}%）。"
            )

        else:  # caution
            # 要検討 → ブレークイーン確保が最優先、競合対応に限界あり
            target_spread = max(break_even_spread + 0.05, comp_spread - 0.10)
            optimal_spread = round(target_spread, 3)
            if base_rate + target_spread <= competitor_rate:
                action_summary = "⚠️ 要注意：採算確保ギリギリで対応可"
            else:
                action_summary = "⚠️ 要注意：採算維持のため競合対応に限界あり"
            reasoning = (
                f"競合金利 {competitor_rate:.2f}%。推定PD {pd_est:.1%} のため"
                f"ブレークイーン {break_even_rate:.2f}% の確保が優先。"
            )

    else:
        # 競合なし → スコア帯に応じた標準提案
        optimal_spread = base_spread
        action_summary = f"📊 スタンダード提案（{tier['label']}顧客・競合情報なし）"
        reasoning = (
            f"スコア {score:.0f}（{tier['label']}帯）、競合情報なし。"
            f"過去実績ベースのスプレッド +{base_spread:.2f}% を提案。"
        )

    optimal_rate = round(base_rate + optimal_spread, 2)

    # ── 3シナリオ生成（ボルツマン温度） ───────────────────────────────────
    T = _boltzmann_temperature(n)
    spread_range = max(0.4, _DEFAULT_BASE_SPREAD + tier["spread_adj"])
    offset = round(T * spread_range * 0.4, 3)

    s_cons = round(max(break_even_spread, optimal_spread - offset), 3)
    s_aggr = round(optimal_spread + offset, 3)

    wp_cons = _win_prob_estimate(base_rate + s_cons, competitor_rate, base_rate, score)
    wp_opt  = _win_prob_estimate(optimal_rate, competitor_rate, base_rate, score)
    wp_aggr = _win_prob_estimate(base_rate + s_aggr, competitor_rate, base_rate, score)

    scenarios = [
        RateScenario(
            label="守り",
            emoji="🛡️",
            rate=round(base_rate + s_cons, 2),
            spread=s_cons,
            win_prob_est=wp_cons,
            color="#16a34a",
            description="成約優先。確実に取りに行く場合。",
        ),
        RateScenario(
            label="推奨",
            emoji="⚖️",
            rate=optimal_rate,
            spread=optimal_spread,
            win_prob_est=wp_opt,
            color="#2563eb",
            description="採算と成約のバランス点。",
        ),
        RateScenario(
            label="強気",
            emoji="⚔️",
            rate=round(base_rate + s_aggr, 2),
            spread=s_aggr,
            win_prob_est=wp_aggr,
            color="#dc2626",
            description="利幅優先。優位性が高い場合。",
        ),
    ]

    return RateProposal(
        optimal_rate=optimal_rate,
        optimal_spread=optimal_spread,
        break_even_rate=break_even_rate,
        break_even_spread=break_even_spread,
        score=score,
        pd_est=pd_est,
        score_tier=tier["label"],
        score_tier_color=tier["color"],
        has_competitor=has_competitor,
        competitor_rate=competitor_rate,
        competitive_action=action,
        scenarios=scenarios,
        reasoning=reasoning,
        action_summary=action_summary,
        base_rate=base_rate,
        data_count=n,
        method="dynamic_engine_v1",
    )
