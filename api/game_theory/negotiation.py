"""
交渉ゲーム理論モジュール（REV-224 / Game Theory #2）

貸し手（リース会社）と借り手の交渉を二者間バーゲニングとして定式化する。

理論的基盤:
  - ナッシュ交渉解（Nash Bargaining Solution, 1950）:
    双方の効用積を最大化する合意点
  - ルービンシュタイン交渉モデル: 時間圧力下での均衡
  - ZOPA（Zone of Possible Agreement）: 合意可能ゾーンの可視化

出力:
  - nash_solution: 最適金利・期間・担保条件
  - zopa: 合意可能ゾーン
  - threat_points: 決裂時の各プレイヤーの利得
  - recommendation: 紫苑が提案すべき条件
"""
from __future__ import annotations

import math
from typing import Any


# ── 効用関数 ────────────────────────────────────────────────────────────────

def _lender_utility(
    interest_rate: float,     # 年利率(%)
    term_months: int,         # リース期間
    lease_amount: float,      # リース金額（千円）
    collateral_ratio: float,  # 担保率 0.0〜1.0
    credit_score: float,      # 審査スコア 0〜100
) -> float:
    """
    貸し手の効用: 収益性 × 安全性

    収益性: 金利収入（高金利・長期ほど高）
    安全性: 担保率・スコアによる損失期待値の低さ
    """
    # 金利収入の正規化（年利3〜8%を0〜1に）
    rate_utility = min(1.0, max(0.0, (interest_rate - 1.5) / 6.5))

    # 期間効用（12〜60ヶ月を正規化）
    term_utility = min(1.0, max(0.0, (term_months - 12) / 48))

    # 安全性（スコアと担保による損失リスクの低さ）
    default_prob = max(0.0, 1 - (credit_score / 100) * 0.8 - collateral_ratio * 0.2)
    safety_utility = 1.0 - default_prob

    # 貸し手効用 = 収益性 × 安全性の加重平均
    return 0.4 * rate_utility + 0.2 * term_utility + 0.4 * safety_utility


def _borrower_utility(
    interest_rate: float,
    term_months: int,
    lease_amount: float,
    monthly_cashflow: float,  # 月次キャッシュフロー（千円）
    asset_useful_life_months: int = 60,
) -> float:
    """
    借り手の効用: 負担の低さ × 期間の適切さ

    負担: 月次支払額 / 月次キャッシュフロー（低いほど良い）
    期間: 資産の残存耐用年数との整合性
    """
    if lease_amount <= 0 or monthly_cashflow <= 0:
        return 0.0

    # 月次リース料の概算（簡易計算: 元本/期間 × (1 + 金利/12)）
    monthly_rate = interest_rate / 100 / 12
    if monthly_rate > 0:
        monthly_payment = lease_amount * monthly_rate / (1 - (1 + monthly_rate) ** (-term_months))
    else:
        monthly_payment = lease_amount / term_months

    # 支払い負担率（低いほど借り手にとって良い）
    payment_ratio = monthly_payment / monthly_cashflow
    burden_utility = max(0.0, 1.0 - payment_ratio * 2)  # 50%超で効用0

    # 金利負担の低さ（低金利ほど良い）
    rate_utility = max(0.0, 1.0 - (interest_rate - 1.5) / 6.5)

    # 期間適切さ（耐用年数に近いほど良い）
    term_fit = 1.0 - abs(term_months - asset_useful_life_months * 0.8) / asset_useful_life_months
    term_utility = max(0.0, term_fit)

    return 0.4 * burden_utility + 0.4 * rate_utility + 0.2 * term_utility


# ── ナッシュ交渉解 ───────────────────────────────────────────────────────────

def _nash_product(
    lender_u: float,
    borrower_u: float,
    lender_threat: float,
    borrower_threat: float,
) -> float:
    """ナッシュ積 = (貸し手効用 - 脅威点) × (借り手効用 - 脅威点)"""
    return max(0.0, lender_u - lender_threat) * max(0.0, borrower_u - borrower_threat)


def solve_nash_bargaining(
    lease_amount: float,
    credit_score: float,
    monthly_cashflow: float,
    collateral_ratio: float = 0.5,
    asset_useful_life_months: int = 60,
    interest_rate_range: tuple[float, float] = (2.0, 7.5),
    term_range: tuple[int, int] = (24, 60),
    lender_threat_point: float = 0.2,
    borrower_threat_point: float = 0.1,
) -> dict[str, Any]:
    """
    ナッシュ交渉解を数値探索で求める。

    格子探索: 金利 0.25%刻み × 期間 6ヶ月刻み
    """
    best_nash = -1.0
    best_rate = interest_rate_range[0]
    best_term = term_range[0]
    best_lender_u = 0.0
    best_borrower_u = 0.0

    rate_step = 0.25
    term_step = 6

    rate = interest_rate_range[0]
    while rate <= interest_rate_range[1] + 0.001:
        term = term_range[0]
        while term <= term_range[1]:
            lu = _lender_utility(rate, term, lease_amount, collateral_ratio, credit_score)
            bu = _borrower_utility(rate, term, lease_amount, monthly_cashflow, asset_useful_life_months)
            nash = _nash_product(lu, bu, lender_threat_point, borrower_threat_point)

            if nash > best_nash:
                best_nash = nash
                best_rate = rate
                best_term = term
                best_lender_u = lu
                best_borrower_u = bu

            term += term_step
        rate = round(rate + rate_step, 2)

    # ZOPA の計算（貸し手脅威点以上かつ借り手脅威点以上の領域）
    zopa_solutions = []
    rate = interest_rate_range[0]
    while rate <= interest_rate_range[1] + 0.001:
        term = term_range[0]
        while term <= term_range[1]:
            lu = _lender_utility(rate, term, lease_amount, collateral_ratio, credit_score)
            bu = _borrower_utility(rate, term, lease_amount, monthly_cashflow, asset_useful_life_months)
            if lu > lender_threat_point and bu > borrower_threat_point:
                zopa_solutions.append({"rate": rate, "term": term})
            term += term_step
        rate = round(rate + rate_step, 2)

    # 月次支払い額の計算
    monthly_rate = best_rate / 100 / 12
    if monthly_rate > 0:
        monthly_payment = lease_amount * monthly_rate / (1 - (1 + monthly_rate) ** (-best_term))
    else:
        monthly_payment = lease_amount / best_term

    return {
        "nash_solution": {
            "interest_rate": best_rate,
            "term_months": best_term,
            "collateral_ratio": collateral_ratio,
            "monthly_payment_estimate": round(monthly_payment, 0),
        },
        "nash_product": round(best_nash, 4),
        "lender_utility": round(best_lender_u, 3),
        "borrower_utility": round(best_borrower_u, 3),
        "threat_points": {
            "lender": lender_threat_point,
            "borrower": borrower_threat_point,
        },
        "zopa_size": len(zopa_solutions),
        "zopa_exists": len(zopa_solutions) > 0,
        "recommendation": _build_recommendation(
            best_rate, best_term, best_nash, len(zopa_solutions), monthly_payment, monthly_cashflow
        ),
    }


def _build_recommendation(
    rate: float,
    term: int,
    nash: float,
    zopa_size: int,
    monthly_payment: float,
    monthly_cashflow: float,
) -> str:
    payment_ratio = monthly_payment / monthly_cashflow if monthly_cashflow > 0 else 0

    lines = [
        f"ナッシュ交渉解: 金利 {rate:.2f}% / 期間 {term}ヶ月",
        f"月次支払い: {monthly_payment:,.0f}千円（CF比率 {payment_ratio:.1%}）",
    ]

    if zopa_size == 0:
        lines.append("⚠️ ZOPAなし。現条件では合意点が存在しない。担保の積み増しまたは期間延長を検討。")
    elif zopa_size < 5:
        lines.append("合意可能ゾーンは狭い。条件を慎重に提示すること。")
    else:
        lines.append(f"合意可能ゾーンは十分（{zopa_size}通りの条件が合意可能）。")

    if nash > 0.15:
        lines.append("双方の効用が均衡している。この条件で提示を推奨。")
    elif nash > 0.05:
        lines.append("均衡はあるが弱い。もう一方の条件を調整すると改善できる。")
    else:
        lines.append("均衡が非常に弱い。条件の見直しが必要。")

    return " ".join(lines)


# ── ルービンシュタイン均衡（時間圧力版） ─────────────────────────────────────

def rubinstein_equilibrium(
    lender_discount: float = 0.95,   # 貸し手の1期あたりの割引率
    borrower_discount: float = 0.90, # 借り手の1期あたりの割引率（借り手の方が急いでいる）
    total_surplus: float = 1.0,
) -> dict[str, Any]:
    """
    ルービンシュタイン交渉モデルの均衡シェアを計算する。
    借り手が急いでいるほど（割引率が低いほど）、貸し手が有利になる。
    """
    # 均衡シェア: 貸し手が先攻の場合
    # lender_share = (1 - borrower_discount) / (1 - lender_discount * borrower_discount)
    denom = 1 - lender_discount * borrower_discount
    if denom <= 0:
        return {"status": "degenerate"}

    lender_share = (1 - borrower_discount) / denom
    borrower_share = 1 - lender_share

    return {
        "lender_share": round(lender_share, 3),
        "borrower_share": round(borrower_share, 3),
        "lender_discount_rate": lender_discount,
        "borrower_discount_rate": borrower_discount,
        "interpretation": (
            f"交渉力の分配: 貸し手 {lender_share:.1%} / 借り手 {borrower_share:.1%}。"
            f"{'借り手が急いでいるため貸し手優位。' if borrower_discount < lender_discount else '双方均等な交渉力。'}"
        ),
    }
