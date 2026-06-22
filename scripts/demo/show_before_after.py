"""デモ用 Before/After 審査比較スクリプト.

ブルドーザーリース（申し込み：5年）に対して
・改善前: リース期間 5年（誤設定）で計算
・改善後: リース期間 6年（正しい耐用年数）で計算
の月額リース料・審査スコアを並べて表示する。
"""
from __future__ import annotations

import math
import sys


# ─── リース試算 ────────────────────────────────────────────────────────────

def monthly_lease_payment(
    asset_price: float,
    annual_interest_rate: float,
    lease_term_months: int,
    residual_rate: float = 0.0,
) -> float:
    """標準PMT公式でリース月額を計算する（残価なしフルペイアウト）."""
    r = annual_interest_rate / 12.0
    n = lease_term_months
    pv = asset_price * (1.0 - residual_rate)
    if r == 0:
        return pv / n
    return pv * r * (1 + r) ** n / ((1 + r) ** n - 1)


# ─── スコアリング（デモ用簡易版） ─────────────────────────────────────────

def calc_demo_score(
    asset_price: float,
    lease_term_years: int,
    annual_sales: float,
    annual_profit: float,
    lease_term_correct_years: int,
) -> dict[str, float | str]:
    """デモ用の審査スコアを計算する.

    返済能力・物件適正期間・収益性を簡易評価してスコアを算出する。
    """
    monthly_payment = monthly_lease_payment(asset_price, 0.025, lease_term_years * 12)
    annual_payment = monthly_payment * 12

    # 返済負担率（年間支払額 / 売上）
    repayment_ratio = annual_payment / annual_sales if annual_sales > 0 else 1.0

    # 収益性（営業利益率）
    profit_margin = annual_profit / annual_sales if annual_sales > 0 else 0.0

    # 物件期間適正スコア: 正しい耐用年数に近いほど高スコア
    period_diff = abs(lease_term_years - lease_term_correct_years)
    period_score = max(0.0, 100.0 - period_diff * 12)  # 1年ズレごとに-12点

    # 返済能力スコア（返済負担率が低いほど高い）
    repayment_score = max(0.0, min(100.0, (1.0 - repayment_ratio / 0.30) * 100))

    # 収益性スコア
    profit_score = min(100.0, profit_margin * 300.0)

    total = period_score * 0.40 + repayment_score * 0.40 + profit_score * 0.20

    if total >= 70:
        judgment = "✅ 承認"
    elif total >= 60:
        judgment = "⚠️  条件付き承認"
    else:
        judgment = "❌ 否決"

    return {
        "monthly_payment": monthly_payment,
        "annual_payment": annual_payment,
        "repayment_ratio": repayment_ratio,
        "period_score": period_score,
        "repayment_score": repayment_score,
        "profit_score": profit_score,
        "total_score": total,
        "judgment": judgment,
    }


# ─── 表示ヘルパー ──────────────────────────────────────────────────────────

def _fmt_yen(v: float) -> str:
    return f"{v:,.0f} 円"


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _score_bar(score: float, width: int = 30) -> str:
    filled = int(score / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.1f}pt"


def print_comparison(before: dict, after: dict, case: dict) -> None:
    W = 70
    print("=" * W)
    print(f"  デモ Before/After 比較: {case['asset']}")
    print(f"  申込企業: {case['company']}  物件価格: {_fmt_yen(case['asset_price'])}")
    print("=" * W)

    print(f"\n{'項目':<26}{'改善前（誤）':>20}{'改善後（正）':>20}")
    print("-" * W)
    print(f"{'リース期間':<26}{str(case['term_before_years']) + '年（誤設定）':>20}{str(case['term_after_years']) + '年（正しい耐用年数）':>20}")
    print(
        f"{'月額リース料':<26}"
        f"{_fmt_yen(before['monthly_payment']):>20}"
        f"{_fmt_yen(after['monthly_payment']):>20}"
    )
    delta_monthly = after['monthly_payment'] - before['monthly_payment']
    print(f"  └ 差額: {_fmt_yen(abs(delta_monthly))} {'削減' if delta_monthly < 0 else '増加'}")
    print(
        f"{'年間支払総額':<26}"
        f"{_fmt_yen(before['annual_payment']):>20}"
        f"{_fmt_yen(after['annual_payment']):>20}"
    )
    print(
        f"{'返済負担率（対売上）':<22}"
        f"{_fmt_pct(before['repayment_ratio']):>20}"
        f"{_fmt_pct(after['repayment_ratio']):>20}"
    )

    print()
    print(f"{'スコア内訳':<26}{'改善前':>20}{'改善後':>20}")
    print("-" * W)
    for label, key in [("  期間適正スコア(40%)", "period_score"),
                        ("  返済能力スコア(40%)", "repayment_score"),
                        ("  収益性スコア(20%)",   "profit_score")]:
        print(f"{label:<26}{before[key]:>19.1f}{after[key]:>20.1f}")

    print()
    print(f"  {'審査スコア（合計）':}")
    print(f"    改善前: {_score_bar(before['total_score'])}  →  {before['judgment']}")
    print(f"    改善後: {_score_bar(after['total_score'])}  →  {after['judgment']}")

    print()
    print("  ポイント:")
    print(f"    ・月額リース料が {_fmt_yen(abs(delta_monthly))} 削減（借主の負担軽減）")
    if before['judgment'] != after['judgment']:
        print(f"    ・審査判定が変わった: {before['judgment']} → {after['judgment']}")
    print(f"    ・誤った期間設定がスコアに {(after['total_score'] - before['total_score']):+.1f}pt 影響していた")
    print("=" * W)


# ─── メイン ────────────────────────────────────────────────────────────────

def main() -> None:
    # ブルドーザー大型機（建設業 中小規模）— 返済負担がやや重いケース
    ASSET_PRICE = 50_000_000      # 5,000万円
    ANNUAL_SALES = 200_000_000    # 年商2億円（建設業 中小）
    ANNUAL_PROFIT = 6_000_000     # 営業利益 600万円（利益率3%）
    ANNUAL_RATE = 0.025           # 金利2.5%
    CORRECT_YEARS = 6             # 国税庁: ブルドーザー 6年

    case = {
        "asset": "ブルドーザー（コマツ D85EX 大型）",
        "company": "田中建設工業（仮名）",
        "asset_price": ASSET_PRICE,
        "term_before_years": 5,   # 誤設定
        "term_after_years": CORRECT_YEARS,
    }

    before = calc_demo_score(ASSET_PRICE, 5, ANNUAL_SALES, ANNUAL_PROFIT, CORRECT_YEARS)
    after  = calc_demo_score(ASSET_PRICE, CORRECT_YEARS, ANNUAL_SALES, ANNUAL_PROFIT, CORRECT_YEARS)

    print_comparison(before, after, case)

    print()
    print("  【自動修正内容】")
    print(f"    useful_life_equipment.json > ブルドーザー > years: 5 → 6")
    print(f"    根拠: 国税庁耐用年数表 別表第二55号（自走式建設機械）")
    print()
    print("  ※このデモは REV-139 デモパイプラインで自動検出・修正された改善例です")


if __name__ == "__main__":
    main()
