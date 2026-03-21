"""
審査スコア計算コア（Streamlit 非依存）
Flask 等の Web アプリから呼び出す用。判定ロジックの簡易版（定量メイン＋業界比較）。
"""
import os
import math
import json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

# パスを追加してからインポート
import sys
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data_cases import get_effective_coeffs, get_score_weights
from coeff_definitions import COEFFS

APPROVAL_LINE = 71  # 承認ライン（71点以上で承認圏内）


def _load_benchmarks():
    """industry_benchmarks.json を読み込む。"""
    path = os.path.join(_REPO_ROOT, "industry_benchmarks.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _safe_sigmoid(x):
    try:
        if x > 700:
            return 1.0
        if x < -700:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))
    except (OverflowError, TypeError):
        return 0.0 if (x or 0) < 0 else 1.0


def _calculate_z(data, coeff_set):
    """係数セットとデータから z (ロジット) を計算。"""
    z = coeff_set.get("intercept", 0)
    major = data.get("industry_major") or ""

    if "医療" in major or "福祉" in major or major.startswith("P"):
        z += coeff_set.get("ind_medical", 0)
    elif "運輸" in major or major.startswith("H"):
        z += coeff_set.get("ind_transport", 0)
    elif "建設" in major or major.startswith("D"):
        z += coeff_set.get("ind_construction", 0)
    elif "製造" in major or major.startswith("E"):
        z += coeff_set.get("ind_manufacturing", 0)
    elif any(x in major for x in ["卸売", "小売", "サービス"]) or (major and major[0] in ["I", "K", "M", "R"]):
        z += coeff_set.get("ind_service", 0)

    nenshu = data.get("nenshu") or 0
    if nenshu > 0:
        z += np.log1p(nenshu) * coeff_set.get("sales_log", 0)
    bc = data.get("bank_credit") or 0
    if bc > 0:
        z += np.log1p(bc) * coeff_set.get("bank_credit_log", 0)
    lc = data.get("lease_credit") or 0
    if lc > 0:
        z += np.log1p(lc) * coeff_set.get("lease_credit_log", 0)

    z += (data.get("op_profit") or 0) * coeff_set.get("op_profit", 0)
    z += (data.get("ord_profit") or 0) * coeff_set.get("ord_profit", 0)
    z += (data.get("net_income") or 0) * coeff_set.get("net_income", 0)
    z += (data.get("machines") or 0) * coeff_set.get("machines", 0)
    z += (data.get("other_assets") or 0) * coeff_set.get("other_assets", 0)
    z += (data.get("rent") or 0) * coeff_set.get("rent", 0)
    z += (data.get("gross_profit") or 0) * coeff_set.get("gross_profit", 0)
    z += (data.get("depreciation") or 0) * coeff_set.get("depreciation", 0)
    z += (data.get("dep_expense") or 0) * coeff_set.get("dep_expense", 0)
    z += (data.get("rent_expense") or 0) * coeff_set.get("rent_expense", 0)

    grade = data.get("grade") or "1-3"
    if "4-6" in grade:
        z += coeff_set.get("grade_4_6", 0)
    elif "要注意" in grade:
        z += coeff_set.get("grade_watch", 0)
    elif "無格付" in grade:
        z += coeff_set.get("grade_none", 0)

    z += (data.get("contracts") or 0) * coeff_set.get("contracts", 0)
    return z


def run_quick_scoring(inputs: dict) -> dict:
    """
    入力辞書からスコア・判定・比較文を計算して返す。
    inputs のキー例: nenshu, op_profit(営業利益千円), ord_profit, net_income, net_assets, total_assets,
                    industry_major, industry_sub, grade, customer_type, main_bank, competitor,
                    bank_credit, lease_credit, contracts, asset_score(省略時50), など。
    返却: score, hantei, comparison, user_op_margin, user_equity_ratio, bench_op_margin, bench_equity_ratio,
          score_borrower, industry_sub, industry_major
    """
    nenshu = max(0, float(inputs.get("nenshu") or 0))
    op_profit = float(inputs.get("op_profit") or inputs.get("rieki") or 0)
    ord_profit = float(inputs.get("ord_profit") or 0)
    net_income = float(inputs.get("net_income") or 0)
    net_assets = float(inputs.get("net_assets") or 0)
    total_assets = float(inputs.get("total_assets") or 0)
    industry_major = (inputs.get("industry_major") or "D 建設業").strip()
    industry_sub = (inputs.get("industry_sub") or "06 総合工事業").strip()
    grade = inputs.get("grade") or "1-3"
    customer_type = inputs.get("customer_type") or "既存先"
    bank_credit = float(inputs.get("bank_credit") or 0)
    lease_credit = float(inputs.get("lease_credit") or 0)
    contracts = int(inputs.get("contracts") or 0)
    asset_score = float(inputs.get("asset_score") or 50)

    user_op_margin = (op_profit / nenshu * 100) if nenshu > 0 else 0.0
    user_equity_ratio = (net_assets / total_assets * 100) if total_assets > 0 else 0.0

    benchmarks = _load_benchmarks()
    bench = benchmarks.get(industry_sub, {})
    bench_op_margin = float(bench.get("op_margin") or 0)
    beq = bench.get("equity_ratio")
    if beq is not None and isinstance(beq, (int, float)):
        bench_equity_ratio = float(beq)
    else:
        bench_equity_ratio = float(bench.get("equity_ratio_display") or 0)
    bench_comment = (bench.get("comment") or "").strip()

    comp_margin = "高い" if user_op_margin >= bench_op_margin else "低い"
    comp_equity = "高い" if user_equity_ratio >= bench_equity_ratio else "低い"
    comparison = (
        f"- **営業利益率**: {user_op_margin:.1f}% (業界目安: {bench_op_margin:.1f}%) → 平均より{comp_margin}\n"
        f"- **自己資本比率**: {user_equity_ratio:.1f}% (業界目安: {bench_equity_ratio:.1f}%) → 平均より{comp_equity}\n"
        f"- **業界**: {bench_comment or '—'}"
    )

    # 百万円単位
    data_scoring = {
        "nenshu": nenshu,
        "bank_credit": bank_credit,
        "lease_credit": lease_credit,
        "op_profit": op_profit / 1000,
        "ord_profit": ord_profit / 1000,
        "net_income": net_income / 1000,
        "gross_profit": float(inputs.get("gross_profit") or 0) / 1000,
        "machines": float(inputs.get("machines") or 0) / 1000,
        "other_assets": float(inputs.get("other_assets") or 0) / 1000,
        "rent": float(inputs.get("rent") or 0) / 1000,
        "depreciation": float(inputs.get("depreciation") or 0) / 1000,
        "dep_expense": float(inputs.get("dep_expense") or 0) / 1000,
        "rent_expense": float(inputs.get("rent_expense") or 0) / 1000,
        "contracts": contracts,
        "grade": grade,
        "industry_major": industry_major,
    }

    coeff_key = "全体_既存先"
    if customer_type == "新規先":
        coeff_key = "全体_新規先"
    coeffs = get_effective_coeffs(coeff_key)
    z_main = _calculate_z(data_scoring, coeffs)
    score_prob = _safe_sigmoid(z_main)
    score_borrower = score_prob * 100

    w_borrower, w_asset, _, _ = get_score_weights()
    final_score = w_borrower * score_borrower + w_asset * asset_score
    final_score = max(0, min(100, round(final_score, 1)))
    hantei = "承認圏内" if final_score >= APPROVAL_LINE else "要審議"

    return {
        "score": final_score,
        "hantei": hantei,
        "comparison": comparison,
        "user_op_margin": user_op_margin,
        "user_equity_ratio": user_equity_ratio,
        "bench_op_margin": bench_op_margin,
        "bench_equity_ratio": bench_equity_ratio,
        "score_borrower": round(score_borrower, 1),
        "industry_sub": industry_sub,
        "industry_major": industry_major,
        "approval_line": APPROVAL_LINE,
    }
