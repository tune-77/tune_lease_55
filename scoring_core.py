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
from app_logger import log_warning

APPROVAL_LINE = 71  # 承認ライン（71点以上で承認圏内）

# 担当者直感スコア（1-5）の最大補正幅（点）
INTUITION_MAX_ADJ = 3.0

# SHAP: 各係数キーの日本語ラベル
_FEATURE_LABELS_JA = {
    "intercept":          "基本スコア（切片）",
    "ind_medical":        "業種（医療・福祉）",
    "ind_transport":      "業種（運輸・運送）",
    "ind_construction":   "業種（建設）",
    "ind_manufacturing":  "業種（製造）",
    "ind_service":        "業種（卸売・サービス）",
    "sales_log":          "売上規模",
    "bank_credit_log":    "銀行信用枠",
    "lease_credit_log":   "リース信用枠",
    "op_profit":          "営業利益",
    "ord_profit":         "経常利益",
    "net_income":         "純利益",
    "gross_profit":       "粗利益",
    "machines":           "機械装置",
    "other_assets":       "その他資産",
    "rent":               "家賃",
    "depreciation":       "減価償却費",
    "dep_expense":        "減価償却費（費用）",
    "rent_expense":       "賃借料（費用）",
    "grade_4_6":          "格付（4-6ランク）",
    "grade_watch":        "格付（要注意）",
    "grade_none":         "格付（無格付）",
    "contracts":          "過去取引件数",
}


def _safe_float(val, default: float = 0.0) -> float:
    """安全なfloat変換。Noneや変換不能な値はdefaultを返す。"""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    """安全なint変換。Noneや変換不能な値はdefaultを返す。"""
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _load_benchmarks():
    """industry_benchmarks.json を読み込む。"""
    path = os.path.join(_SCRIPT_DIR, "industry_benchmarks.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_warning(f"業界目安JSON読み込み失敗: {e}", context="_load_benchmarks")
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


def compute_score_contributions(data: dict, coeff_set: dict) -> list[dict]:
    """
    線形モデルの各特徴量の寄与度（SHAP近似）を計算して返す。
    ロジスティック回帰では β_i * x_i がそのままロジット空間での寄与に対応する。

    Returns:
        [{"feature": str, "label_ja": str, "contribution": float}, ...] を
        |contribution| 降順でソートしたリスト。
    """
    import math

    contributions = []

    # 切片
    intercept_val = coeff_set.get("intercept", 0)
    contributions.append({
        "feature": "intercept",
        "label_ja": _FEATURE_LABELS_JA["intercept"],
        "contribution": intercept_val,
    })

    major = data.get("industry_major") or ""
    # 業種フラグ（いずれか1つが有効）
    ind_flags = {
        "ind_medical":       ("医療" in major or "福祉" in major or major.startswith("P")),
        "ind_transport":     ("運輸" in major or major.startswith("H")),
        "ind_construction":  ("建設" in major or major.startswith("D")),
        "ind_manufacturing": ("製造" in major or major.startswith("E")),
        "ind_service":       (any(x in major for x in ["卸売", "小売", "サービス"]) or
                              (bool(major) and major[0] in ["I", "K", "M", "R"])),
    }
    for feat, active in ind_flags.items():
        coeff_val = coeff_set.get(feat, 0)
        if active and coeff_val != 0:
            contributions.append({
                "feature": feat,
                "label_ja": _FEATURE_LABELS_JA.get(feat, feat),
                "contribution": coeff_val,
            })

    # 連続値特徴量
    def _log_contrib(key_data, key_coeff, log_func=np.log1p):
        val = data.get(key_data) or 0
        coeff_val = coeff_set.get(key_coeff, 0)
        if val > 0 and coeff_val != 0:
            return coeff_val * log_func(val)
        return 0.0

    for data_key, coeff_key in [("nenshu", "sales_log"),
                                  ("bank_credit", "bank_credit_log"),
                                  ("lease_credit", "lease_credit_log")]:
        c = _log_contrib(data_key, coeff_key)
        if c != 0:
            contributions.append({
                "feature": coeff_key,
                "label_ja": _FEATURE_LABELS_JA.get(coeff_key, coeff_key),
                "contribution": c,
            })

    for linear_key in ["op_profit", "ord_profit", "net_income", "machines",
                        "other_assets", "rent", "gross_profit", "depreciation",
                        "dep_expense", "rent_expense"]:
        val = data.get(linear_key) or 0
        coeff_val = coeff_set.get(linear_key, 0)
        if val != 0 and coeff_val != 0:
            contributions.append({
                "feature": linear_key,
                "label_ja": _FEATURE_LABELS_JA.get(linear_key, linear_key),
                "contribution": coeff_val * val,
            })

    # 格付フラグ
    grade = data.get("grade") or "1-3"
    grade_flags = {
        "grade_4_6":    "4-6" in grade,
        "grade_watch":  "要注意" in grade,
        "grade_none":   "無格付" in grade,
    }
    for feat, active in grade_flags.items():
        coeff_val = coeff_set.get(feat, 0)
        if active and coeff_val != 0:
            contributions.append({
                "feature": feat,
                "label_ja": _FEATURE_LABELS_JA.get(feat, feat),
                "contribution": coeff_val,
            })

    # 取引件数
    contracts_val = data.get("contracts") or 0
    contracts_coeff = coeff_set.get("contracts", 0)
    if contracts_val != 0 and contracts_coeff != 0:
        contributions.append({
            "feature": "contracts",
            "label_ja": _FEATURE_LABELS_JA["contracts"],
            "contribution": contracts_coeff * contracts_val,
        })

    # |寄与度| 降順でソート
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions


def run_quick_scoring(inputs: dict) -> dict:
    """
    入力辞書からスコア・判定・比較文を計算して返す。
    inputs のキー例: nenshu, op_profit(営業利益千円), ord_profit, net_income, net_assets, total_assets,
                    industry_major, industry_sub, grade, customer_type, main_bank, competitor,
                    bank_credit, lease_credit, contracts, asset_score(省略時50), など。
    返却: score, hantei, comparison, user_op_margin, user_equity_ratio, bench_op_margin, bench_equity_ratio,
          score_borrower, industry_sub, industry_major
    """
    nenshu = max(0, _safe_float(inputs.get("nenshu")))
    op_profit = _safe_float(inputs.get("op_profit") or inputs.get("rieki"))
    ord_profit = _safe_float(inputs.get("ord_profit"))
    net_income = _safe_float(inputs.get("net_income"))
    net_assets = _safe_float(inputs.get("net_assets"))
    total_assets = _safe_float(inputs.get("total_assets"))
    industry_major = (inputs.get("industry_major") or "D 建設業").strip()
    industry_sub = (inputs.get("industry_sub") or "06 総合工事業").strip()
    grade = inputs.get("grade") or "1-3"
    customer_type = inputs.get("customer_type") or "既存先"
    bank_credit = _safe_float(inputs.get("bank_credit"))
    lease_credit = _safe_float(inputs.get("lease_credit"))
    contracts = _safe_int(inputs.get("contracts"))
    _raw_asset_score = inputs.get("asset_score")
    used_default_asset_score = _raw_asset_score is None or str(_raw_asset_score).strip() == ""
    asset_score = _safe_float(_raw_asset_score, default=50.0)

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

    # ── 単位変換メモ ───────────────────────────────────────────────────────────
    # inputs の財務数値はすべて「千円」単位で入力される（UI・Slack Bot 共通）。
    # COEFFS の係数は以下のスケールで推定されている:
    #   nenshu / bank_credit / lease_credit : 千円のまま log1p に渡す（係数で吸収）
    #   op_profit / ord_profit 等の P/L 項目 : 千円 → 百万円（÷1000）に変換して渡す
    # 変換基準を変更した場合は coeff_definitions.py の係数を必ず再推定すること。
    data_scoring = {
        "nenshu": nenshu,           # 千円単位のまま（log1p 変換して使用）
        "bank_credit": bank_credit,  # 千円単位のまま（log1p 変換して使用）
        "lease_credit": lease_credit,  # 千円単位のまま（log1p 変換して使用）
        "op_profit": op_profit / 1000,     # 千円 → 百万円
        "ord_profit": ord_profit / 1000,   # 千円 → 百万円
        "net_income": net_income / 1000,   # 千円 → 百万円
        "gross_profit": _safe_float(inputs.get("gross_profit")) / 1000,   # 千円 → 百万円
        "machines": _safe_float(inputs.get("machines")) / 1000,           # 千円 → 百万円
        "other_assets": _safe_float(inputs.get("other_assets")) / 1000,   # 千円 → 百万円
        "rent": _safe_float(inputs.get("rent")) / 1000,                   # 千円 → 百万円
        "depreciation": _safe_float(inputs.get("depreciation")) / 1000,   # 千円 → 百万円
        "dep_expense": _safe_float(inputs.get("dep_expense")) / 1000,     # 千円 → 百万円
        "rent_expense": _safe_float(inputs.get("rent_expense")) / 1000,   # 千円 → 百万円
        "contracts": contracts,          # 件数（無次元）
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
    base_score = w_borrower * score_borrower + w_asset * asset_score
    base_score = max(0, min(100, round(base_score, 1)))

    # ── 担当者直感スコア補正（1-5スケール、中立=3、±INTUITION_MAX_ADJ 点まで）──
    intuition_score = _safe_float(inputs.get("intuition_score"), default=0)
    intuition_adj = 0.0
    if 1.0 <= intuition_score <= 5.0:
        # 3を中立として ±INTUITION_MAX_ADJ 点に線形マッピング
        intuition_adj = round((intuition_score - 3.0) / 2.0 * INTUITION_MAX_ADJ, 2)

    final_score = max(0, min(100, round(base_score + intuition_adj, 1)))
    hantei = "承認圏内" if final_score >= APPROVAL_LINE else "要審議"

    # 直感スコアが高いのに要審議 → 上長確認フラグ
    manager_review_flag = (intuition_score >= 4.0) and (final_score < APPROVAL_LINE)

    # ── SHAP近似: 各特徴量の寄与度を計算 ──
    score_contributions = compute_score_contributions(data_scoring, coeffs)

    # 物件スコアのデフォルト使用フラグ
    asset_score_warnings = []
    if used_default_asset_score:
        asset_score_warnings.append("物件スコア未入力のためデフォルト値(50)を使用")

    return {
        "score": final_score,
        "score_base": base_score,
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
        "used_default_asset_score": used_default_asset_score,
        "asset_score_warnings": asset_score_warnings,
        # 直感スコア関連
        "intuition_score": intuition_score,
        "intuition_adj": intuition_adj,
        "manager_review_flag": manager_review_flag,
        # SHAP近似: 各特徴量の寄与度（上位5件はUI表示に活用）
        "score_contributions": score_contributions,
    }
