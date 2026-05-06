"""
回帰・要因分析モジュール（lease_logic_sumaho10）
設計行列構築、ロジスティック回帰、定性/定量要因分析、成約ドライバー分析、重み最適化を提供。
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data_cases import load_all_cases, get_effective_coeffs
from coeff_definitions import COEFFS, BAYESIAN_PRIOR_EXTRA, STRENGTH_TAG_WEIGHTS, DEFAULT_STRENGTH_WEIGHT
from charts import _equity_ratio_display
from config import LGBM_PARAMS
COEFF_MAIN_KEYS = [
    "ind_medical", "ind_transport", "ind_construction", "ind_manufacturing", "ind_service",
    "sales_log", "bank_credit_log", "lease_credit_log",
    "op_profit", "ord_profit", "net_income", "machines", "other_assets", "rent",
    "gross_profit", "depreciation", "dep_expense", "rent_expense",
    "grade_4_6", "grade_watch", "grade_none", "contracts",
]
# 追加項目（ベイズ補完）: 回帰・スコア両方で使用
COEFF_EXTRA_KEYS = [
    "main_bank", "competitor_present", "competitor_none",
    "customer_new", "deal_source_bank",
    "rate_diff_z", "industry_sentiment_z", "qualitative_tag_score", "qualitative_passion",
    "equity_ratio",          # 自己資本比率（%）
    "dscr_approx",           # 返済余力の近似
    "interest_coverage",     # 支払利息カバー近似
    "qualitative_combined",  # 定性スコアリング合計（総合×60%＋定性×40%）を0-1で正規化
    # BNエンジン出力（スコア≤70の案件でBN推論を実行した場合のみ値あり、未実行時は0）
    "bn_approval_prob",      # BN最終承認確率（0-1）
    "bn_fc",                 # 財務信用度（Financial_Creditworthiness, 0-1）
    "bn_hc",                 # ヘッジ条件（Hedge_Condition, 0-1）
    "bn_av",                 # 物件価値（Asset_Value, 0-1）
    # ── 定性スコアリング項目（IVで最重要と確認済み・sumaho13追加） ──
    "qual_weighted",         # 定性スコアリング weighted_score を 0-1 正規化
    "qual_rank_good",        # 定性ランク A/B = 1, それ以外 = 0（優良フラグ）
    "qual_repayment",        # 返済履歴スコア（items.repayment_history.value / 4.0）
    "quantum_risk",          # 量子インスパイア矛盾スコア
    # 新規先専用の商流・競争圧力特徴
    "new_customer_main_bank",
    "new_customer_competitor_present",
    "new_customer_competitor_count",
    "new_customer_competitor_rate",
    "new_customer_deal_source_bank",
    "new_customer_deal_occurrence_nomination",
    "new_customer_deal_occurrence_comp",
    "new_customer_contract_auto",
]

# 業種ごと・既存先/新規先のモデルキー（ベイズ回帰で更新対象）
INDUSTRY_MODEL_KEYS = [
    "全体_既存先", "全体_新規先",
    "医療_既存先", "医療_新規先",
    "運送業_既存先", "運送業_新規先",
    "サービス業_既存先", "サービス業_新規先",
    "製造業_既存先", "製造業_新規先",
]
# 指標モデルも既存先/新規先で分けて回帰
INDICATOR_MODEL_KEYS = [
    "全体_指標_既存先", "全体_指標_新規先",
    "医療_指標_既存先", "医療_指標_新規先",
    "運送業_指標_既存先", "運送業_指標_新規先",
    "サービス業_指標_既存先", "サービス業_指標_新規先",
    "製造業_指標_既存先", "製造業_指標_新規先",
]
# 事前係数入力画面で編集可能なモデル一覧（業種＋指標のベース）
PRIOR_COEFF_MODEL_KEYS = [
    "全体_既存先", "全体_新規先", "医療_既存先", "医療_新規先",
    "運送業_既存先", "運送業_新規先", "サービス業_既存先", "サービス業_新規先",
    "製造業_既存先", "製造業_新規先",
    "全体_指標", "医療_指標", "運送業_指標", "サービス業_指標", "製造業_指標",
]
# 指標モデル用の説明変数（ratio + grade + ind ダミー）。全体_指標の係数キー順に合わせる
INDICATOR_MAIN_KEYS = [
    "ind_service", "ind_medical", "ind_transport", "ind_construction", "ind_manufacturing",
    "ratio_op_margin", "ratio_gross_margin", "ratio_ord_margin", "ratio_net_margin",
    "ratio_fixed_assets", "ratio_rent", "ratio_depreciation", "ratio_machines",
    "grade_4_6", "grade_watch", "grade_none",
]


def _normalize_competitor_count(value) -> float:
    """競合社数の入力を 0〜3 の連続値に正規化する。"""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        try:
            v = float(value)
            if v <= 0:
                return 0.0
            return min(3.0, v)
        except (TypeError, ValueError):
            return 0.0
    s = str(value)
    if "3" in s:
        return 3.0
    if "2" in s:
        return 2.0
    if "1" in s:
        return 1.0
    if "0" in s or "指名" in s:
        return 0.0
    return 0.0


def _normalize_deal_occurrence(value) -> float:
    """発生経緯を 0=不明, 1=指名, 2=相見積もり に正規化する。"""
    if value is None:
        return 0.0
    s = str(value)
    if "相見積" in s or "競争" in s:
        return 2.0
    if "指名" in s:
        return 1.0
    return 0.0


def _normalize_competitor_rate_value(value) -> float:
    """競合提示金利を 0〜1 の圧力指標に正規化する。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v <= 0:
        return 0.0
    if v <= 1.0:
        v *= 100.0
    elif v > 1000:
        v /= 1000.0
    return max(0.0, min(1.0, v / 30.0))


def _build_new_customer_context_features(source: dict, customer_type: str | None = None) -> dict[str, float]:
    """新規先に効く商流・競争圧力の交互作用特徴をまとめて返す。"""
    ct = customer_type or source.get("customer_type") or "既存先"
    is_new = 1.0 if ct == "新規先" else 0.0
    main_bank = source.get("main_bank") or "非メイン先"
    competitor = source.get("competitor") or "競合なし"
    deal_source = source.get("deal_source") or "その他"
    contract_type = source.get("contract_type") or "一般"
    num_competitors = source.get("num_competitors")
    deal_occurrence = source.get("deal_occurrence")
    competitor_rate = source.get("competitor_rate")

    return {
        "new_customer_main_bank": is_new if main_bank == "メイン先" else 0.0,
        "new_customer_competitor_present": is_new if competitor == "競合あり" else 0.0,
        "new_customer_competitor_count": is_new * _normalize_competitor_count(num_competitors),
        "new_customer_competitor_rate": is_new * _normalize_competitor_rate_value(competitor_rate),
        "new_customer_deal_source_bank": is_new if deal_source == "銀行紹介" else 0.0,
        "new_customer_deal_occurrence_nomination": is_new if _normalize_deal_occurrence(deal_occurrence) == 1.0 else 0.0,
        "new_customer_deal_occurrence_comp": is_new if _normalize_deal_occurrence(deal_occurrence) == 2.0 else 0.0,
        "new_customer_contract_auto": is_new if contract_type == "自動車" else 0.0,
    }


def _get_ind_key_from_log(log):
    """ログから業種モデルキー（既存先/新規先）を算出。"""
    res = log.get("result") or {}
    major = res.get("industry_major") or log.get("industry_major") or "D 建設業"
    major_code = major.split(" ")[0] if isinstance(major, str) and " " in major else (major[0] if major else "D")
    customer_type = log.get("customer_type") or "既存先"
    if major_code == "H":
        base = "運送業"
    elif major_code == "P":
        base = "医療"
    elif major_code in ["I", "K", "M", "R"]:
        base = "サービス業"
    elif major_code == "E":
        base = "製造業"
    else:
        base = "全体"
    suffix = "新規先" if customer_type == "新規先" else "既存先"
    return f"{base}_{suffix}"


def _get_bench_key_from_log(log):
    """ログから指標モデルのベースキー（業種）を算出。"""
    res = log.get("result") or {}
    major = res.get("industry_major") or log.get("industry_major") or "D 建設業"
    major_code = major.split(" ")[0] if isinstance(major, str) and " " in major else (major[0] if major else "D")
    if major_code == "D":
        return "全体_指標"
    if major_code == "P":
        return "医療_指標"
    if major_code == "H":
        return "運送業_指標"
    if major_code in ["I", "K", "M", "R"]:
        return "サービス業_指標"
    if major_code == "E":
        return "製造業_指標"
    return "全体_指標"


def _get_indicator_model_key_from_log(log):
    """ログから指標モデルキー（既存先/新規先付き）を算出。"""
    base = _get_bench_key_from_log(log)
    customer_type = log.get("customer_type") or "既存先"
    suffix = "新規先" if customer_type == "新規先" else "既存先"
    return f"{base}_{suffix}"


def _log_to_data_scoring(log):
    """1件のログからスコア計算用 data_scoring 相当の辞書を組み立てる（単位: 千円→百万円）。"""
    inp = log.get("inputs") or {}
    res = log.get("result") or {}
    nenshu = float(inp.get("nenshu") or 0)
    bank_credit = float(inp.get("bank_credit") or 0)
    lease_credit = float(inp.get("lease_credit") or 0)
    # 百万円換算
    to_mill = 1.0 / 1000.0
    op_profit = float(inp.get("op_profit") or 0) * to_mill
    ord_profit = float(inp.get("ord_profit") or 0) * to_mill
    net_income = float(inp.get("net_income") or 0) * to_mill
    gross_profit = float(inp.get("gross_profit") or 0) * to_mill
    machines = float(inp.get("machines") or 0) * to_mill
    other_assets = float(inp.get("other_assets") or 0) * to_mill
    rent = float(inp.get("rent") or 0) * to_mill
    depreciation = float(inp.get("depreciation") or 0) * to_mill
    dep_expense = float(inp.get("dep_expense") or 0) * to_mill
    rent_expense = float(inp.get("rent_expense") or 0) * to_mill
    contracts = float(inp.get("contracts") or 0)
    grade = (inp.get("grade") or res.get("grade") or "")
    industry_major = res.get("industry_major") or (log.get("industry_major") or "D 建設業")
    customer_type = log.get("customer_type") or inp.get("customer_type") or "既存先"
    new_ctx = _build_new_customer_context_features({
        "main_bank": log.get("main_bank") or inp.get("main_bank") or "非メイン先",
        "competitor": log.get("competitor") or inp.get("competitor") or "競合なし",
        "deal_source": log.get("deal_source") or inp.get("deal_source") or "その他",
        "contract_type": inp.get("contract_type") or log.get("contract_type") or "一般",
        "num_competitors": inp.get("num_competitors") or log.get("num_competitors") or "未入力",
        "deal_occurrence": inp.get("deal_occurrence") or log.get("deal_occurrence") or "不明",
        "competitor_rate": log.get("competitor_rate") or inp.get("competitor_rate"),
        "customer_type": customer_type,
    }, customer_type=customer_type)
    return {
        "nenshu": nenshu, "bank_credit": bank_credit, "lease_credit": lease_credit,
        "op_profit": op_profit, "ord_profit": ord_profit, "net_income": net_income,
        "gross_profit": gross_profit, "machines": machines, "other_assets": other_assets,
        "rent": rent, "depreciation": depreciation, "dep_expense": dep_expense, "rent_expense": rent_expense,
        "contracts": contracts, "grade": grade, "industry_major": industry_major,
        **new_ctx,
    }


def _compute_dscr_approx_from_log(log) -> float:
    inp = log.get("inputs") or {}
    op_profit = float(inp.get("op_profit") or inp.get("rieki") or 0)
    dep_expense = float(inp.get("dep_expense") or inp.get("depreciation") or 0)
    rent_expense = float(inp.get("rent_expense") or inp.get("rent") or 0)
    denominator = dep_expense + rent_expense
    if denominator <= 0:
        return 1.0
    return round(op_profit / denominator, 3)


def _compute_interest_coverage_from_log(log) -> float:
    inp = log.get("inputs") or {}
    op_profit = float(inp.get("op_profit") or inp.get("rieki") or 0)
    interest = float(inp.get("interest_expense") or 0)
    if interest <= 0:
        return 10.0
    return round(op_profit / interest, 3)


def _build_one_row_industry(log, data):
    """1ログから業種モデル用の1行（既存22+追加9）を構築。"""
    major = data["industry_major"]
    ind_medical = 1.0 if ("医療" in major or "福祉" in major or (isinstance(major, str) and major.startswith("P"))) else 0.0
    ind_transport = 1.0 if ("運輸" in major or (isinstance(major, str) and major.startswith("H"))) else 0.0
    ind_construction = 1.0 if ("建設" in major or (isinstance(major, str) and major.startswith("D"))) else 0.0
    ind_manufacturing = 1.0 if ("製造" in major or (isinstance(major, str) and major.startswith("E"))) else 0.0
    ind_service = 1.0 if ("卸売" in major or "小売" in major or "サービス" in major or (isinstance(major, str) and major[0] in ["I", "K", "M", "R"])) else 0.0
    sales_log = np.log1p(data["nenshu"])
    bank_credit_log = np.log1p(data["bank_credit"])
    lease_credit_log = np.log1p(data["lease_credit"])
    grade = data["grade"]
    grade_4_6 = 1.0 if "4-6" in grade else 0.0
    grade_watch = 1.0 if "要注意" in grade else 0.0
    grade_none = 1.0 if "無格付" in grade else 0.0
    row = [
        ind_medical, ind_transport, ind_construction, ind_manufacturing, ind_service,
        sales_log, bank_credit_log, lease_credit_log,
        data["op_profit"], data["ord_profit"], data["net_income"], data["machines"], data["other_assets"], data["rent"],
        data["gross_profit"], data["depreciation"], data["dep_expense"], data["rent_expense"],
        grade_4_6, grade_watch, grade_none, data["contracts"],
    ]
    inp, res = log.get("inputs") or {}, log.get("result") or {}
    main_bank = 1.0 if log.get("main_bank") == "メイン先" else 0.0
    competitor_present = 1.0 if log.get("competitor") == "競合あり" else 0.0
    competitor_none = 1.0 if log.get("competitor") == "競合なし" else 0.0
    customer_new = 1.0 if (log.get("customer_type") or inp.get("customer_type") or "既存先") == "新規先" else 0.0
    deal_source_bank = 1.0 if (log.get("deal_source") or inp.get("deal_source") or "") == "銀行紹介" else 0.0
    y_pred, comp_rate = res.get("yield_pred"), log.get("competitor_rate")
    if y_pred is not None and comp_rate is not None and isinstance(comp_rate, (int, float)):
        rate_diff_pt = float(y_pred) - float(comp_rate)
        rate_diff_z = max(-2.0, min(2.0, rate_diff_pt / 5.0))
    else:
        rate_diff_z = 0.0
    industry_sentiment_z = float(res.get("industry_sentiment_z", 0))
    qual = inp.get("qualitative") or {}
    tags = qual.get("strength_tags") or []
    qualitative_tag_score = min(sum(STRENGTH_TAG_WEIGHTS.get(t, DEFAULT_STRENGTH_WEIGHT) for t in tags), 10.0)
    qualitative_passion = 1.0 if qual.get("passion_text") else 0.0
    equity_ratio = float(res.get("user_eq") or 0)
    qsc = (res.get("qualitative_scoring_correction") or inp.get("qualitative_scoring")) or {}
    combined = qsc.get("combined_score") or qsc.get("weighted_score")
    qualitative_combined = (float(combined) / 100.0) if combined is not None else 0.0
    # BNエンジン出力（スコア≤70の案件のみ値あり、未実行時は 0 で埋める）
    bn      = log.get("bn_engine") or {}
    bn_im   = bn.get("intermediate") or {}
    bn_approval_prob = float(bn.get("approval_prob") or 0)
    bn_fc   = float(bn_im.get("Financial_Creditworthiness") or 0)
    bn_hc   = float(bn_im.get("Hedge_Condition") or 0)
    bn_av   = float(bn_im.get("Asset_Value") or 0)
    # 定性スコアリング項目（sumaho13追加）
    qsc_items  = qsc.get("items") or {}
    qual_weighted   = (float(qsc.get("weighted_score") or 0) / 100.0)
    qual_rank       = qsc.get("rank") or ""
    qual_rank_good  = 1.0 if qual_rank in ("A", "B") else 0.0
    repayment_val   = (qsc_items.get("repayment_history") or {}).get("value") or 0
    qual_repayment  = float(repayment_val) / 4.0  # 最大4段階 → 0-1 正規化
    dscr_approx     = _compute_dscr_approx_from_log(log)
    interest_coverage = _compute_interest_coverage_from_log(log)
    
    # 量子矛盾スコア (Q_risk) のオンザフライ計算
    quantum_risk = 0.0
    try:
        from evaluators.quantum_adapter import QuantumAdapter
        _qa = QuantumAdapter()
        if _qa.is_ready():
            quantum_risk = _qa.predict(log).risk
    except Exception:
        pass

    new_ctx = _build_new_customer_context_features({
        "main_bank": log.get("main_bank") or inp.get("main_bank") or "非メイン先",
        "competitor": log.get("competitor") or inp.get("competitor") or "競合なし",
        "deal_source": log.get("deal_source") or inp.get("deal_source") or "その他",
        "contract_type": inp.get("contract_type") or log.get("contract_type") or "一般",
        "num_competitors": inp.get("num_competitors") or log.get("num_competitors") or "未入力",
        "deal_occurrence": inp.get("deal_occurrence") or log.get("deal_occurrence") or "不明",
        "competitor_rate": log.get("competitor_rate") or inp.get("competitor_rate"),
        "customer_type": inp.get("customer_type") or log.get("customer_type") or "既存先",
    })

    row.extend([main_bank, competitor_present, competitor_none, customer_new, deal_source_bank, rate_diff_z, industry_sentiment_z,
                qualitative_tag_score, qualitative_passion, equity_ratio, dscr_approx, interest_coverage, qualitative_combined,
                bn_approval_prob, bn_fc, bn_hc, bn_av,
                qual_weighted, qual_rank_good, qual_repayment, quantum_risk,
                new_ctx["new_customer_main_bank"],
                new_ctx["new_customer_competitor_present"],
                new_ctx["new_customer_competitor_count"],
                new_ctx["new_customer_competitor_rate"],
                new_ctx["new_customer_deal_source_bank"],
                new_ctx["new_customer_deal_occurrence_nomination"],
                new_ctx["new_customer_deal_occurrence_comp"],
                new_ctx["new_customer_contract_auto"]])
    return row


def build_design_matrix_from_logs(all_logs, model_key=None):
    """
    成約/失注が登録されたログから、業種モデル用の説明変数行列 X と目的変数 y を構築する。
    model_key を指定した場合はその業種・既存先/新規先のログのみ使用。
    目的変数: 成約=1, 失注=0。
    """
    rows = []
    y_list = []
    for log in all_logs:
        if log.get("final_status") not in ["成約", "失注"]:
            continue
        if "inputs" not in log:
            continue
        if model_key is not None:
            # 「全体_*」は業種問わず既存先/新規先のみで絞る（全業種を集約）
            if model_key.startswith("全体_"):
                suffix = model_key[len("全体_"):]  # "既存先" or "新規先"
                ct = log.get("customer_type") or "既存先"
                if ct != suffix:
                    continue
            elif _get_ind_key_from_log(log) != model_key:
                continue
        data = _log_to_data_scoring(log)
        row = _build_one_row_industry(log, data)
        rows.append(row)
        y_list.append(1 if log.get("final_status") == "成約" else 0)
    if not rows:
        return None, None
    X = np.array(rows, dtype=float)
    y = np.array(y_list, dtype=int)
    return X, y


# 業種ごと・指標ごと分析用のベースキー
INDUSTRY_BASES = ["全体", "医療", "運送業", "サービス業", "製造業"]
BENCH_BASES = ["全体_指標", "医療_指標", "運送業_指標", "サービス業_指標", "製造業_指標"]


def build_design_matrix_from_logs_by_industry(all_logs, industry_base):
    """業種ベースでフィルタし、業種モデル用 X, y を構築。

    industry_base == "全体" のときは全件を対象にする。
    それ以外は _get_ind_key_from_log() の業種ベースで絞る。
    """
    rows = []
    y_list = []
    for log in all_logs:
        if log.get("final_status") not in ["成約", "失注"]:
            continue
        if "inputs" not in log:
            continue
        if industry_base != "全体":
            ind_key = _get_ind_key_from_log(log)
            if not ind_key.startswith(industry_base + "_"):
                continue
        data = _log_to_data_scoring(log)
        row = _build_one_row_industry(log, data)
        rows.append(row)
        y_list.append(1 if log.get("final_status") == "成約" else 0)
    if not rows:
        return None, None
    return np.array(rows, dtype=float), np.array(y_list, dtype=int)


def build_design_matrix_indicator_by_bench(all_logs, bench_key):
    """指標ベース（全体_指標/医療_指標/…）でフィルタし、指標モデル用 X, y を構築。"""
    rows = []
    y_list = []
    for log in all_logs:
        if log.get("final_status") not in ["成約", "失注"]:
            continue
        if "inputs" not in log:
            continue
        if _get_bench_key_from_log(log) != bench_key:
            continue
        data = _log_to_data_scoring(log)
        row = _build_one_row_indicator(log, data)
        rows.append(row)
        y_list.append(1 if log.get("final_status") == "成約" else 0)
    if not rows:
        return None, None
    return np.array(rows, dtype=float), np.array(y_list, dtype=int)


def _bootstrap_to_min_size(X, y, min_size=50, random_state=42):
    """件数が min_size 未満の場合、復元抽出で min_size 件にリサンプル。返り値 (X, y, was_resampled)。"""
    n = len(y)
    if n >= min_size:
        return X, y, False
    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=min_size, replace=True)
    return X[idx], y[idx], True


def _run_single_quant_analysis(X, y, feature_names, min_cases=50):
    """
    X, y で LR + LGB + アンサンブル最適化を実行。件数不足時はブートストラップで min_cases に補完。
    戻り値: dict（n_cases, n_positive, n_negative, lr_coef, lgb_importance, auc_lr, auc_lgb, accuracy_*,
            ensemble_alpha, auc_ensemble, accuracy_ensemble, bootstrapped）
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score
    n_orig = len(y)
    X, y, bootstrapped = _bootstrap_to_min_size(X, y, min_size=min_cases)
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    out = {
        "n_cases": len(y),
        "n_cases_orig": n_orig,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "feature_names": feature_names,
        "bootstrapped": bootstrapped,
    }
    prob_lr_te = None
    prob_lgb_te = None
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        lr.fit(X_tr, y_tr)
        out["lr_coef"] = list(zip(feature_names, lr.coef_[0].tolist()))
        out["lr_intercept"] = float(lr.intercept_[0])
        out["accuracy_lr"] = float(accuracy_score(y_te, lr.predict(X_te)))
        if len(np.unique(y_te)) >= 2:
            out["auc_lr"] = float(roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1]))
            prob_lr_te = lr.predict_proba(X_te)[:, 1]
        else:
            out["auc_lr"] = None
    except Exception as e:
        out["lr_error"] = str(e)
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgb_model.fit(X_tr, y_tr)
        out["accuracy_lgb"] = float(accuracy_score(y_te, lgb_model.predict(X_te)))
        if len(np.unique(y_te)) >= 2:
            out["auc_lgb"] = float(roc_auc_score(y_te, lgb_model.predict_proba(X_te)[:, 1]))
            prob_lgb_te = lgb_model.predict_proba(X_te)[:, 1]
        else:
            out["auc_lgb"] = None
        out["lgb_importance"] = list(zip(feature_names, lgb_model.feature_importances_.tolist()))
    except Exception as e:
        out["lgb_error"] = str(e)
    if prob_lr_te is not None and prob_lgb_te is not None and len(np.unique(y_te)) >= 2:
        best_alpha, best_auc, best_acc = _optimize_ensemble_ratio(prob_lr_te, prob_lgb_te, y_te)
        out["ensemble_alpha"] = best_alpha
        out["auc_ensemble"] = best_auc
        out["accuracy_ensemble"] = best_acc
    return out




def _compute_fisher_and_shrink(X, theta):
    """
    ロジスティック回帰パラメータに対する Fisher 情報行列を計算し、
    標準誤差に基づく soft-shrink 係数を返す。
    Returns: (fisher_inv, shrink_factors, ci_cross_zero)
    """
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float)
    z = X @ theta
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    w = np.clip(p * (1.0 - p), 1e-8, None)
    Xw = X * w[:, None]
    fisher = X.T @ Xw
    fisher_inv = np.linalg.pinv(fisher)
    se = np.sqrt(np.clip(np.diag(fisher_inv), 0.0, None))
    abs_beta = np.abs(theta)
    shrink_factors = abs_beta / (abs_beta + se + 1e-12)
    ci_cross_zero = (theta - 1.96 * se) * (theta + 1.96 * se) <= 0
    return fisher_inv, shrink_factors, ci_cross_zero


def _bayesian_map_estimate(X, y, prior_mu, prior_precision):
    """
    正規事前分布 N(μ₀, Λ₀⁻¹) を使ったロジスティック回帰の MAP 推定。

    X: (n, p) 特徴行列（intercept なし）
    y: (n,) バイナリラベル (0/1)
    prior_mu: (p,) 前回の係数ベクトル（intercept なし）
    prior_precision: (p, p) 前回の Fisher 情報行列（精度行列 Λ₀）

    Returns:
        coef_map: (p,) MAP 係数
        intercept_map: float
        posterior_precision: (p, p) 事後精度行列（次回の事前分布として使用）
        posterior_cov: (p, p) 事後共分散行列（fisher_inv 相当）
    """
    from scipy.optimize import minimize
    from scipy.special import expit

    n, p = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])  # intercept を最後の列に追加

    mu0 = np.asarray(prior_mu, dtype=float)
    Lambda0 = np.asarray(prior_precision, dtype=float)

    theta0 = np.append(mu0, 0.0)

    def _neg_log_posterior(theta):
        logits = Xb @ theta
        log_lik = np.sum(
            y * np.log(expit(logits) + 1e-15) +
            (1 - y) * np.log(1 - expit(logits) + 1e-15)
        )
        diff = theta[:p] - mu0
        log_prior = -0.5 * (diff @ Lambda0 @ diff)
        return -(log_lik + log_prior)

    def _grad(theta):
        p_hat = expit(Xb @ theta)
        grad_lik = Xb.T @ (p_hat - y)
        diff = theta[:p] - mu0
        grad_prior = np.append(Lambda0 @ diff, 0.0)  # intercept は正則化しない
        return grad_lik + grad_prior

    result = minimize(_neg_log_posterior, theta0, jac=_grad,
                      method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-12})

    theta_map = result.x
    coef_map = theta_map[:p]
    intercept_map = float(theta_map[p])

    # 事後精度行列 = 事前精度 + 新データの Fisher 情報行列
    p_hat = expit(Xb @ theta_map)
    w = np.clip(p_hat * (1 - p_hat), 1e-8, None)
    Xw = X * w[:, None]
    fisher_new = X.T @ Xw
    posterior_precision = Lambda0 + fisher_new
    posterior_cov = np.linalg.pinv(posterior_precision)

    return coef_map, intercept_map, posterior_precision, posterior_cov


def _load_prior_for_model(model_key: str):
    """
    coeff_auto.json から model_key の事前係数と精度行列を読み込む。
    存在しない場合は (None, None) を返す。
    """
    try:
        from data_cases import load_auto_coeffs
        auto = load_auto_coeffs()
        prior_mu_raw = auto.get(f"lr_coef_{model_key}")
        prior_prec_raw = auto.get(f"fisher_prec_{model_key}")
        if prior_mu_raw is None or prior_prec_raw is None:
            return None, None
        prior_mu = np.asarray(prior_mu_raw, dtype=float)
        prior_prec = np.asarray(prior_prec_raw, dtype=float)
        return prior_mu, prior_prec
    except Exception:
        return None, None


def _save_posterior_for_model(model_key: str, coef_arr, intercept_val, posterior_precision):
    """MAP 推定後の係数と事後精度行列を coeff_auto.json に保存する。"""
    try:
        from data_cases import load_auto_coeffs, save_auto_coeffs
        auto = load_auto_coeffs()
        auto[f"lr_coef_{model_key}"] = np.asarray(coef_arr, dtype=float).tolist()
        auto[f"lr_intercept_{model_key}"] = float(intercept_val)
        auto[f"fisher_prec_{model_key}"] = np.asarray(posterior_precision, dtype=float).tolist()
        save_auto_coeffs(auto)
    except Exception:
        pass


def run_regression_and_get_coeffs(X, y, model_key: str | None = None):
    """
    X, y に対してロジスティック回帰を実行し、既存項目＋追加項目の係数辞書を返す。
    X の列順: COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS。

    model_key が指定されると前回の係数を事前分布として MAP 推定を行い、
    推定後の係数と事後精度行列を coeff_auto.json に保存する（ベイズ更新）。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    p = X.shape[1]

    used_bayes = False
    if model_key:
        prior_mu, prior_prec = _load_prior_for_model(model_key)
        if prior_mu is not None and prior_prec is not None and len(prior_mu) == p:
            try:
                coef_map, intercept_map, post_prec, post_cov = _bayesian_map_estimate(
                    X, y, prior_mu, prior_prec
                )
                coefs = coef_map.tolist()
                intercept = intercept_map
                _save_posterior_for_model(model_key, coef_map, intercept_map, post_prec)
                used_bayes = True
                fisher_inv = post_cov
            except Exception:
                used_bayes = False

    sklearn_model = None
    if not used_bayes:
        from sklearn.linear_model import LogisticRegression
        sklearn_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
        sklearn_model.fit(X, y)
        intercept = float(sklearn_model.intercept_[0])
        raw_coefs = sklearn_model.coef_[0].astype(float)
        fisher_inv, shrink_factors, ci_cross_zero = _compute_fisher_and_shrink(X, raw_coefs)
        coefs_shrunk = raw_coefs.copy()
        coefs_shrunk[ci_cross_zero] = coefs_shrunk[ci_cross_zero] * shrink_factors[ci_cross_zero]
        coefs = coefs_shrunk.tolist()
        if model_key:
            # 初回: Fisher 情報行列を精度行列として保存（次回の事前分布になる）
            fisher_prec = np.linalg.pinv(fisher_inv)
            _save_posterior_for_model(model_key, np.array(coefs), intercept, fisher_prec)

    # 精度計算（MAP/MLE 共通）
    from scipy.special import expit as _expit
    logits = X @ np.array(coefs) + intercept
    y_pred_bin = (logits >= 0.0).astype(int)
    accuracy = float((y_pred_bin == y).mean())

    coeff_dict = {"intercept": intercept, "_used_bayesian_update": used_bayes, "_accuracy": accuracy}
    for i, key in enumerate(COEFF_MAIN_KEYS):
        if i < len(coefs):
            coeff_dict[key] = float(coefs[i])
    for j, key in enumerate(COEFF_EXTRA_KEYS):
        idx = len(COEFF_MAIN_KEYS) + j
        if idx < len(coefs):
            coeff_dict[key] = float(coefs[idx])
    return coeff_dict, sklearn_model


def _build_one_row_indicator(log, data):
    """1ログから指標モデル用の1行（ind+ratio+grade + 追加特徴）を構築。"""
    major = data["industry_major"]
    ind_medical = 1.0 if ("医療" in major or "福祉" in major or (isinstance(major, str) and major.startswith("P"))) else 0.0
    ind_transport = 1.0 if ("運輸" in major or (isinstance(major, str) and major.startswith("H"))) else 0.0
    ind_construction = 1.0 if ("建設" in major or (isinstance(major, str) and major.startswith("D"))) else 0.0
    ind_manufacturing = 1.0 if ("製造" in major or (isinstance(major, str) and major.startswith("E"))) else 0.0
    ind_service = 1.0 if ("卸売" in major or "小売" in major or "サービス" in major or (isinstance(major, str) and major[0] in ["I", "K", "M", "R"])) else 0.0
    grade = data["grade"]
    grade_4_6 = 1.0 if "4-6" in grade else 0.0
    grade_watch = 1.0 if "要注意" in grade else 0.0
    grade_none = 1.0 if "無格付" in grade else 0.0
    raw_nenshu = max(float(data["nenshu"] or 0), 1.0)
    raw_op = data["op_profit"] * 1000
    raw_gross = data["gross_profit"] * 1000
    raw_ord = data["ord_profit"] * 1000
    raw_net = data["net_income"] * 1000
    raw_fixed = data["machines"] * 1000 + data["other_assets"] * 1000
    raw_rent = data["rent_expense"] * 1000
    raw_dep = data["depreciation"] * 1000 + data["dep_expense"] * 1000
    raw_machines = data["machines"] * 1000
    ratio_op = raw_op / raw_nenshu if raw_nenshu else 0
    ratio_gross = raw_gross / raw_nenshu if raw_nenshu else 0
    ratio_ord = raw_ord / raw_nenshu if raw_nenshu else 0
    ratio_net = raw_net / raw_nenshu if raw_nenshu else 0
    ratio_fixed = raw_fixed / raw_nenshu if raw_nenshu else 0
    ratio_rent = raw_rent / raw_nenshu if raw_nenshu else 0
    ratio_dep = raw_dep / raw_nenshu if raw_nenshu else 0
    ratio_machines = raw_machines / raw_nenshu if raw_nenshu else 0
    row = [
        ind_service, ind_medical, ind_transport, ind_construction, ind_manufacturing,
        ratio_op, ratio_gross, ratio_ord, ratio_net, ratio_fixed, ratio_rent, ratio_dep, ratio_machines,
        grade_4_6, grade_watch, grade_none,
    ]
    inp, res = log.get("inputs") or {}, log.get("result") or {}
    main_bank = 1.0 if log.get("main_bank") == "メイン先" else 0.0
    competitor_present = 1.0 if log.get("competitor") == "競合あり" else 0.0
    competitor_none = 1.0 if log.get("competitor") == "競合なし" else 0.0
    y_pred, comp_rate = res.get("yield_pred"), log.get("competitor_rate")
    if y_pred is not None and comp_rate is not None and isinstance(comp_rate, (int, float)):
        rate_diff_z = max(-2.0, min(2.0, (float(y_pred) - float(comp_rate)) / 5.0))
    else:
        rate_diff_z = 0.0
    industry_sentiment_z = float(res.get("industry_sentiment_z", 0))
    qual = inp.get("qualitative") or {}
    tags = qual.get("strength_tags") or []
    qualitative_tag_score = min(sum(STRENGTH_TAG_WEIGHTS.get(t, DEFAULT_STRENGTH_WEIGHT) for t in tags), 10.0)
    qualitative_passion = 1.0 if qual.get("passion_text") else 0.0
    equity_ratio = float(res.get("user_eq") or 0)
    qsc_i = (res.get("qualitative_scoring_correction") or inp.get("qualitative_scoring")) or {}
    combined_i = qsc_i.get("combined_score") or qsc_i.get("weighted_score")
    qualitative_combined = (float(combined_i) / 100.0) if combined_i is not None else 0.0
    
    bn      = log.get("bn_engine") or {}
    bn_im   = bn.get("intermediate") or {}
    bn_approval_prob = float(bn.get("approval_prob") or 0)
    bn_fc   = float(bn_im.get("Financial_Creditworthiness") or 0)
    bn_hc   = float(bn_im.get("Hedge_Condition") or 0)
    bn_av   = float(bn_im.get("Asset_Value") or 0)

    # 定性スコアリング項目（sumaho13追加）
    qsc_items_i = qsc_i.get("items") or {}
    qual_weighted_i  = float(qsc_i.get("weighted_score") or 0) / 100.0
    qual_rank_i      = qsc_i.get("rank") or ""
    qual_rank_good_i = 1.0 if qual_rank_i in ("A", "B") else 0.0
    repayment_val_i  = (qsc_items_i.get("repayment_history") or {}).get("value") or 0
    qual_repayment_i = float(repayment_val_i) / 4.0
    new_ctx_i = _build_new_customer_context_features({
        "main_bank": log.get("main_bank") or inp.get("main_bank") or "非メイン先",
        "competitor": log.get("competitor") or inp.get("competitor") or "競合なし",
        "deal_source": log.get("deal_source") or inp.get("deal_source") or "その他",
        "contract_type": inp.get("contract_type") or log.get("contract_type") or "一般",
        "num_competitors": inp.get("num_competitors") or log.get("num_competitors") or "未入力",
        "deal_occurrence": inp.get("deal_occurrence") or log.get("deal_occurrence") or "不明",
        "competitor_rate": log.get("competitor_rate") or inp.get("competitor_rate"),
        "customer_type": inp.get("customer_type") or log.get("customer_type") or "既存先",
    })
    
    # 量子矛盾スコア (Q_risk) のオンザフライ計算
    quantum_risk_i = 0.0
    try:
        from evaluators.quantum_adapter import QuantumAdapter
        _qa = QuantumAdapter()
        if _qa.is_ready():
            quantum_risk_i = _qa.predict(log).risk
    except Exception:
        pass

    row.extend([main_bank, competitor_present, competitor_none, rate_diff_z, industry_sentiment_z,
                qualitative_tag_score, qualitative_passion, equity_ratio, qualitative_combined,
                bn_approval_prob, bn_fc, bn_hc, bn_av,
                qual_weighted_i, qual_rank_good_i, qual_repayment_i, quantum_risk_i,
                new_ctx_i["new_customer_main_bank"],
                new_ctx_i["new_customer_competitor_present"],
                new_ctx_i["new_customer_competitor_count"],
                new_ctx_i["new_customer_competitor_rate"],
                new_ctx_i["new_customer_deal_source_bank"],
                new_ctx_i["new_customer_deal_occurrence_nomination"],
                new_ctx_i["new_customer_deal_occurrence_comp"],
                new_ctx_i["new_customer_contract_auto"]])
    return row


def build_design_matrix_indicator_from_logs(all_logs, indicator_model_key):
    """
    指標モデル用の説明変数行列 X と目的変数 y を構築。
    indicator_model_key は "全体_指標_既存先" などの形式。該当するログのみ使用。
    """
    rows = []
    y_list = []
    for log in all_logs:
        if log.get("final_status") not in ["成約", "失注"]:
            continue
        if "inputs" not in log:
            continue
        # 「全体_指標_*」は業種問わず既存先/新規先のみで絞る（全業種を集約）
        if indicator_model_key.startswith("全体_指標_"):
            suffix = indicator_model_key[len("全体_指標_"):]  # "既存先" or "新規先"
            ct = log.get("customer_type") or "既存先"
            if ct != suffix:
                continue
        elif _get_indicator_model_key_from_log(log) != indicator_model_key:
            continue
        data = _log_to_data_scoring(log)
        row = _build_one_row_indicator(log, data)
        rows.append(row)
        y_list.append(1 if log.get("final_status") == "成約" else 0)
    if not rows:
        return None, None
    return np.array(rows, dtype=float), np.array(y_list, dtype=int)


def run_regression_indicator_and_get_coeffs(X, y, model_key: str | None = None):
    """
    指標モデル用の回帰。列順: INDICATOR_MAIN_KEYS + COEFF_EXTRA_KEYS。

    model_key が指定されると前回の係数を事前分布として MAP 推定を行う（ベイズ更新）。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    p = X.shape[1]

    used_bayes = False
    if model_key:
        prior_mu, prior_prec = _load_prior_for_model(model_key)
        if prior_mu is not None and prior_prec is not None and len(prior_mu) == p:
            try:
                coef_map, intercept_map, post_prec, post_cov = _bayesian_map_estimate(
                    X, y, prior_mu, prior_prec
                )
                coefs = coef_map.tolist()
                intercept = intercept_map
                _save_posterior_for_model(model_key, coef_map, intercept_map, post_prec)
                used_bayes = True
                fisher_inv = post_cov
            except Exception:
                used_bayes = False

    sklearn_model = None
    if not used_bayes:
        from sklearn.linear_model import LogisticRegression
        sklearn_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
        sklearn_model.fit(X, y)
        intercept = float(sklearn_model.intercept_[0])
        raw_coefs = sklearn_model.coef_[0].astype(float)
        fisher_inv, shrink_factors, ci_cross_zero = _compute_fisher_and_shrink(X, raw_coefs)
        coefs_shrunk = raw_coefs.copy()
        coefs_shrunk[ci_cross_zero] = coefs_shrunk[ci_cross_zero] * shrink_factors[ci_cross_zero]
        coefs = coefs_shrunk.tolist()
        if model_key:
            fisher_prec = np.linalg.pinv(fisher_inv)
            _save_posterior_for_model(model_key, np.array(coefs), intercept, fisher_prec)

    logits = X @ np.array(coefs) + intercept
    y_pred_bin = (logits >= 0.0).astype(int)
    accuracy = float((y_pred_bin == y).mean())

    coeff_dict = {"intercept": intercept, "_used_bayesian_update": used_bayes, "_accuracy": accuracy}
    for i, key in enumerate(INDICATOR_MAIN_KEYS):
        if i < len(coefs):
            coeff_dict[key] = float(coefs[i])
    for j, key in enumerate(COEFF_EXTRA_KEYS):
        idx = len(INDICATOR_MAIN_KEYS) + j
        if idx < len(coefs):
            coeff_dict[key] = float(coefs[idx])
    return coeff_dict, sklearn_model


# 成約に寄与する上位3ドライバーは回帰係数（全体_既存先）の絶対値で算出。
# 利用箇所: 成約の正体レポート画面、分析結果ダッシュボード先頭の3因子表示。
# =============================================================================
COEFF_LABELS = {
    "intercept": "定数項",
    "ind_medical": "業種: 医療・福祉",
    "ind_transport": "業種: 運輸",
    "ind_construction": "業種: 建設",
    "ind_manufacturing": "業種: 製造",
    "ind_service": "業種: サービス",
    "sales_log": "売上高(対数)",
    "bank_credit_log": "銀行与信(対数)",
    "lease_credit_log": "リース与信(対数)",
    "op_profit": "営業利益",
    "ord_profit": "経常利益",
    "net_income": "当期純利益",
    "machines": "機械装置",
    "other_assets": "その他資産",
    "rent": "賃借料",
    "gross_profit": "売上総利益",
    "depreciation": "減価償却",
    "dep_expense": "減価償却費",
    "rent_expense": "賃借料等",
    "grade_4_6": "格付4〜6",
    "grade_watch": "要注意",
    "grade_none": "無格付",
    "contracts": "契約数",
    "main_bank": "メイン取引先",
    "competitor_present": "競合あり",
    "competitor_none": "競合なし",
    "rate_diff_z": "金利差(有利)",
    "industry_sentiment_z": "業界景気動向",
    "qualitative_tag_score": "定性スコア(強みタグ)",
    "qualitative_passion": "熱意・裏事情",
    "equity_ratio": "自己資本比率",
    "qualitative_combined": "定性スコアリング合計(0-1)",
    "bn_approval_prob": "BN承認確率",
    "bn_fc": "BN財務信用度",
    "bn_hc": "BNヘッジ条件",
    "bn_av": "BN物件価値",
    "qual_weighted":    "定性加重スコア(0-1)",
    "qual_rank_good":   "定性優良ランク(A/B)",
    "qual_repayment":   "返済履歴スコア(0-1)",
    "quantum_risk":     "量子矛盾リスク",
    "new_customer_main_bank": "新規先×メイン先",
    "new_customer_competitor_present": "新規先×競合あり",
    "new_customer_competitor_count": "新規先×競合社数",
    "new_customer_competitor_rate": "新規先×競合提示金利",
    "new_customer_deal_source_bank": "新規先×銀行紹介",
    "new_customer_deal_occurrence_nomination": "新規先×指名案件",
    "new_customer_deal_occurrence_comp": "新規先×相見積もり",
    "new_customer_contract_auto": "新規先×自動車契約",
}


def run_contract_driver_analysis():
    """
    成約要因分析: 成約データのみ抽出し、共通項（平均財務・定性タグランキング）と
    成約に寄与する上位3ドライバー（回帰係数ベース）を返す。
    成約が5件未満の場合は None を返す。
    """
    from collections import Counter
    cases = load_all_cases()
    closed = [c for c in cases if c.get("final_status") == "成約"]
    if len(closed) < 5:
        return None
    # 平均財務数値（成約案件のみ）
    fin_keys = ["nenshu", "op_profit", "ord_profit", "net_income", "bank_credit", "lease_credit", "contracts"]
    fin_labels = {"nenshu": "売上高(千円)", "op_profit": "営業利益(千円)", "ord_profit": "経常利益(千円)", "net_income": "当期純利益(千円)", "bank_credit": "銀行与信(千円)", "lease_credit": "リース与信(千円)", "contracts": "契約数"}
    sums = {k: 0.0 for k in fin_keys}
    counts = {k: 0 for k in fin_keys}
    for c in closed:
        inp = c.get("inputs") or {}
        res = c.get("result") or {}
        for k in fin_keys:
            v = inp.get(k) if k in inp else res.get("user_eq") if k == "user_eq" else None
            if k == "contracts":
                v = inp.get(k)
            if v is not None and isinstance(v, (int, float)):
                sums[k] += float(v)
                counts[k] += 1
    avg_financials = {}
    for k in fin_keys:
        if counts[k] > 0:
            avg_financials[fin_labels.get(k, k)] = sums[k] / counts[k]
    user_eq_list = []
    for c in closed:
        res = c.get("result") or {}
        eq = res.get("user_eq")
        if eq is not None and isinstance(eq, (int, float)):
            user_eq_list.append(float(_equity_ratio_display(eq) or 0))
    if user_eq_list:
        avg_financials["自己資本比率(%)"] = sum(user_eq_list) / len(user_eq_list)
    # 定性タグ頻出ランキング
    tag_counter = Counter()
    for c in closed:
        inp = c.get("inputs") or {}
        qual = inp.get("qualitative") or {}
        for t in qual.get("strength_tags") or []:
            tag_counter[t] += 1
    tag_ranking = tag_counter.most_common(20)
    # 成約に寄与する上位3ドライバー（全体_既存先の係数で絶対値が大きい順）
    coeffs = get_effective_coeffs("全体_既存先")
    driver_candidates = [(k, coeffs.get(k, 0)) for k in (COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS) if k in coeffs]
    driver_candidates = [(k, v) for k, v in driver_candidates if isinstance(v, (int, float)) and k != "intercept"]
    driver_candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    top3_drivers = []
    for k, v in driver_candidates[:3]:
        label = COEFF_LABELS.get(k, k)
        direction = "プラス" if v > 0 else "マイナス"
        top3_drivers.append({"key": k, "label": label, "coef": v, "direction": direction})
    # 定性スコアリングの集計（成約案件で入力されているもののみ）
    # 失注案件も同様に集計（比較用）
    lost = [c for c in cases if c.get("final_status") == "失注"]

    def _collect_qual_data(case_list):
        """案件リストから定性スコアデータを集計して返す。"""
        weighted_list = []
        combined_list = []
        rank_cnt = Counter()
        item_sums = {}   # item_id -> list[float]
        for c in case_list:
            q = (c.get("result") or {}).get("qualitative_scoring_correction") \
                or (c.get("inputs") or {}).get("qualitative_scoring")
            if not q:
                continue
            w = q.get("weighted_score")
            if w is not None:
                weighted_list.append(float(w))
            comb = q.get("combined_score")
            if comb is not None:
                combined_list.append(float(comb))
            r = q.get("rank")
            if r:
                rank_cnt[r] += 1
            # 項目別スコア（items キー内: {item_id: {value: int, ...}}）
            for item_id, item_data in (q.get("items") or {}).items():
                val = item_data.get("value")
                if val is not None:
                    item_sums.setdefault(item_id, []).append(float(val))
        item_avg = {k: sum(v) / len(v) for k, v in item_sums.items()}
        return weighted_list, combined_list, rank_cnt, item_avg

    qual_weighted_list, qual_combined_list, rank_counter, item_avg_closed = _collect_qual_data(closed)
    qual_weighted_lost, qual_combined_lost, rank_counter_lost, item_avg_lost = _collect_qual_data(lost)

    qual_summary = None
    if qual_weighted_list or qual_combined_list or rank_counter:
        qual_summary = {
            # 成約案件の集計
            "avg_weighted":       sum(qual_weighted_list) / len(qual_weighted_list) if qual_weighted_list else None,
            "avg_combined":       sum(qual_combined_list) / len(qual_combined_list) if qual_combined_list else None,
            "n_with_qual":        len(qual_weighted_list) or len(qual_combined_list) or sum(rank_counter.values()),
            "rank_distribution":  dict(rank_counter.most_common()),
            "qual_weighted_list": qual_weighted_list,          # 分布ヒストグラム用
            "item_avg":           item_avg_closed,             # 項目別平均（成約）
            # 失注案件の比較データ
            "avg_weighted_lost":       sum(qual_weighted_lost) / len(qual_weighted_lost) if qual_weighted_lost else None,
            "n_with_qual_lost":        len(qual_weighted_lost) or sum(rank_counter_lost.values()),
            "rank_distribution_lost":  dict(rank_counter_lost.most_common()),
            "qual_weighted_lost":      qual_weighted_lost,     # 分布ヒストグラム用（失注）
            "item_avg_lost":           item_avg_lost,          # 項目別平均（失注）
        }
    return {
        "closed_cases": closed,
        "closed_count": len(closed),
        "avg_financials": avg_financials,
        "tag_ranking": tag_ranking,
        "top3_drivers": top3_drivers,
        "qualitative_summary": qual_summary,
    }


QUALITATIVE_ANALYSIS_MIN_CASES = 50  # 定性要因分析は成約+失注がこの件数以上で実施


def optimize_score_weights_from_regression():
    """
    成約/失注データを使って、回帰で以下を最適化する。
    - 借手スコア vs 物件スコア の重み（現在 85% / 15%）
    - 総合スコア vs 定性スコア の重み（現在 60% / 40%、ランク用）
    戻り値: None（件数不足）, または dict（recommended_borrower_pct, recommended_asset_pct,
            recommended_quant_pct, recommended_qual_pct, n_cases, auc_borrower_asset, auc_quant_qual）
    """
    cases = load_all_cases()
    registered = [c for c in cases if c.get("final_status") in ["成約", "失注"]]
    if len(registered) < QUALITATIVE_ANALYSIS_MIN_CASES:
        return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    # 借手・物件の抽出
    X_ba = []
    y_list = []
    for c in registered:
        res = c.get("result") or {}
        sb = res.get("score_borrower")
        ast = res.get("asset_score")
        if sb is None or ast is None:
            continue
        X_ba.append([float(sb), float(ast)])
        y_list.append(1 if c.get("final_status") == "成約" else 0)
    y_arr = np.array(y_list)
    if len(X_ba) < 20 or len(np.unique(y_arr)) < 2:
        return None
    X_ba = np.array(X_ba)
    scaler_ba = StandardScaler()
    X_ba_s = scaler_ba.fit_transform(X_ba)
    lr_ba = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_ba.fit(X_ba_s, y_arr)
    coef_ba = lr_ba.coef_[0]
    c_b, c_a = max(1e-6, coef_ba[0]), max(1e-6, coef_ba[1])
    total_ba = c_b + c_a
    rec_borrower = float(c_b / total_ba)
    rec_asset = float(c_a / total_ba)
    prob_ba = lr_ba.predict_proba(X_ba_s)[:, 1]
    auc_ba = float(roc_auc_score(y_arr, prob_ba))

    # 総合 vs 定性（ランク用）。定性が入っている案件のみ
    X_qq = []
    y_qq = []
    for c in registered:
        res = c.get("result") or {}
        inp = c.get("inputs") or {}
        fs = res.get("score")
        if fs is None:
            sb = res.get("score_borrower")
            ast = res.get("asset_score")
            if sb is not None and ast is not None:
                fs = 0.85 * float(sb) + 0.15 * float(ast)
        qw = None
        q = res.get("qualitative_scoring_correction") or inp.get("qualitative_scoring")
        if q:
            qw = q.get("weighted_score")
        if fs is not None and qw is not None:
            X_qq.append([float(fs), float(qw)])
            y_qq.append(1 if c.get("final_status") == "成約" else 0)
    out = {
        "n_cases": len(y_list),
        "recommended_borrower_pct": rec_borrower,
        "recommended_asset_pct": rec_asset,
        "auc_borrower_asset": auc_ba,
    }
    if len(X_qq) >= 20 and len(np.unique(y_qq)) >= 2:
        X_qq = np.array(X_qq)
        y_qq = np.array(y_qq)
        scaler_qq = StandardScaler()
        X_qq_s = scaler_qq.fit_transform(X_qq)
        lr_qq = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr_qq.fit(X_qq_s, y_qq)
        coef_qq = lr_qq.coef_[0]
        c_quant = max(1e-6, coef_qq[0])
        c_qual = max(1e-6, coef_qq[1])
        out["recommended_quant_pct"] = float(c_quant / (c_quant + c_qual))
        out["recommended_qual_pct"] = float(c_qual / (c_quant + c_qual))
        out["auc_quant_qual"] = float(roc_auc_score(y_qq, lr_qq.predict_proba(X_qq_s)[:, 1]))
        out["n_cases_with_qual"] = len(y_qq)
    else:
        out["recommended_quant_pct"] = 0.6
        out["recommended_qual_pct"] = 0.4
        out["n_cases_with_qual"] = len(X_qq) if X_qq else 0
    return out


def optimize_model_blend_weights():
    """
    ① 全体モデル / ② 指標モデル / ③ 業種別モデル の3モデル混合比率を
    成約/失注データからクロスバリデーションで最適化する。

    手順:
      1. ログから (score_borrower, bench_score, ind_score) と (成約=1/失注=0) を抽出
      2. KFold(5) で各フォールドの LogisticRegression 係数を平均
      3. 正の係数を正規化して (w_main, w_bench, w_ind) を推定
      4. OOF AUC を返す

    戻り値:
      None（件数不足）、または dict:
        {
          "w_main":  float,  # ① 全体モデルの重み
          "w_bench": float,  # ② 指標モデルの重み
          "w_ind":   float,  # ③ 業種別モデルの重み
          "auc_cv":  float,  # OOF AUC
          "n_cases": int,
        }
    """
    cases = load_all_cases()
    registered = [c for c in cases if c.get("final_status") in ["成約", "失注"]]
    if len(registered) < QUALITATIVE_ANALYSIS_MIN_CASES:
        return None

    rows = []
    y_list = []
    for c in registered:
        res = c.get("result") or {}
        sb  = res.get("score_borrower")   # ① 全体モデル (0-100)
        bsc = res.get("bench_score")       # ② 指標モデル (0-100)
        isc = res.get("ind_score")         # ③ 業種別モデル (0-100)
        if sb is None or bsc is None or isc is None:
            continue
        rows.append([float(sb), float(bsc), float(isc)])
        y_list.append(1 if c.get("final_status") == "成約" else 0)

    if len(rows) < 20 or len(set(y_list)) < 2:
        return None

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler

        X = np.array(rows)
        y = np.array(y_list)

        # --- KFold クロスバリデーション ---
        kf = StratifiedKFold(n_splits=min(5, len(set(y_list)) * 5), shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y), dtype=float)
        oof_valid = np.zeros(len(y), dtype=bool)
        coef_list = []

        for train_idx, val_idx in kf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
            if len(set(y_tr)) < 2:
                continue
            lr.fit(X_tr_s, y_tr)
            oof_probs[val_idx] = lr.predict_proba(X_val_s)[:, 1]
            oof_valid[val_idx] = True
            coef_list.append(lr.coef_[0])

        if not coef_list:
            return None

        # --- 係数平均 → 正規化 ---
        mean_coef = np.mean(coef_list, axis=0)
        pos_coef = np.maximum(mean_coef, 1e-6)   # 負係数をフロア
        total = pos_coef.sum()
        w_main, w_bench, w_ind = (pos_coef / total).tolist()

        # --- OOF AUC（予測が計算された行のみ） ---
        if oof_valid.sum() >= 10 and len(set(y[oof_valid])) >= 2:
            auc_cv = float(roc_auc_score(y[oof_valid], oof_probs[oof_valid]))
        else:
            auc_cv = None

        return {
            "w_main":  round(w_main, 4),
            "w_bench": round(w_bench, 4),
            "w_ind":   round(w_ind, 4),
            "auc_cv":  round(auc_cv, 4) if auc_cv else None,
            "n_cases": len(y_list),
            "n_scored_oof": int(oof_valid.sum()),
        }
    except Exception:
        return None


def run_qualitative_contract_analysis(qual_correction_items):
    """
    定性項目のみで成約/不成約をロジスティック回帰とLightGBMで分析する。
    使用項目: 取引区分, 競合状況, 顧客区分, 商談ソース, リース物件,
              設立・経営年数, 顧客安定性, 返済履歴, 事業将来性, 設置目的, メイン取引銀行（定性6項目）。
    成約+失注が QUALITATIVE_ANALYSIS_MIN_CASES 件以上あるときのみ実行可能。
    qual_correction_items: 定性スコアリング訂正項目のリスト（メインの QUALITATIVE_SCORING_CORRECTION_ITEMS を渡す）。
    戻り値: None（件数不足）, または dict（n_cases, feature_names, lr_coef, lgb_importance, auc_lr, auc_lgb, accuracy_lr, accuracy_lgb 等）
    """
    if not qual_correction_items:
        return None
    cases = load_all_cases()
    registered = [c for c in cases if c.get("final_status") in ["成約", "失注"]]
    if len(registered) < QUALITATIVE_ANALYSIS_MIN_CASES:
        return None
    # リース物件のユニーク一覧（ラベルエンコード用）
    asset_ids = list({(c.get("inputs") or {}).get("lease_asset_id") or (c.get("inputs") or {}).get("lease_asset_name") or "未選択" for c in registered})
    asset_to_idx = {a: i for i, a in enumerate(asset_ids)}
    # 定性項目ID（qual_correction_items の id）
    qual_ids = [it["id"] for it in qual_correction_items]
    feature_names = [
        "取引区分_メイン先",
        "競合状況_競合あり",
        "顧客区分_新規先",
        "商談ソース_銀行紹介",
        "リース物件",
    ] + [it["label"] for it in qual_correction_items]
    rows = []
    y_list = []
    for c in registered:
        inp = c.get("inputs") or {}
        status = c.get("final_status")
        y_list.append(1 if status == "成約" else 0)
        main_bank = c.get("main_bank") or inp.get("main_bank") or "非メイン先"
        competitor = c.get("competitor") or inp.get("competitor") or "競合なし"
        customer_type = c.get("customer_type") or inp.get("customer_type") or "既存先"
        deal_source = inp.get("deal_source") or "その他"
        asset_id = inp.get("lease_asset_id") or inp.get("lease_asset_name") or "未選択"
        row = [
            1.0 if main_bank == "メイン先" else 0.0,
            1.0 if competitor == "競合あり" else 0.0,
            1.0 if customer_type == "新規先" else 0.0,
            1.0 if deal_source == "銀行紹介" else 0.0,
            float(asset_to_idx.get(asset_id, 0)),
        ]
        q = (c.get("result") or {}).get("qualitative_scoring_correction") or inp.get("qualitative_scoring") or {}
        items = q.get("items") or {}
        for it in qual_correction_items:
            qid = it["id"]
            val = items.get(qid, {})
            if isinstance(val, dict):
                v = val.get("value")
            else:
                v = None
            if v is not None and isinstance(v, (int, float)):
                row.append(float(v))
            else:
                row.append(-1.0)
        rows.append(row)
    X = np.array(rows, dtype=float)
    y = np.array(y_list, dtype=int)
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    out = {
        "n_cases": len(y),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "feature_names": feature_names,
    }
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_tr, y_tr)
        out["lr_coef"] = list(zip(feature_names, lr.coef_[0].tolist()))
        out["lr_intercept"] = float(lr.intercept_[0])
        out["accuracy_lr"] = float(accuracy_score(y_te, lr.predict(X_te)))
        if len(np.unique(y_te)) >= 2:
            out["auc_lr"] = float(roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1]))
        else:
            out["auc_lr"] = None
    except Exception as e:
        out["lr_error"] = str(e)
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgb_model.fit(X_tr, y_tr)
        out["accuracy_lgb"] = float(accuracy_score(y_te, lgb_model.predict(X_te)))
        if len(np.unique(y_te)) >= 2:
            out["auc_lgb"] = float(roc_auc_score(y_te, lgb_model.predict_proba(X_te)[:, 1]))
            prob_lgb_te = lgb_model.predict_proba(X_te)[:, 1]
        else:
            out["auc_lgb"] = None
        out["lgb_importance"] = list(zip(feature_names, lgb_model.feature_importances_.tolist()))
        out["shap_importance"] = []
        try:
            import joblib as _jbl, os as _os2
            _mp = _os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), "data", "lgb_qual_model.joblib")
            _jbl.dump({"model": lgb_model, "feature_names": feature_names, "asset_to_idx": asset_to_idx}, _mp)
        except Exception:
            pass
    except Exception as e:
        out["lgb_error"] = str(e)
    return out


def _optimize_ensemble_ratio(prob_lr, prob_lgb, y_true):
    """LRとLGBの予測確率を alpha * p_lr + (1-alpha) * p_lgb でブレンドし、AUCを最大化する alpha を返す。"""
    from sklearn.metrics import roc_auc_score, accuracy_score
    best_alpha, best_auc, best_acc = 0.5, 0.0, 0.0
    for alpha in np.linspace(0, 1, 11):
        p_ens = alpha * prob_lr + (1 - alpha) * prob_lgb
        if len(np.unique(y_true)) < 2:
            continue
        auc = roc_auc_score(y_true, p_ens)
        pred = (p_ens >= 0.5).astype(int)
        acc = accuracy_score(y_true, pred)
        if auc >= best_auc:
            best_auc = auc
            best_alpha = alpha
            best_acc = acc
    return float(best_alpha), float(best_auc), float(best_acc)


def run_quantitative_contract_analysis():
    """
    定量項目（業種モデルと同様の説明変数）のみで成約/不成約をロジスティック回帰とRandomForestで分析する。
    build_design_matrix_from_logs と同じ X（COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS）を使用。
    成約+失注が QUALITATIVE_ANALYSIS_MIN_CASES 件以上あるときのみ実行可能。
    LR と RandomForest を比較し、RandomForest を本体モデルとして保存する。
    """
    all_logs = load_all_cases()
    X, y = build_design_matrix_from_logs(all_logs, model_key=None)
    if X is None or y is None or len(y) < QUALITATIVE_ANALYSIS_MIN_CASES:
        return None
    feature_names = COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    out = {
        "n_cases": len(y),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "feature_names": feature_names,
    }
    try:
        from scipy.special import expit as _expit_quant
        # ベイズ更新可能なら MAP 推定、なければ MLE にフォールバック
        _prior_mu, _prior_prec = _load_prior_for_model("全体_既存先")
        _p = X_tr.shape[1]
        _used_bayes_lr = False
        if _prior_mu is not None and _prior_prec is not None and len(_prior_mu) == _p:
            try:
                _coef_map, _int_map, _post_prec, _ = _bayesian_map_estimate(
                    X_tr, y_tr, _prior_mu, _prior_prec
                )
                prob_lr_te = _expit_quant(X_te @ _coef_map + _int_map)
                _lr_pred = (prob_lr_te >= 0.5).astype(int)
                out["lr_coef"] = list(zip(feature_names, _coef_map.tolist()))
                out["lr_intercept"] = float(_int_map)
                out["accuracy_lr"] = float(accuracy_score(y_te, _lr_pred))
                if len(np.unique(y_te)) >= 2:
                    out["auc_lr"] = float(roc_auc_score(y_te, prob_lr_te))
                out["lr_used_bayesian"] = True
                _used_bayes_lr = True
            except Exception:
                _used_bayes_lr = False
        if not _used_bayes_lr:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
            lr.fit(X_tr, y_tr)
            out["lr_coef"] = list(zip(feature_names, lr.coef_[0].tolist()))
            out["lr_intercept"] = float(lr.intercept_[0])
            out["accuracy_lr"] = float(accuracy_score(y_te, lr.predict(X_te)))
            if len(np.unique(y_te)) >= 2:
                out["auc_lr"] = float(roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1]))
            else:
                out["auc_lr"] = None
            out["lr_used_bayesian"] = False
    except Exception as e:
        out["lr_error"] = str(e)
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgb_model.fit(X_tr, y_tr)
        out["accuracy_lgb"] = float(accuracy_score(y_te, lgb_model.predict(X_te)))
        if len(np.unique(y_te)) >= 2:
            out["auc_lgb"] = float(roc_auc_score(y_te, lgb_model.predict_proba(X_te)[:, 1]))
        else:
            out["auc_lgb"] = None
        out["lgb_importance"] = list(zip(feature_names, lgb_model.feature_importances_.tolist()))
        # LGB モデルを保存（スコア計算時の本体モデル）
        try:
            import joblib as _jbl, os as _os2
            _mp = _os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), "data", "lgb_main_model.joblib")
            _jbl.dump({"model": lgb_model, "feature_names": feature_names}, _mp)
        except Exception:
            pass
    except Exception as e:
        out["lgb_error"] = str(e)
    return out


def compare_non_lr_models_and_stack(min_cases: int = 50, random_state: int = 42):
    """
    LGBM を基準に、非LRモデルの単体性能と stacking を比較する。

    返り値:
      {
        "n_cases": int,
        "n_positive": int,
        "n_negative": int,
        "base_models": {name: {...}},
        "stacking": {name: {...}},
        "selected_stackings": [...],
      }
    """
    all_logs = load_all_cases()
    X, y = build_design_matrix_from_logs(all_logs, model_key=None)
    if X is None or y is None or len(y) < min_cases:
        return None

    # customer_type を追跡して、既存先/新規先のAUCも見る
    segments = []
    rows = []
    y_list = []
    for log in all_logs:
        if log.get("final_status") not in ["成約", "失注"]:
            continue
        if "inputs" not in log:
            continue
        data = _log_to_data_scoring(log)
        row = _build_one_row_industry(log, data)
        rows.append(row)
        y_list.append(1 if log.get("final_status") == "成約" else 0)
        segments.append(log.get("customer_type") or "既存先")

    X = np.asarray(rows, dtype=float)
    y = np.asarray(y_list, dtype=int)
    if len(y) < min_cases or len(np.unique(y)) < 2:
        return None

    from sklearn.base import clone
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        import lightgbm as lgb
    except Exception as e:
        return {"error": f"lightgbm import failed: {e}"}

    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except Exception:
        XGBClassifier = None
        xgb_available = False

    cv = StratifiedKFold(n_splits=min(5, max(2, len(np.unique(y)) * 2)), shuffle=True, random_state=random_state)

    def _segment_auc(y_true, proba):
        out = {}
        for seg in sorted(set(segments)):
            idx = np.array([s == seg for s in segments], dtype=bool)
            if idx.sum() < 10 or len(np.unique(y_true[idx])) < 2:
                continue
            out[seg] = float(roc_auc_score(y_true[idx], proba[idx]))
        return out

    models = {}
    base_estimators = {
        "LGBM": lgb.LGBMClassifier(**LGBM_PARAMS),
        "RandomForest": RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=500,
                early_stopping=True,
                random_state=random_state,
            ),
        ),
    }
    if xgb_available:
        base_estimators["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )

    report = {
        "n_cases": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - int(y.sum())),
        "base_models": {},
        "stacking": {},
    }

    # 単体モデルの OOF
    oof_pred_map = {}
    for name, estimator in base_estimators.items():
        try:
            probs = cross_val_predict(
                estimator,
                X,
                y,
                cv=cv,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]
            auc = float(roc_auc_score(y, probs))
            acc = float(accuracy_score(y, (probs >= 0.5).astype(int)))
            seg_auc = _segment_auc(y, probs)
            oof_pred_map[name] = probs
            report["base_models"][name] = {
                "auc": auc,
                "accuracy": acc,
                "segment_auc": seg_auc,
            }
        except Exception as e:
            report["base_models"][name] = {"error": str(e)}

    # stacking 候補: 単体AUCの上位モデルを組み合わせる
    base_rank = sorted(
        (
            (name, data.get("auc", -1.0))
            for name, data in report["base_models"].items()
            if "auc" in data
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    if not base_rank:
        return report

    best_single_auc = base_rank[0][1]
    candidate_names = [name for name, _ in base_rank[:3]]

    # 2〜3モデルの stacking を評価（上位3モデルの全組み合わせ）
    from itertools import combinations
    stack_sets = []
    for r in (2, 3):
        for combo in combinations(candidate_names, r):
            stack_sets.append(("+".join(combo), list(combo)))

    selected_stackings = []
    for stack_name, names in stack_sets:
        estimators = [(nm, clone(base_estimators[nm])) for nm in names]
        try:
            stack = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
                cv=cv,
                stack_method="predict_proba",
                n_jobs=1,
                passthrough=False,
            )
            probs = cross_val_predict(
                stack,
                X,
                y,
                cv=cv,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]
            auc = float(roc_auc_score(y, probs))
            acc = float(accuracy_score(y, (probs >= 0.5).astype(int)))
            seg_auc = _segment_auc(y, probs)
            report["stacking"][stack_name] = {
                "models": names,
                "auc": auc,
                "accuracy": acc,
                "segment_auc": seg_auc,
            }
            if auc >= best_single_auc + 0.002:
                selected_stackings.append(stack_name)
        except Exception as e:
            report["stacking"][stack_name] = {"models": names, "error": str(e)}

    report["selected_stackings"] = selected_stackings
    report["best_single_model"] = base_rank[0][0]
    report["best_single_auc"] = best_single_auc
    return report


def run_quantitative_by_industry():
    """業種ベースごとに定量分析（LR+LGB+アンサンブル）。データ不足時は50件にブートストラップ。"""
    all_logs = load_all_cases()
    registered = [c for c in all_logs if c.get("final_status") in ["成約", "失注"]]
    if len(registered) < 10:
        return None
    feature_names = COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS
    results = {}
    for base in INDUSTRY_BASES:
        X, y = build_design_matrix_from_logs_by_industry(all_logs, base)
        if X is None or len(y) < 5:
            results[base] = {"skip": True, "reason": "データなしまたは5件未満"}
            continue
        try:
            results[base] = _run_single_quant_analysis(X, y, feature_names, min_cases=QUALITATIVE_ANALYSIS_MIN_CASES)
        except Exception as e:
            results[base] = {"skip": True, "reason": str(e)}
    return results


def run_quantitative_by_indicator():
    """指標ベースごとに定量分析（LR+LGB+アンサンブル）。データ不足時は50件にブートストラップ。"""
    all_logs = load_all_cases()
    registered = [c for c in all_logs if c.get("final_status") in ["成約", "失注"]]
    if len(registered) < 10:
        return None
    # 指標モデルは INDICATOR_MAIN_KEYS + COEFF_EXTRA_KEYS の先頭8（指標用は9列目なし）
    ind_extra = [k for k in COEFF_EXTRA_KEYS if k != "qualitative_combined"][:8]
    feature_names = INDICATOR_MAIN_KEYS + ind_extra
    results = {}
    for bench in BENCH_BASES:
        X, y = build_design_matrix_indicator_by_bench(all_logs, bench)
        if X is None or len(y) < 5:
            results[bench] = {"skip": True, "reason": "データなしまたは5件未満"}
            continue
        try:
            results[bench] = _run_single_quant_analysis(X, y, feature_names, min_cases=QUALITATIVE_ANALYSIS_MIN_CASES)
        except Exception as e:
            results[bench] = {"skip": True, "reason": str(e)}
    return results



# ──────────────────────────────────────────────────────────────────────────────
# Youden 指数で最適承認ラインを自動計算
# ──────────────────────────────────────────────────────────────────────────────

def calc_optimal_approval_line() -> dict | None:
    """
    成約/失注の過去データから ROC 曲線と Youden 指数を用いて
    最適な承認ライン（0〜100点スケール）を計算して返す。

    必要最小件数: 成約・失注それぞれ3件以上（合計6件以上）

    戻り値 dict:
        optimal       : int   最適承認ライン（点）
        youden_index  : float Youden 指数の最大値
        sensitivity   : float 感度（成約を承認と判定できた割合）
        specificity   : float 特異度（失注を要審議以下と判定できた割合）
        auc           : float AUC
        n_closed      : int   分析に使った成約件数
        n_lost        : int   分析に使った失注件数
        threshold_candidates : list[dict]  各閾値の詳細（上位5件）
        current_line  : int   現在の設定値（business_rules.json）
    """
    try:
        from sklearn.metrics import roc_curve, roc_auc_score
    except ImportError:
        return {"error": "scikit-learn が見つかりません。pip install scikit-learn でインストールしてください。"}

    cases = load_all_cases()
    registered = [c for c in cases if c.get("final_status") in ["成約", "失注"]]
    closed = [c for c in registered if c.get("final_status") == "成約"]
    lost   = [c for c in registered if c.get("final_status") == "失注"]

    if len(closed) < 3 or len(lost) < 3:
        return {
            "error": f"件数不足です（成約 {len(closed)}件・失注 {len(lost)}件）。"
                     "それぞれ3件以上の登録が必要です。",
            "n_closed": len(closed),
            "n_lost": len(lost),
        }

    scores, labels = [], []
    for c in registered:
        s = (c.get("result") or {}).get("score")
        if s is None:
            continue
        scores.append(float(s))
        labels.append(1 if c.get("final_status") == "成約" else 0)

    if len(scores) < 6:
        return {"error": "スコアが記録されていた案件が6件未満のため計算できません。"}

    import numpy as np
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
    youden_vals = tpr - fpr          # = 感度 + 特異度 - 1
    best_idx   = int(np.argmax(youden_vals))
    best_thresh = float(thresholds[best_idx])
    best_thresh_int = int(round(best_thresh))

    auc = float(roc_auc_score(labels_arr, scores_arr))

    # 感度・特異度を計算
    sensitivity = float(tpr[best_idx])
    specificity = float(1 - fpr[best_idx])

    # 上位5候補（Youden 指数が高い順）
    top_idx = np.argsort(youden_vals)[::-1][:5]
    candidates = []
    for i in top_idx:
        t = int(round(float(thresholds[i])))
        candidates.append({
            "threshold":   t,
            "youden":      round(float(youden_vals[i]), 4),
            "sensitivity": round(float(tpr[i]), 4),
            "specificity": round(float(1 - fpr[i]), 4),
        })

    # 現在の設定値を取得
    try:
        from rule_manager import load_business_rules
        _r = load_business_rules()
        current_line = int(_r.get("thresholds", {}).get("approval", 0.71) * 100)
    except Exception:
        current_line = 71

    return {
        "optimal":      best_thresh_int,
        "youden_index": round(float(youden_vals[best_idx]), 4),
        "sensitivity":  round(sensitivity, 4),
        "specificity":  round(specificity, 4),
        "auc":          round(auc, 4),
        "n_closed":     int(labels_arr.sum()),
        "n_lost":       int((1 - labels_arr).sum()),
        "threshold_candidates": candidates,
        "current_line": current_line,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 統合再学習: ベイズ warm-start LR（全モデルキー）+ RandomForest
# ──────────────────────────────────────────────────────────────────────────────

def run_bayesian_warm_start_all_keys(all_logs, min_n: int = 5):
    """
    全モデルキー（INDUSTRY_MODEL_KEYS + INDICATOR_MODEL_KEYS）について、
    前回の係数と Fisher 精度行列を事前分布として MAP 推定を実行し係数を更新する。

    事前分布の優先順位:
      1. coeff_auto.json の fisher_prec_{key}（精度行列あり → 真のMAP推定）
      2. coeff_overrides.json の係数のみ（精度行列なし → Fisher精度を単位行列で代替）

    係数は coeff_overrides.json に保存（スコアリングで使用）。
    Fisher 精度行列は coeff_auto.json に保存（次回の事前分布として使用）。

    Returns:
        (overrides_dict, results_list)
    """
    from data_cases import load_coeff_overrides, get_effective_coeffs
    from scipy.special import expit as _expit_ws
    from sklearn.metrics import accuracy_score as _acc_ws

    overrides = load_coeff_overrides() or {}
    results = []

    industry_feature_keys = COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS
    indicator_feature_keys = INDICATOR_MAIN_KEYS + COEFF_EXTRA_KEYS

    def _run_one(model_key, X, y, feature_keys):
        p = X.shape[1]
        # 事前分布: coeff_auto.json の精度行列 → なければ coeff_overrides.json の係数 + 単位行列
        prior_mu, prior_prec = _load_prior_for_model(model_key)
        if prior_mu is None or len(prior_mu) != p:
            eff = get_effective_coeffs(model_key)
            prior_mu = np.array([eff.get(k, 0.0) for k in feature_keys[:p]], dtype=float)
            prior_prec = np.eye(p)  # 無情報事前分布相当

        coef_map, int_map, post_prec, _ = _bayesian_map_estimate(X, y, prior_mu, prior_prec)
        _save_posterior_for_model(model_key, coef_map, int_map, post_prec)

        coeff_dict = {"intercept": float(int_map), "_used_bayesian_update": True}
        coeff_dict.update(dict(zip(feature_keys[:p], coef_map.tolist())))

        logits = X @ coef_map + int_map
        acc = float(_acc_ws(y, (logits >= 0).astype(int)))
        return coeff_dict, acc

    for model_key in INDUSTRY_MODEL_KEYS:
        X_k, y_k = build_design_matrix_from_logs(all_logs, model_key=model_key)
        n_k = len(y_k) if y_k is not None else 0
        if n_k < min_n:
            results.append(f"{model_key}: データ不足 ({n_k}件) — 前回係数を保持")
            continue
        try:
            coeff_dict, acc = _run_one(model_key, X_k, y_k, industry_feature_keys)
            overrides[model_key] = coeff_dict
            results.append(f"{model_key}: {n_k}件, Accuracy={acc:.1%} [MAP推定]")
        except Exception as e:
            results.append(f"{model_key}: エラー {e}")

    for ind_key in INDICATOR_MODEL_KEYS:
        X_i, y_i = build_design_matrix_indicator_from_logs(all_logs, ind_key)
        n_i = len(y_i) if y_i is not None else 0
        if n_i < min_n:
            results.append(f"{ind_key}: データ不足 ({n_i}件) — 前回係数を保持")
            continue
        try:
            coeff_dict, acc = _run_one(ind_key, X_i, y_i, indicator_feature_keys)
            overrides[ind_key] = coeff_dict
            results.append(f"{ind_key}: {n_i}件, Accuracy={acc:.1%} [MAP推定]")
        except Exception as e:
            results.append(f"{ind_key}: エラー {e}")

    return overrides, results


def _train_main_bundle_from_cases(all_logs, model_key: str | None = None, save_path: str | None = None):
    """指定セグメントの主モデルを学習し、保存用メタデータを返す。"""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from config import RF_PARAMS

    X, y = build_design_matrix_from_logs(all_logs, model_key=model_key)
    if X is None or y is None or len(y) < 5:
        raise ValueError(f"学習データ不足: {len(y) if y is not None else 0}件（5件以上必要）")

    if save_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_key == "全体_新規先":
            save_path = os.path.join(base_dir, "data", "lgb_main_model_new.joblib")
        elif model_key == "全体_既存先":
            save_path = os.path.join(base_dir, "data", "lgb_main_model.joblib")
        else:
            save_path = os.path.join(base_dir, "data", "lgbm_contract_model.pkl")

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    # 新規先は件数が少なく、木モデルが過学習しやすいので線形モデルに寄せる。
    if model_key == "全体_新規先":
        X_tr, y_tr = X, y
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=42,
        )
    else:
        # 既存先や共通モデルは木モデルを使う
        X_tr, y_tr, _ = _bootstrap_to_min_size(X, y, min_size=50)
        model = RandomForestClassifier(**RF_PARAMS)

    model.fit(X_tr, y_tr)

    acc = float(accuracy_score(y, model.predict(X)))
    auc = None
    if len(set(y.tolist())) >= 2:
        auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))

    feature_names = COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS
    if hasattr(model, "feature_importances_"):
        raw_importance = model.feature_importances_.tolist()
    elif hasattr(model, "coef_"):
        raw_importance = np.abs(model.coef_[0]).tolist()
    else:
        raw_importance = [0.0 for _ in feature_names]
    importance = list(zip(feature_names, raw_importance))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "n_cases": len(y),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "accuracy": acc,
        "auc": auc,
        "model_type": type(model).__name__,
        "importance": importance,
    }, save_path)

    return acc, auc, save_path, n_pos, n_neg


def train_lgbm_from_cases(all_logs, save_path: str | None = None, model_key: str | None = None):
    """
    成約/失注データで RandomForest を再学習し pkl に保存する。

    Returns:
        (accuracy, auc_or_None, save_path, n_positive, n_negative)
    """
    acc, auc, save_path, n_pos, n_neg = _train_main_bundle_from_cases(all_logs, model_key=model_key, save_path=save_path)

    if model_key is None:
        try:
            _train_main_bundle_from_cases(all_logs, model_key="全体_既存先")
        except Exception:
            pass
        try:
            _train_main_bundle_from_cases(all_logs, model_key="全体_新規先")
        except Exception:
            pass

    return acc, auc, save_path, n_pos, n_neg


def train_random_forest_from_cases(all_logs, save_path: str | None = None, model_key: str | None = None):
    """主モデル再学習の明示名。互換のため train_lgbm_from_cases と同じ動作。"""
    return train_lgbm_from_cases(all_logs, save_path=save_path, model_key=model_key)


def backup_coeff_overrides(backup_dir: str | None = None) -> str | None:
    """
    現在の coeff_overrides（全モデルキーの係数）をタイムスタンプ付き JSON に保存する。

    Returns:
        保存先パス（str）、または保存すべきデータがない場合は None
    """
    from datetime import datetime
    from data_cases import load_coeff_overrides

    current = load_coeff_overrides()
    if not current:
        return None

    if backup_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        backup_dir = os.path.join(base_dir, "data", "backups")

    os.makedirs(backup_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(backup_dir, f"coeff_backup_{ts}.json")

    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)

    return path


def backup_lgbm_model(lgbm_path: str | None = None, backup_dir: str | None = None) -> str | None:
    """
    既存の lgbm_contract_model.pkl をタイムスタンプ付きでコピー保存する。

    Returns:
        コピー先パス（str）、または元ファイルが存在しない場合は None
    """
    import shutil
    from datetime import datetime

    if lgbm_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        lgbm_path = os.path.join(base_dir, "data", "lgbm_contract_model.pkl")

    if not os.path.exists(lgbm_path):
        return None

    if backup_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        backup_dir = os.path.join(base_dir, "data", "backups")

    os.makedirs(backup_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(backup_dir, f"lgbm_backup_{ts}.pkl")
    shutil.copy2(lgbm_path, dest)

    return dest
