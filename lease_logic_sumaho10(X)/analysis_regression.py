"""
回帰・要因分析モジュール（lease_logic_sumaho10）
設計行列構築、ロジスティック回帰、定性/定量要因分析、成約ドライバー分析、重み最適化を提供。
"""
import os
import sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data_cases import load_all_cases, get_effective_coeffs
from coeff_definitions import COEFFS, BAYESIAN_PRIOR_EXTRA, STRENGTH_TAG_WEIGHTS, DEFAULT_STRENGTH_WEIGHT
from charts import _equity_ratio_display
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
    "rate_diff_z", "industry_sentiment_z", "qualitative_tag_score", "qualitative_passion",
    "equity_ratio",  # 自己資本比率（%）
    "qualitative_combined",  # 定性スコアリング合計（総合×60%＋定性×40%）を0-1で正規化
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
    return {
        "nenshu": nenshu, "bank_credit": bank_credit, "lease_credit": lease_credit,
        "op_profit": op_profit, "ord_profit": ord_profit, "net_income": net_income,
        "gross_profit": gross_profit, "machines": machines, "other_assets": other_assets,
        "rent": rent, "depreciation": depreciation, "dep_expense": dep_expense, "rent_expense": rent_expense,
        "contracts": contracts, "grade": grade, "industry_major": industry_major,
    }


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
    row.extend([main_bank, competitor_present, competitor_none, rate_diff_z, industry_sentiment_z, qualitative_tag_score, qualitative_passion, equity_ratio, qualitative_combined])
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
        if model_key is not None and _get_ind_key_from_log(log) != model_key:
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
    """業種ベース（全体/医療/運送業/サービス業/製造業）でフィルタし、業種モデル用 X, y を構築。"""
    rows = []
    y_list = []
    for log in all_logs:
        if log.get("final_status") not in ["成約", "失注"]:
            continue
        if "inputs" not in log:
            continue
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
        lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbosity=-1)
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


def run_regression_and_get_coeffs(X, y):
    """
    X, y に対してロジスティック回帰を実行し、既存項目＋追加項目の係数辞書を返す。
    X の列順: COEFF_MAIN_KEYS (22) + COEFF_EXTRA_KEYS (9)。
    """
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    model.fit(X, y)
    intercept = float(model.intercept_[0])
    coefs = model.coef_[0].tolist()
    coeff_dict = {"intercept": intercept}
    for i, key in enumerate(COEFF_MAIN_KEYS):
        if i < len(coefs):
            coeff_dict[key] = float(coefs[i])
    for j, key in enumerate(COEFF_EXTRA_KEYS):
        idx = len(COEFF_MAIN_KEYS) + j
        if idx < len(coefs):
            coeff_dict[key] = float(coefs[idx])
    return coeff_dict, model


def _build_one_row_indicator(log, data):
    """1ログから指標モデル用の1行（ind+ratio+grade 16 + 追加8）を構築。"""
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
    row.extend([main_bank, competitor_present, competitor_none, rate_diff_z, industry_sentiment_z, qualitative_tag_score, qualitative_passion, equity_ratio])
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
        if _get_indicator_model_key_from_log(log) != indicator_model_key:
            continue
        data = _log_to_data_scoring(log)
        row = _build_one_row_indicator(log, data)
        rows.append(row)
        y_list.append(1 if log.get("final_status") == "成約" else 0)
    if not rows:
        return None, None
    return np.array(rows, dtype=float), np.array(y_list, dtype=int)


def run_regression_indicator_and_get_coeffs(X, y):
    """指標モデル用の回帰。列順: INDICATOR_MAIN_KEYS (16) + COEFF_EXTRA_KEYS (8)。"""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    model.fit(X, y)
    intercept = float(model.intercept_[0])
    coefs = model.coef_[0].tolist()
    coeff_dict = {"intercept": intercept}
    for i, key in enumerate(INDICATOR_MAIN_KEYS):
        if i < len(coefs):
            coeff_dict[key] = float(coefs[i])
    for j, key in enumerate(COEFF_EXTRA_KEYS):
        idx = len(INDICATOR_MAIN_KEYS) + j
        if idx < len(coefs):
            coeff_dict[key] = float(coefs[idx])
    return coeff_dict, model


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
    qual_weighted_list = []
    qual_combined_list = []
    rank_counter = Counter()
    for c in closed:
        q = (c.get("result") or {}).get("qualitative_scoring_correction") or (c.get("inputs") or {}).get("qualitative_scoring")
        if not q:
            continue
        w = q.get("weighted_score")
        if w is not None:
            qual_weighted_list.append(float(w))
        comb = q.get("combined_score")
        if comb is not None:
            qual_combined_list.append(float(comb))
        r = q.get("rank")
        if r:
            rank_counter[r] += 1
    qual_summary = None
    if qual_weighted_list or qual_combined_list or rank_counter:
        qual_summary = {
            "avg_weighted": sum(qual_weighted_list) / len(qual_weighted_list) if qual_weighted_list else None,
            "avg_combined": sum(qual_combined_list) / len(qual_combined_list) if qual_combined_list else None,
            "n_with_qual": len(qual_weighted_list) or len(qual_combined_list) or sum(rank_counter.values()),
            "rank_distribution": dict(rank_counter.most_common()),
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
    prob_lr_te = None
    prob_lgb_te = None
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
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
        lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbosity=-1)
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
    定量項目（業種モデルと同様の説明変数）のみで成約/不成約をロジスティック回帰とLightGBMで分析する。
    build_design_matrix_from_logs と同じ X（COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS）を使用。
    成約+失注が QUALITATIVE_ANALYSIS_MIN_CASES 件以上あるときのみ実行可能。
    アンサンブル割合を最適化して返す。
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
        lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbosity=-1)
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

