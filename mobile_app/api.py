"""
めぶきちゃん簡易審査 — Flask API
顧客区分ごとのモデル構成:
  既存先: Step1 lgb_main_model.joblib（LGBM→sys_score_b）
          Step2 simple_model.pkl（RF 52特徴量）             AUC≈0.853
  新規先: rf_new_customer_v1.pkl（RF 52特徴量・新規先専用）  AUC≈0.653
POST /predict: リース成約スコア + 推奨金利を返す
推奨金利: spread_predictor_v2.pkl（RF回帰 50特徴量）
          + base_rate_master（最新月×リース期間別基準金利）
"""
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

# ── 既存先 Step2 / 共通: RF 52特徴量メインモデル ─────────────────────────
_bundle   = joblib.load(os.path.join(_HERE, "simple_model.pkl"))
_le_ind   = joblib.load(os.path.join(_HERE, "industry_encoder.pkl"))
_model    = _bundle["model"]       # 既存先の最終予測に使う
_features = _bundle["feature_names"]
_encoders = _bundle["encoders"]
_imputer  = _bundle["imputer"]
_medians  = dict(zip(_features, _imputer.statistics_))

# ── 新規先専用 RF モデル（52特徴量・新規先データのみで学習） ───────────────
_rf_new_bundle = None
_rf_new_model  = None
try:
    _rf_new_bundle = joblib.load(
        os.path.join(_PROJECT_ROOT, "data", "rf_new_customer_v1.pkl")
    )
    _rf_new_model = _rf_new_bundle["model"]
    print(f"[api] 新規先RF: {type(_rf_new_model).__name__} "
          f"n_cases={_rf_new_bundle['n_cases']} "
          f"holdout_auc={_rf_new_bundle.get('holdout_auc', 'n/a')} 読み込み完了")
except Exception as e:
    print(f"[api] 新規先RF: 未ロード → メインRFにフォールバック ({e})")

# ── 既存先 Step1: LGBM借手スコアモデル ───────────────────────────────────
_lgb_existing = None
try:
    _lgb_existing = joblib.load(
        os.path.join(_PROJECT_ROOT, "data", "lgb_main_model.joblib")
    )
    print(f"[api] 既存先LGBM: {type(_lgb_existing['model']).__name__} "
          f"n_cases={_lgb_existing['n_cases']} 読み込み完了")
except Exception as e:
    print(f"[api] 既存先LGBM: 未ロード → sys_score_bは中央値フォールバック ({e})")

print(f"[api] メインRF: {type(_model).__name__} ({len(_features)}特徴量) 読み込み完了")

# ── 推奨金利モデル: spread_predictor_v2.pkl ──────────────────────────────
_spread_bundle   = None
_spread_model    = None
_spread_imputer  = None
_spread_features = None
_spread_dept_map = {}
_brm_df          = None
_latest_ym       = None

try:
    _spread_bundle   = joblib.load(
        os.path.join(_PROJECT_ROOT, "data", "spread_predictor_v2.pkl")
    )
    _spread_model    = _spread_bundle["model"]
    _spread_imputer  = _spread_bundle["imputer"]
    _spread_features = _spread_bundle["feature_names"]
    _spread_dept_map = _spread_bundle.get("dept_map", {})
    _brm_df          = pd.DataFrame(_spread_bundle["brm_full"])
    _latest_ym       = max(_brm_df.index)
    print(f"[api] spread_predictor v2: {len(_spread_features)}特徴量 "
          f"RMSE={_spread_bundle.get('rmse','n/a')} "
          f"最新基準金利月={_latest_ym} 読み込み完了")
except Exception as e:
    print(f"[api] spread_predictor v2: 未ロード → 推奨金利は算出不可 ({e})")

# ── aurion: 財務矛盾検知モジュール (P2-002) ─────────────────────────────
_aurion_loaded = False
_detect_q_risk = None
try:
    from aurion.q_risk import detect_q_risk as _detect_q_risk
    _aurion_loaded = True
    print("[api] aurion.q_risk: 読み込み完了")
except Exception as e:
    print(f"[api] aurion.q_risk: 未ロード → フォールバック ({e})")

# ── aurion: ステルス競合推定モジュール (P3-002) ──────────────────────────
_stealth_loaded = False
try:
    from aurion.stealth_competitor import detect_stealth_competitor
    _stealth_loaded = True
    print("[api] aurion.stealth_competitor: 読み込み完了")
except ImportError as e:
    print(f"[api.aurion.stealth] import failed ({e})")

_DEFAULT_IND        = "R サービス業(他に分類されないもの)"
_DEFAULT_CONTRACT_T = "一般"
_DEFAULT_DEAL_SRC   = "銀行紹介"
_DEFAULT_SALES_DEPT = "足利営業部"


def _get_period_rate(lease_term_months: float) -> tuple:
    """リース期間（月数）から最新月の基準金利（%）と対応カラム名を返す。"""
    if _brm_df is None:
        return 2.65, "r_5y"
    term = float(lease_term_months) if lease_term_months else 60
    if term <= 24:
        col = "r_2y"
    elif term <= 36:
        col = "r_3y"
    elif term <= 48:
        col = "r_4y"
    elif term <= 60:
        col = "r_5y"
    elif term <= 84:
        col = "r_7y"
    else:
        col = "r_over9y"
    rate = float(_brm_df.loc[_latest_ym, col]) if col in _brm_df.columns else 2.65
    return rate, col


def _predict_spread(
    gp, op, ep, ni, dep, depr, ns, mach, oa, rent, rexp, bk, lc, acq, lt,
    contracts, gpm, ord_margin, net_margin, dep_ratio, bank_to_ns, lease_to_ns,
    mach_to_ns, acq_to_ns, op_margin, dep_to_loan,
    ind_code, customer_type, main_bank, competitor, grade,
    ct_code, ds_code, sales_dept_str,
    q_history, q_stability, q_repayment, q_future, q_equip, q_mainbk, q_weighted,
    sys_score, sys_score_b, sys_dscr, sys_op_m, sys_icr, sys_appr, sys_ind, sys_bench,
    lease_asset_score,
) -> float:
    """spread_predictor_v2 で予測スプレッド（%）を返す。モデル未ロード時は中央値。"""
    if _spread_model is None:
        return 2.5
    sd_spread = _spread_dept_map.get(sales_dept_str, 0)
    feat_map = {
        "gross_profit": gp,       "op_profit": op,        "ord_profit": ep,
        "net_income": ni,         "dep_expense": dep,     "depreciation": depr,
        "nenshu": ns,             "machines": mach,        "other_assets": oa,
        "rent": rent,             "rent_expense": rexp,
        "bank_credit": bk,        "lease_credit": lc,
        "gpm": gpm,               "ord_margin": ord_margin, "net_margin": net_margin,
        "dep_ratio": dep_ratio,   "bank_to_ns": bank_to_ns,
        "lease_to_ns": lease_to_ns, "mach_to_ns": mach_to_ns,
        "acq_to_ns": acq_to_ns,  "op_margin": op_margin,  "dep_to_loan": dep_to_loan,
        "acquisition_cost": acq,  "lease_term": lt,        "contracts": contracts,
        "lease_asset_score": lease_asset_score,
        "industry": ind_code,     "customer_type": customer_type,
        "main_bank": main_bank,   "competitor": competitor, "grade": grade,
        "contract_type": ct_code, "deal_source": ds_code,  "sales_dept": sd_spread,
        "q_history": q_history,   "q_stability": q_stability,
        "q_repayment": q_repayment, "q_future": q_future,
        "q_equip": q_equip,       "q_mainbk": q_mainbk,   "q_weighted": q_weighted,
        "sys_score": sys_score,   "sys_score_b": sys_score_b,
        "sys_dscr": sys_dscr,     "sys_op_margin": sys_op_m,
        "sys_icr": sys_icr,       "sys_approval": sys_appr,
        "sys_ind_score": sys_ind, "sys_bench": sys_bench,
    }
    X_raw = np.array([[feat_map.get(f, 0.0) for f in _spread_features]])
    X_imp = _spread_imputer.transform(X_raw)
    return float(_spread_model.predict(X_imp)[0])


def _enc(enc_key: str, val: str, default: str) -> int:
    le = _encoders[enc_key]
    v  = val if val in le.classes_ else default
    return int(le.transform([v])[0])


def safe_div(a, b, fill=0.0):
    return a / b if b and b != 0 else fill


def _compute_sys_score_b_existing(
    industry: str, main_bank: int, competitor: int, comp_rate: float,
    deal_source: str, contract_type: str,
    ns_m: float, bk_m: float, lc_m: float,
    op_m: float, ep_m: float, ni_m: float, gp_m: float,
    dep_m: float, depr_m: float, mach_m: float, oa_m: float,
    rent_m: float, rexp_m: float, contracts: float, grade: int,
) -> float:
    """
    既存先 Step1: LGBM借手スコアモデルで sys_score_b（0-100）を計算する。
    LGB の log変換特徴量は千円スケール（百万円×1000）で学習済み。
    LGBM未ロード時は学習データ中央値を返す。
    """
    if _lgb_existing is None:
        return float(_medians.get("sys_score_b", 50.0))

    feat_names = _lgb_existing["feature_names"]

    ind_medical      = 1.0 if ("医療" in industry or "福祉" in industry
                                or industry.startswith("P")) else 0.0
    ind_transport    = 1.0 if ("運輸" in industry
                                or industry.startswith("H")) else 0.0
    ind_construction = 1.0 if ("建設" in industry
                                or industry.startswith("D")) else 0.0
    ind_manufacturing= 1.0 if ("製造" in industry
                                or industry.startswith("E")) else 0.0
    ind_service      = 1.0 if (
        any(x in industry for x in ["卸売", "小売", "サービス"])
        or (bool(industry) and industry[0] in ["I", "K", "M", "R"])
    ) else 0.0

    grade_4_6   = 1.0 if grade == 2 else 0.0
    grade_watch = 1.0 if grade == 3 else 0.0
    grade_none  = 1.0 if grade == 4 else 0.0

    deal_source_bank = 1.0 if deal_source == "銀行紹介" else 0.0
    comp_present     = float(competitor)
    comp_none        = 1.0 if competitor == 0 else 0.0

    dscr_denom  = dep_m + rexp_m
    dscr_approx = op_m / dscr_denom if dscr_denom > 0 else 1.0

    # LGB の log変換特徴量は千円スケールで学習済み（百万円×1000 で換算）
    sales_log        = float(np.log1p(ns_m * 1000))
    bank_credit_log  = float(np.log1p(bk_m * 1000))
    lease_credit_log = float(np.log1p(lc_m * 1000))

    feat_vals = {
        "ind_medical": ind_medical, "ind_transport": ind_transport,
        "ind_construction": ind_construction, "ind_manufacturing": ind_manufacturing,
        "ind_service": ind_service,
        "sales_log": sales_log, "bank_credit_log": bank_credit_log,
        "lease_credit_log": lease_credit_log,
        "op_profit": op_m, "ord_profit": ep_m, "net_income": ni_m,
        "machines": mach_m, "other_assets": oa_m, "rent": rent_m,
        "gross_profit": gp_m, "depreciation": depr_m, "dep_expense": dep_m,
        "rent_expense": rexp_m,
        "grade_4_6": grade_4_6, "grade_watch": grade_watch, "grade_none": grade_none,
        "contracts": contracts,
        "main_bank": float(main_bank),
        "competitor_present": comp_present, "competitor_none": comp_none,
        "customer_new": 0.0, "deal_source_bank": deal_source_bank,
        "rate_diff_z": 0.0, "industry_sentiment_z": 0.0,
        "qualitative_tag_score": 0.0, "qualitative_passion": 0.0,
        "equity_ratio": 0.0, "dscr_approx": dscr_approx, "interest_coverage": 10.0,
        "qualitative_combined": 0.0,
        "bn_approval_prob": 0.0, "bn_fc": 0.0, "bn_hc": 0.0, "bn_av": 0.0,
        "qual_weighted": 0.0, "qual_rank_good": 0.0, "qual_repayment": 0.0,
        "quantum_risk": 0.0,
        "new_customer_main_bank": 0.0, "new_customer_competitor_present": 0.0,
        "new_customer_competitor_count": 0.0, "new_customer_competitor_rate": 0.0,
        "new_customer_deal_source_bank": 0.0,
        "new_customer_deal_occurrence_nomination": 0.0,
        "new_customer_deal_occurrence_comp": 0.0, "new_customer_contract_auto": 0.0,
    }

    X = np.array([[feat_vals.get(f, 0.0) for f in feat_names]])
    try:
        return float(_lgb_existing["model"].predict_proba(X)[0][1]) * 100.0
    except Exception:
        return float(_medians.get("sys_score_b", 50.0))


def _build_feat_vector(
    gp, op, ep, ni, dep, depr, ns, mach, oa, rent, rexp, bk, lc, acq, lt,
    contracts, gpm, ord_margin, net_margin, dep_ratio, bank_to_ns, lease_to_ns,
    mach_to_ns, acq_to_ns, op_margin, dep_to_loan,
    ind_code, customer_type, main_bank, competitor, comp_rate, grade,
    ct_code, ds_code, sd_code, base_rate,
    q_history, q_stability, q_repayment, q_future, q_equip, q_mainbk, q_weighted,
    sys_score, sys_score_b, sys_dscr, sys_op_m, sys_icr, sys_appr, sys_ind, sys_bench,
    lease_asset_score, winning_spread=None,
) -> np.ndarray:
    """53特徴量ベクトル（winning_spread含む）を組み立てて impute し返す。"""
    # winning_spread 未指定時は imputer 中央値を使用
    ws = winning_spread if winning_spread is not None else float(_medians.get("winning_spread", 2.5))
    feat_map = {
        "gross_profit": gp,    "op_profit": op,      "ord_profit": ep,
        "net_income": ni,      "dep_expense": dep,   "depreciation": depr,
        "nenshu": ns,          "machines": mach,     "other_assets": oa,
        "rent": rent,          "rent_expense": rexp,
        "bank_credit": bk,     "lease_credit": lc,
        "gpm": gpm,            "ord_margin": ord_margin,  "net_margin": net_margin,
        "dep_ratio": dep_ratio, "bank_to_ns": bank_to_ns,
        "lease_to_ns": lease_to_ns, "mach_to_ns": mach_to_ns,
        "acq_to_ns": acq_to_ns, "op_margin": op_margin, "dep_to_loan": dep_to_loan,
        "acquisition_cost": acq, "lease_term": lt,
        "contracts": contracts,  "lease_asset_score": lease_asset_score,
        "industry": ind_code,  "customer_type": customer_type,
        "main_bank": main_bank, "competitor": competitor,
        "competitor_rate": comp_rate, "grade": grade,
        "contract_type": ct_code, "deal_source": ds_code, "sales_dept": sd_code,
        "base_rate": base_rate,
        "q_history": q_history, "q_stability": q_stability,
        "q_repayment": q_repayment, "q_future": q_future,
        "q_equip": q_equip, "q_mainbk": q_mainbk, "q_weighted": q_weighted,
        "sys_score": sys_score, "sys_score_b": sys_score_b,
        "sys_dscr": sys_dscr,  "sys_op_margin": sys_op_m,
        "sys_icr": sys_icr,    "sys_approval": sys_appr,
        "sys_ind_score": sys_ind, "sys_bench": sys_bench,
        "winning_spread": ws,
    }
    return np.array([[feat_map[f] for f in _features]])


_STATIC_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp"}


@app.get("/")
def index():
    return send_from_directory(_HERE, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _STATIC_EXTS:
        return "", 404
    return send_from_directory(_HERE, filename)


@app.post("/predict")
def predict():
    data = request.get_json(force=True, silent=True) or {}
    try:
        # ── UI から受け取る主要フィールド（百万円単位）──────────────
        gp   = float(data.get("gross_profit",    0))
        op   = float(data.get("op_profit",       0))
        ep   = float(data.get("ord_profit",      0))
        ni   = float(data.get("net_income",      0))
        dep  = float(data.get("dep_expense",     0))
        ns   = float(data.get("nenshu",          0))
        acq  = float(data.get("acquisition_cost",0))
        lc   = float(data.get("lease_credit",    0))
        bk   = float(data.get("bank_credit",     0))
        mach = float(data.get("machines",        0))
        lt   = float(data.get("lease_term",      60))
        contracts = float(data.get("contracts",  0))

        industry      = str(data.get("industry",      _DEFAULT_IND))
        customer_type = 1 if data.get("customer_type") == "既存先" else 0
        main_bank     = 1 if data.get("main_bank")    == "メイン先" else 0
        competitor    = 1 if data.get("competitor")   == "競合あり" else 0
        comp_rate     = float(data.get("competitor_rate", 0))
        contract_type = str(data.get("contract_type", _DEFAULT_CONTRACT_T))
        deal_source   = str(data.get("deal_source",   _DEFAULT_DEAL_SRC))
        sales_dept    = str(data.get("sales_dept",    _DEFAULT_SALES_DEPT))
        grade         = int(data.get("grade", 4))

    except (TypeError, ValueError) as e:
        return jsonify({"error": f"数値変換エラー: {e}"}), 400

    # P1-002: リースルールチェック用フィールド（全てオプション）
    asset_type_str = str(data.get("asset_type", ""))
    is_re_lease    = bool(data.get("is_re_lease", False))
    insurance_app  = str(data.get("insurance_applicable", "不明"))
    re_lease_ins   = str(data.get("re_lease_insurance", "不明"))

    # ── 派生比率 ──────────────────────────────────────────────────
    depr = float(data.get("depreciation", 0))
    oa   = float(data.get("other_assets", 0))
    rent = float(data.get("rent", 0))
    rexp = float(data.get("rent_expense", 0))

    gpm         = safe_div(gp, ns)
    ord_margin  = safe_div(ep, gp)
    net_margin  = safe_div(ni, gp)
    dep_ratio   = safe_div(dep, gp)
    bank_to_ns  = safe_div(bk, ns)
    lease_to_ns = safe_div(lc, ns)
    mach_to_ns  = safe_div(mach, ns)
    acq_to_ns   = safe_div(acq, ns)
    op_margin   = safe_div(op, ns)
    dep_to_loan = safe_div(dep, bk + lc) if (bk + lc) > 0 else 0.0

    # ── カテゴリエンコード ─────────────────────────────────────────
    ind_code = _enc("industry",      industry,      _DEFAULT_IND)
    ct_code  = _enc("contract_type", contract_type, _DEFAULT_CONTRACT_T)
    ds_code  = _enc("deal_source",   deal_source,   _DEFAULT_DEAL_SRC)
    sd_code  = _enc("sales_dept",    sales_dept,    _DEFAULT_SALES_DEPT)

    # ── 定性・補助スコア（中央値フォールバック）────────────────────
    base_rate   = float(data.get("base_rate",    _medians.get("base_rate",   1.5)))
    q_history   = float(data.get("q_history",    _medians.get("q_history",   3.0)))
    q_stability = float(data.get("q_stability",  _medians.get("q_stability", 3.0)))
    q_repayment = float(data.get("q_repayment",  _medians.get("q_repayment", 4.0)))
    q_future    = float(data.get("q_future",     _medians.get("q_future",    2.0)))
    q_equip     = float(data.get("q_equip",      _medians.get("q_equip",     3.0)))
    q_mainbk    = float(data.get("q_mainbk",     _medians.get("q_mainbk",    1.0)))
    q_weighted  = float(data.get("q_weighted",   _medians.get("q_weighted",  70.0)))
    sys_score   = _medians.get("sys_score",      50.0)
    sys_dscr    = _medians.get("sys_dscr",        0.5)
    sys_op_m    = _medians.get("sys_op_margin",   5.0)
    sys_icr     = _medians.get("sys_icr",        10.0)
    sys_appr    = _medians.get("sys_approval",   71.0)
    sys_ind     = _medians.get("sys_ind_score",  50.0)
    sys_bench   = _medians.get("sys_bench",      50.0)
    las         = float(data.get("lease_asset_score", 78))

    common = dict(
        gp=gp, op=op, ep=ep, ni=ni, dep=dep, depr=depr, ns=ns,
        mach=mach, oa=oa, rent=rent, rexp=rexp, bk=bk, lc=lc, acq=acq, lt=lt,
        contracts=contracts, gpm=gpm, ord_margin=ord_margin, net_margin=net_margin,
        dep_ratio=dep_ratio, bank_to_ns=bank_to_ns, lease_to_ns=lease_to_ns,
        mach_to_ns=mach_to_ns, acq_to_ns=acq_to_ns, op_margin=op_margin,
        dep_to_loan=dep_to_loan, ind_code=ind_code, customer_type=customer_type,
        main_bank=main_bank, competitor=competitor, comp_rate=comp_rate, grade=grade,
        ct_code=ct_code, ds_code=ds_code, sd_code=sd_code, base_rate=base_rate,
        q_history=q_history, q_stability=q_stability, q_repayment=q_repayment,
        q_future=q_future, q_equip=q_equip, q_mainbk=q_mainbk, q_weighted=q_weighted,
        sys_score=sys_score, sys_dscr=sys_dscr, sys_op_m=sys_op_m, sys_icr=sys_icr,
        sys_appr=sys_appr, sys_ind=sys_ind, sys_bench=sys_bench, lease_asset_score=las,
    )

    # ── sys_score_b 計算（顧客区分で分岐）──────────────────────────
    if customer_type == 0:
        sys_score_b    = float(_medians.get("sys_score_b", 50.0))
        model_pipeline = "NewCust_RF"
        use_model      = _rf_new_model if _rf_new_model is not None else _model
        if _rf_new_model is None:
            model_pipeline = "MainRF_fallback"
    else:
        sys_score_b = _compute_sys_score_b_existing(
            industry=industry, main_bank=main_bank, competitor=competitor,
            comp_rate=comp_rate, deal_source=deal_source, contract_type=contract_type,
            ns_m=ns, bk_m=bk, lc_m=lc,
            op_m=op, ep_m=ep, ni_m=ni, gp_m=gp,
            dep_m=dep, depr_m=depr,
            mach_m=mach, oa_m=oa, rent_m=rent, rexp_m=rexp,
            contracts=contracts, grade=grade,
        )
        model_pipeline = "LGBM→RF" if _lgb_existing else "RF_only"
        use_model      = _model

    # ── 推奨金利計算（分類モデルへの winning_spread 入力として使用）──
    base_rate_val, _col = _get_period_rate(lt)
    spread_pred = _predict_spread(
        gp=gp, op=op, ep=ep, ni=ni, dep=dep, depr=depr, ns=ns,
        mach=mach, oa=oa, rent=rent, rexp=rexp, bk=bk, lc=lc, acq=acq, lt=lt,
        contracts=contracts, gpm=gpm, ord_margin=ord_margin, net_margin=net_margin,
        dep_ratio=dep_ratio, bank_to_ns=bank_to_ns, lease_to_ns=lease_to_ns,
        mach_to_ns=mach_to_ns, acq_to_ns=acq_to_ns, op_margin=op_margin,
        dep_to_loan=dep_to_loan, ind_code=ind_code, customer_type=customer_type,
        main_bank=main_bank, competitor=competitor, grade=grade,
        ct_code=ct_code, ds_code=ds_code, sales_dept_str=sales_dept,
        q_history=q_history, q_stability=q_stability, q_repayment=q_repayment,
        q_future=q_future, q_equip=q_equip, q_mainbk=q_mainbk, q_weighted=q_weighted,
        sys_score=sys_score, sys_score_b=sys_score_b,
        sys_dscr=sys_dscr, sys_op_m=sys_op_m, sys_icr=sys_icr,
        sys_appr=sys_appr, sys_ind=sys_ind, sys_bench=sys_bench,
        lease_asset_score=las,
    )
    spread_pred       = max(-0.5, min(6.0, spread_pred))
    recommended_rate  = spread_pred + base_rate_val

    # ── 分類モデルで成約スコア予測 ──────────────────────────────────
    if customer_type == 0 and _rf_new_model is not None:
        # 新規先専用モデルは52特徴量（winning_spread未学習）→ そのまま渡す
        rf_new_feats = _rf_new_bundle["feature_names"]
        rf_new_imp   = _rf_new_bundle["imputer"]
        feat_map_new = {
            "gross_profit": gp,    "op_profit": op,      "ord_profit": ep,
            "net_income": ni,      "dep_expense": dep,   "depreciation": depr,
            "nenshu": ns,          "machines": mach,     "other_assets": oa,
            "rent": rent,          "rent_expense": rexp,
            "bank_credit": bk,     "lease_credit": lc,
            "gpm": gpm,            "ord_margin": ord_margin, "net_margin": net_margin,
            "dep_ratio": dep_ratio, "bank_to_ns": bank_to_ns,
            "lease_to_ns": lease_to_ns, "mach_to_ns": mach_to_ns,
            "acq_to_ns": acq_to_ns, "op_margin": op_margin, "dep_to_loan": dep_to_loan,
            "acquisition_cost": acq, "lease_term": lt, "contracts": contracts,
            "lease_asset_score": las, "industry": ind_code, "customer_type": customer_type,
            "main_bank": main_bank, "competitor": competitor,
            "competitor_rate": comp_rate, "grade": grade,
            "contract_type": ct_code, "deal_source": ds_code, "sales_dept": sd_code,
            "base_rate": base_rate,
            "q_history": q_history, "q_stability": q_stability,
            "q_repayment": q_repayment, "q_future": q_future,
            "q_equip": q_equip, "q_mainbk": q_mainbk, "q_weighted": q_weighted,
            "sys_score": sys_score, "sys_score_b": sys_score_b,
            "sys_dscr": sys_dscr, "sys_op_margin": sys_op_m,
            "sys_icr": sys_icr, "sys_approval": sys_appr,
            "sys_ind_score": sys_ind, "sys_bench": sys_bench,
        }
        X_raw = np.array([[feat_map_new.get(f, 0.0) for f in rf_new_feats]])
        X     = rf_new_imp.transform(X_raw)
    else:
        # 既存先（53特徴量: winning_spread含む）
        X = _build_feat_vector(**common, sys_score_b=sys_score_b, winning_spread=spread_pred)
    proba = float(use_model.predict_proba(X)[0, 1])

    score = int(round(proba * 100))
    if score >= 65:
        judgment = "承認"
    elif score >= 45:
        judgment = "条件付"
    else:
        judgment = "否認"

    # P1-002: リースルールチェック（BR-111〜BR-115）
    try:
        from lease_rule_checks import check_lease_rules
        rule_result       = check_lease_rules(
            lease_term_months=int(lt),
            asset_type=asset_type_str,
            is_re_lease=is_re_lease,
            insurance_applicable=insurance_app,
            re_lease_insurance=re_lease_ins,
        )
        warnings          = rule_result["warnings"]
        rule_check_status = rule_result["status"]
    except Exception:
        warnings          = []
        rule_check_status = "skipped"

    if rule_check_status not in ("ok", "unknown"):
        codes = [w["code"] for w in warnings]
        print(f"[api] rule_check_status={rule_check_status} warnings={codes}")

    # P2-002: aurion q_risk 検知（参考値、スコアに影響しない）
    _Q_RISK_FALLBACK = {"score": 0, "level": "ok", "patterns": [], "pattern_details": []}
    try:
        q_risk_result = _detect_q_risk(
            gross_profit=gp,
            op_profit=op,
            net_income=ni,
            nenshu=ns,
            dep_expense=dep,
            depreciation=depr,
            machines=mach,
            bank_credit=bk,
            lease_credit=lc,
            acquisition_cost=acq,
        )
    except Exception:
        q_risk_result = _Q_RISK_FALLBACK

    if q_risk_result["level"] in ("caution", "high_risk"):
        print(f"[api.aurion] level={q_risk_result['level']} score={q_risk_result['score']} patterns={q_risk_result['patterns']}")

    # P3-002: aurion stealth_competitor 検知（参考値、スコアに影響しない）
    # 提案金利（出来上がり金利）が入力されていればベースレートを差し引いてスプレッドに変換し優先使用
    _proposed_rate_raw = data.get("proposed_rate")
    try:
        if _proposed_rate_raw is not None:
            stealth_spread = float(_proposed_rate_raw) - base_rate_val
        else:
            stealth_spread = spread_pred
    except (TypeError, ValueError):
        stealth_spread = spread_pred

    _STEALTH_FALLBACK = {"score": 0, "level": "ok", "patterns": [], "pattern_details": []}
    try:
        stealth_result = detect_stealth_competitor(
            spread_pred=stealth_spread,
            base_rate=base_rate_val,
            competitor=competitor,
            competitor_rate=comp_rate,
            grade=grade,
            acquisition_cost=acq,
            nenshu=ns,
        ) if _stealth_loaded else _STEALTH_FALLBACK
    except Exception:
        stealth_result = _STEALTH_FALLBACK

    if stealth_result["level"] in ("caution", "high_risk"):
        print(f"[api.aurion.stealth] level={stealth_result['level']} score={stealth_result['score']} patterns={stealth_result['patterns']}")

    return jsonify({
        "score":             score,
        "probability":       round(proba, 4),
        "judgment":          judgment,
        "sys_score_b":       round(sys_score_b, 1),
        "model_pipeline":    model_pipeline,
        "spread_pred":       round(spread_pred, 2),
        "spread_proposed":   round(stealth_spread, 2) if _proposed_rate_raw is not None else None,
        "base_rate":         round(base_rate_val, 2),
        "recommended_rate":  round(recommended_rate, 2),
        "rate_range": {
            "low":  round(recommended_rate - 0.3, 2),
            "high": round(recommended_rate + 0.3, 2),
        },
        "warnings":          warnings,
        "rule_check_status": rule_check_status,
        "aurion": {
            "q_risk": q_risk_result,
            "competitor_pressure": stealth_result,
        },
    })


@app.get("/health")
def health():
    return jsonify({
        "status":          "ok",
        "model_existing":  type(_model).__name__,
        "model_new":       type(_rf_new_model).__name__ if _rf_new_model else None,
        "version":         _bundle.get("version", "unknown"),
        "n_features":      len(_features),
        "industries":      list(_le_ind.classes_),
        "lgbm_existing":   _lgb_existing is not None,
        "rf_new_loaded":   _rf_new_model is not None,
        "new_cust_auc":    _rf_new_bundle.get("holdout_auc") if _rf_new_bundle else None,
        "pipeline_existing": "LGBM→RF" if _lgb_existing else "RF_only",
        "pipeline_new":      "NewCust_RF" if _rf_new_model else "MainRF_fallback",
        "spread_model_loaded": _spread_model is not None,
        "spread_model_rmse":   _spread_bundle.get("rmse") if _spread_bundle else None,
        "spread_base_rate_ym": _latest_ym,
        "aurion_module_loaded": _aurion_loaded,
        "stealth_competitor_module_loaded": _stealth_loaded,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
