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
import json
import time
import uuid
import sqlite3 as _sqlite3
import pathlib
import datetime as _datetime
import threading
import unicodedata
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def _normalize_numeric(value, default=0.0):
    """全角数字・全角記号を半角化して float にする。"""
    if value in (None, ""):
        return default
    if isinstance(value, str):
        value = unicodedata.normalize("NFKC", value).strip()
        value = value.replace(",", "")
        if value == "":
            return default
    return float(value)


def _normalize_int(value, default=0):
    return int(_normalize_numeric(value, default))

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
import sys as _sys_mod
if _PROJECT_ROOT not in _sys_mod.path:
    _sys_mod.path.insert(0, _PROJECT_ROOT)

from data_cases import load_past_cases
from industry_normalizer import normalize_industry_major, normalize_industry_sub

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

# ── scoring_core: sys_score を Streamlit と同じロジックで計算 ──────────────
_scoring_core_loaded = False
_run_quick_scoring = None
try:
    from scoring_core import run_quick_scoring as _run_quick_scoring
    _scoring_core_loaded = True
    print("[api] scoring_core: 読み込み完了")
except Exception as _sc_import_err:
    print(f"[api] scoring_core: 未ロード → sys_scoreは中央値フォールバック ({_sc_import_err})")

_indicator_analysis_loaded = False
_get_indicator_analysis_for_advice = None
try:
    from indicators import get_indicator_analysis_for_advice as _get_indicator_analysis_for_advice
    _indicator_analysis_loaded = True
    print("[api] indicators: 読み込み完了")
except Exception as _indicator_import_err:
    print(f"[api] indicators: 未ロード ({_indicator_import_err})")

# ── advisor: 軍師AI（ルールベース参謀）───────────────────────────────
_advisor_loaded = False
try:
    from advisor_strategy import build_gemini_strategy_advice, build_strategy_advice
    _advisor_loaded = True
    print("[api] advisor_strategy: 読み込み完了")
except Exception as _advisor_import_err:
    build_gemini_strategy_advice = None
    build_strategy_advice = None
    print(f"[api] advisor_strategy: 未ロード ({_advisor_import_err})")

_chat_loaded = False
try:
    from chat_assistant import build_chat_reply
    _chat_loaded = True
    print("[api] chat_assistant: 読み込み完了")
except Exception as _chat_import_err:
    build_chat_reply = None
    print(f"[api] chat_assistant: 未ロード ({_chat_import_err})")

try:
    from obsidian_bridge import append_case_log as _append_case_log
    print("[api] obsidian_bridge: 読み込み完了")
except Exception as _obs_import_err:
    _append_case_log = None
    print(f"[api] obsidian_bridge: 未ロード ({_obs_import_err})")

try:
    import sys as _sys
    _sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from data_cases import CustomJSONEncoder as _CJE, update_case as _update_case, delete_case as _delete_case
    print("[api] data_cases: 読み込み完了")
except Exception as _dc_err:
    _CJE = None
    _update_case = None
    _delete_case = None
    print(f"[api] data_cases: 未ロード ({_dc_err})")

_DB_PATH = str(pathlib.Path(__file__).parent.parent / "data" / "lease_data.db")


def _init_wal() -> None:
    with _sqlite3.connect(_DB_PATH, timeout=10) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")


try:
    _init_wal()
    print("[api] lease_data.db WAL初期化完了")
except Exception as _wal_err:
    print(f"[api] WAL初期化失敗（無視）: {_wal_err}")


def _insert_flask_case(response_payload: dict, input_data: dict, latency_ms: int) -> None:
    try:
        now = _datetime.datetime.utcnow()
        case_id = now.strftime("%Y%m%d%H%M%S%f") + "_" + uuid.uuid4().hex[:8]
        data_json = {
            **input_data,
            "score":            response_payload.get("score"),
            "probability":      response_payload.get("probability"),
            "judgment":         response_payload.get("judgment"),
            "recommended_rate": response_payload.get("recommended_rate"),
            "spread_pred":      response_payload.get("spread_pred"),
            "quantum_risk":     response_payload.get("quantum_risk"),
            "source":           "flask_api",
            "latency_ms":       latency_ms,
        }
        user_eq = None
        try:
            na = float(input_data.get("net_assets") or 0)
            ta = float(input_data.get("total_assets") or 1)
            if ta > 0:
                user_eq = round(na / ta, 4)
        except Exception:
            pass
        json_str = json.dumps(data_json, ensure_ascii=False, cls=_CJE)
        with _sqlite3.connect(_DB_PATH, timeout=10) as conn:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                """INSERT OR IGNORE INTO past_cases
                   (id, timestamp, industry_sub, score, user_eq,
                    final_status, data, sales_dept)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    case_id, now.isoformat(),
                    str(input_data.get("industry") or ""),
                    response_payload.get("score"),
                    user_eq,
                    "スコアリングのみ",
                    json_str,
                    str(input_data.get("sales_dept") or "未設定"),
                )
            )
    except Exception as _e:
        print(f"[api] past_cases INSERT失敗（無視）: {_e}")


_DEFAULT_IND        = "R サービス業(他に分類されないもの)"
_DEFAULT_CONTRACT_T = "一般"
_DEFAULT_DEAL_SRC   = "銀行紹介"
_DEFAULT_SALES_DEPT = "足利営業部"


@lru_cache(maxsize=1)
def _load_web_industry_benchmarks() -> dict:
    path = os.path.join(_PROJECT_ROOT, "web_industry_benchmarks.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_local_industry_benchmarks() -> dict:
    path = os.path.join(_PROJECT_ROOT, "data", "industry_benchmarks.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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


def _clamp_float(value, low=0.0, high=100.0):
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return low


def _build_mobile_chart_payload(response_payload: dict, sc_result: dict | None, q_risk_result: dict, stealth_result: dict) -> dict:
    """mobile_app/index.html 用の軽量チャートデータを組み立てる。"""
    streamlit = response_payload.get("streamlit") or {}
    score = _clamp_float(response_payload.get("score") or streamlit.get("score") or 0)
    probability = _clamp_float((response_payload.get("probability") or 0) * 100)
    score_borrower = _clamp_float(streamlit.get("score_borrower") or 0)
    q_risk_score = _clamp_float((q_risk_result or {}).get("score") or 0)
    q_risk_readiness = _clamp_float(100.0 - q_risk_score * 2.0)
    competitor_score = _clamp_float((stealth_result or {}).get("score") or 0)
    competitor_readiness = _clamp_float(100.0 - competitor_score * 2.0)
    base_rate = float(response_payload.get("base_rate") or 0)
    recommended_rate = float(response_payload.get("recommended_rate") or base_rate or 0)
    rate_alignment = _clamp_float(100.0 - abs(recommended_rate - base_rate) * 20.0)
    approval_margin = _clamp_float(50.0 + (score - 71.0) * 2.0)

    radar = [
        {"label": "総合スコア", "value": score, "max": 100, "note": "最終判定"},
        {"label": "成約確率", "value": probability, "max": 100, "note": "予測確率"},
        {"label": "借手スコア", "value": score_borrower, "max": 100, "note": "Streamlit値"},
        {"label": "金利妥当性", "value": rate_alignment, "max": 100, "note": "ベースレートとの差"},
        {"label": "Q_risk健全度", "value": q_risk_readiness, "max": 100, "note": "矛盾が少ないほど高い"},
        {"label": "競合耐性", "value": competitor_readiness, "max": 100, "note": "圧力が低いほど高い"},
    ]

    factors = []
    if sc_result:
        for row in (sc_result.get("score_contributions") or []):
            feature = str(row.get("feature") or "")
            if feature == "intercept":
                continue
            value = row.get("contribution")
            try:
                value = float(value)
            except Exception:
                continue
            factors.append({
                "label": row.get("label_ja") or feature,
                "value": round(value, 2),
            })
    factors = factors[:6]

    return {
        "radar": radar,
        "factors": factors,
        "summary": {
            "approval_margin": approval_margin,
            "q_risk": q_risk_score,
            "competitor_pressure": competitor_score,
        },
    }


def _build_chat_analysis_graphs(message: str, input_data: dict | None, score_result: dict | None) -> dict:
    """AIチャット用の計算結果・業種平均との差分を組み立てる。"""
    data = input_data if isinstance(input_data, dict) else {}
    score_result = score_result if isinstance(score_result, dict) else {}

    def _num(key: str, default: float = 0.0) -> float:
      try:
        return _normalize_numeric(data.get(key, default), default)
      except Exception:
        return default

    industry_major = normalize_industry_major(
        str(data.get("industry_major") or data.get("industry") or data.get("industry_detail") or _DEFAULT_IND)
    )
    industry_sub = normalize_industry_sub(
        str(data.get("industry_sub") or data.get("industry_detail") or ""),
        industry_major,
    )
    benchmark = {}
    if industry_sub:
        local_benchmarks = _load_local_industry_benchmarks()
        web_benchmarks = _load_web_industry_benchmarks()
        benchmark.update(local_benchmarks.get(industry_sub, {}) or {})
        benchmark.update(web_benchmarks.get(industry_sub, {}) or {})

    gp = _num("gross_profit")
    op = _num("op_profit")
    ep = _num("ord_profit")
    ni = _num("net_income")
    dep = _num("dep_expense")
    depr = _num("depreciation")
    ns = _num("nenshu")
    bk = _num("bank_credit")
    lc = _num("lease_credit")
    mach = _num("machines")
    oa = _num("other_assets")
    net_assets = _num("net_assets")
    total_assets = _num("total_assets")
    rent = _num("rent")
    rent_expense = _num("rent_expense")
    acquisition_cost = _num("acquisition_cost")
    contracts = _num("contracts")
    lease_term = _num("lease_term")
    proposed_rate = _num("proposed_rate")
    competitor_rate = _num("competitor_rate")
    lease_asset_score = _num("lease_asset_score")
    q_weighted = _num("q_weighted")

    fixed_assets = mach + oa
    debt_total = bk + lc

    computed = []
    if ns > 0:
        computed.extend([
            {"label": "売上高総利益率", "value": round(gp / ns * 100, 2), "unit": "%", "formula": "粗利÷売上高", "category": "収益性"},
            {"label": "営業利益率", "value": round(op / ns * 100, 2), "unit": "%", "formula": "営業利益÷売上高", "category": "収益性"},
            {"label": "経常利益率", "value": round(ep / ns * 100, 2), "unit": "%", "formula": "経常利益÷売上高", "category": "収益性"},
            {"label": "当期純利益率", "value": round(ni / ns * 100, 2), "unit": "%", "formula": "当期純利益÷売上高", "category": "収益性"},
            {"label": "減価償却費率", "value": round(dep / ns * 100, 2), "unit": "%", "formula": "減価償却（経費）÷売上高", "category": "収益性"},
            {"label": "賃借料率", "value": round(rent_expense / ns * 100, 2), "unit": "%", "formula": "賃借料（経費）÷売上高", "category": "収益性"},
            {"label": "銀行借入率", "value": round(bk / ns * 100, 2), "unit": "%", "formula": "銀行借入残高÷売上高", "category": "負債"},
            {"label": "リース与信率", "value": round(lc / ns * 100, 2), "unit": "%", "formula": "リース与信残高÷売上高", "category": "負債"},
            {"label": "機械設備率", "value": round(mach / ns * 100, 2), "unit": "%", "formula": "機械設備残高÷売上高", "category": "資産"},
            {"label": "その他資産率", "value": round(oa / ns * 100, 2), "unit": "%", "formula": "その他資産÷売上高", "category": "資産"},
            {"label": "取得価額率", "value": round(acquisition_cost / ns * 100, 2), "unit": "%", "formula": "リース取得価額÷売上高", "category": "資産"},
            {"label": "現在リース件数", "value": round(contracts, 0), "unit": "件", "formula": "入力値", "category": "取引"},
        ])
    if total_assets > 0:
        computed.extend([
            {"label": "自己資本比率", "value": round(net_assets / total_assets * 100, 2), "unit": "%", "formula": "自己資本÷総資本", "category": "資本"},
            {"label": "ROA", "value": round(ni / total_assets * 100, 2), "unit": "%", "formula": "当期純利益÷総資本", "category": "資本"},
            {"label": "総資産回転率", "value": round(ns / total_assets, 2), "unit": "回", "formula": "売上高÷総資本", "category": "効率"},
            {"label": "固定資産比率", "value": round(fixed_assets / total_assets * 100, 2), "unit": "%", "formula": "固定資産÷総資本", "category": "資産"},
            {"label": "流動資産比率", "value": round((total_assets - fixed_assets) / total_assets * 100, 2), "unit": "%", "formula": "流動資産÷総資本", "category": "資産"},
            {"label": "借入依存度", "value": round(debt_total / total_assets * 100, 2), "unit": "%", "formula": "借入金等÷総資本", "category": "負債"},
        ])
    if net_assets > 0:
        computed.extend([
            {"label": "ROE", "value": round(ni / net_assets * 100, 2), "unit": "%", "formula": "当期純利益÷自己資本", "category": "資本"},
            {"label": "負債比率", "value": round((total_assets - net_assets) / net_assets * 100, 2), "unit": "%", "formula": "負債÷自己資本", "category": "負債"},
        ])

    ratio_specs = [
        ("売上高総利益率", gp / ns * 100 if ns > 0 else None, "gross_margin", "%", True, "収益性"),
        ("営業利益率", op / ns * 100 if ns > 0 else None, "op_margin", "%", True, "収益性"),
        ("経常利益率", ep / ns * 100 if ns > 0 else None, "ord_margin", "%", True, "収益性"),
        ("当期純利益率", ni / ns * 100 if ns > 0 else None, "net_margin", "%", True, "収益性"),
        ("減価償却費率", dep / ns * 100 if ns > 0 else None, "dep_ratio", "%", False, "収益性"),
        ("自己資本比率", net_assets / total_assets * 100 if total_assets > 0 else None, "equity_ratio", "%", True, "資本"),
        ("ROA", ni / total_assets * 100 if total_assets > 0 else None, "roa", "%", True, "資本"),
        ("ROE", ni / net_assets * 100 if net_assets > 0 else None, "roe", "%", True, "資本"),
        ("総資産回転率", ns / total_assets if total_assets > 0 else None, "asset_turnover", "回", True, "効率"),
    ]

    ratios = []
    for label, value, bench_key, unit, higher_better, category in ratio_specs:
        if value is None:
            continue
        bench_value = benchmark.get(bench_key)
        if bench_value is None:
            continue
        diff = float(value) - float(bench_value)
        favorable = diff if higher_better else -diff
        ratios.append({
            "label": label,
            "value": round(float(value), 2),
            "benchmark": round(float(bench_value), 2),
            "diff": round(diff, 2),
            "favorable": round(favorable, 2),
            "unit": unit,
            "category": category,
            "higher_better": higher_better,
        })

    computed.extend([
        {"label": "リース期間", "value": round(lease_term, 0), "unit": "ヶ月", "formula": "入力値", "category": "取引"},
        {"label": "提案金利", "value": round(proposed_rate, 2), "unit": "%", "formula": "入力値", "category": "金利"},
        {"label": "競合金利", "value": round(competitor_rate, 2), "unit": "%", "formula": "入力値", "category": "競合"},
        {"label": "物件スコア", "value": round(lease_asset_score, 0), "unit": "点", "formula": "入力値", "category": "資産"},
        {"label": "定性加重平均", "value": round(q_weighted, 0), "unit": "点", "formula": "入力値", "category": "定性"},
        {"label": "基準金利", "value": round(float(score_result.get("base_rate") or 0), 2), "unit": "%", "formula": "API算出", "category": "金利"},
        {"label": "推奨金利", "value": round(float(score_result.get("recommended_rate") or 0), 2), "unit": "%", "formula": "API算出", "category": "金利"},
        {"label": "スプレッド予測", "value": round(float(score_result.get("spread_pred") or 0), 2), "unit": "%", "formula": "推奨金利-基準金利", "category": "金利"},
        {"label": "AURION Q_risk", "value": round(float(((score_result.get("aurion") or {}).get("q_risk") or {}).get("score", score_result.get("quantum_risk") or 0)), 2), "unit": "点", "formula": "整合性チェック", "category": "警戒"},
        {"label": "競合圧力", "value": round(float(((score_result.get("aurion") or {}).get("competitor_pressure") or {}).get("score", 0)), 2), "unit": "点", "formula": "競合推定", "category": "警戒"},
        {"label": "成約確率", "value": round(float(score_result.get("probability") or 0) * 100, 2), "unit": "%", "formula": "モデル出力", "category": "予測"},
        {"label": "総合スコア", "value": round(float(score_result.get("score") or 0), 2), "unit": "点", "formula": "モデル出力", "category": "予測"},
    ])

    message_flags = []
    msg = (message or "").strip()
    if msg:
        message_flags.append({"label": "チャット入力あり", "value": 1, "unit": "", "formula": "入力検知", "category": "会話"})
        if any(token in msg for token in ("グラフ", "図", "比較", "差", "業種")):
            message_flags.append({"label": "グラフ要望", "value": 1, "unit": "", "formula": "文面解析", "category": "会話"})

    computed.extend(message_flags)

    return {
        "industry": {
            "major": industry_major,
            "sub": industry_sub,
            "benchmark_found": bool(ratios),
        },
        "ratios": ratios,
        "computed": computed,
        "summary": {
            "ratio_count": len(ratios),
            "computed_count": len(computed),
        },
    }


def _normalize_case_status(status: str) -> str:
    if status in ("検収完了", "検収"):
        return "成約"
    if status in ("成約", "失注"):
        return status
    return "その他"


def _build_visual_insights_payload(current_case: dict | None = None, limit_cases: int = 300) -> dict:
    """visual_insights.py 相当の集計を mobile 用に軽量化して返す。"""
    cases = load_past_cases()[-limit_cases:]
    rows = []
    for c in cases:
        status = _normalize_case_status(str(c.get("final_status", "未登録")))
        if status not in ("成約", "失注"):
            continue
        score = float(c.get("score") or (c.get("result") or {}).get("score") or 0)
        if score <= 0:
            continue
        final_rate = float(c.get("final_rate") or 0)
        base_rate = float(c.get("base_rate_at_time") or 0)
        spread = (final_rate - base_rate) if base_rate > 0 else float(c.get("winning_spread") or 0)
        inputs = c.get("inputs") or {}
        rows.append({
            "score": score,
            "spread": spread,
            "acquisition_cost": float(inputs.get("acquisition_cost") or 0),
            "status": status,
            "industry_major": c.get("industry_major") or "不明",
            "industry_sub": c.get("industry_sub") or "不明",
            "sales_dept": c.get("sales_dept") or "未設定",
            "competitor": c.get("competitor_name") or "",
        })

    bubble = {
        "points": rows[-80:],
        "current": {
            "score": float((current_case or {}).get("score") or 0),
            "spread": float((current_case or {}).get("spread_pred") or 0),
        },
    }

    return {
        "bubble": bubble,
        "meta": {
            "case_count": len(rows),
            "current_case_score": float((current_case or {}).get("score") or 0),
        },
    }


def _build_indicator_analysis_payload(data: dict, industry_major: str, industry_sub: str) -> dict:
    if _get_indicator_analysis_for_advice is None:
        return {"summary": "", "detail": "", "indicators": [], "text": ""}
    try:
        last_result = {
            "industry_major": industry_major,
            "industry_sub": industry_sub,
            "financials": {
                "nenshu": _normalize_numeric(data.get("nenshu", 0)),
                "gross_profit": _normalize_numeric(data.get("gross_profit", 0)),
                "op_profit": _normalize_numeric(data.get("op_profit", 0)),
                "ord_profit": _normalize_numeric(data.get("ord_profit", 0)),
                "net_income": _normalize_numeric(data.get("net_income", 0)),
                "dep_expense": _normalize_numeric(data.get("dep_expense", 0)),
                "depreciation": _normalize_numeric(data.get("depreciation", 0)),
                "machines": _normalize_numeric(data.get("machines", 0)),
                "other_assets": _normalize_numeric(data.get("other_assets", 0)),
                "bank_credit": _normalize_numeric(data.get("bank_credit", 0)),
                "lease_credit": _normalize_numeric(data.get("lease_credit", 0)),
                "net_assets": _normalize_numeric(data.get("net_assets", 0)),
                "assets": _normalize_numeric(data.get("total_assets", 0)),
            },
        }
        summary, detail, text = _get_indicator_analysis_for_advice(last_result)
        indicators = []
        if text:
            for line in text.splitlines():
                if ":" not in line:
                    continue
                name, rest = line.split(":", 1)
                indicators.append({"name": name.strip("- "), "text": rest.strip()})
        return {
            "summary": summary,
            "detail": detail,
            "indicators": indicators,
            "text": text,
        }
    except Exception as exc:
        return {"summary": "", "detail": "", "indicators": [], "text": "", "error": str(exc)}

    return {
        "bubble": bubble,
        "meta": {
            "case_count": len(rows),
            "current_case_score": float((current_case or {}).get("score") or 0),
        },
    }


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


_STATIC_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".json", ".js", ".webmanifest"}


def _grade_label(raw_grade) -> str:
    if isinstance(raw_grade, str):
        if "4-6" in raw_grade:
            return "4-6"
        if "要注意" in raw_grade:
            return "要注意先"
        if "無格付" in raw_grade:
            return "無格付"
        if "1-3" in raw_grade:
            return "1-3"
    try:
        grade_int = int(raw_grade)
    except (TypeError, ValueError):
        grade_int = 1
    return {1: "1-3", 2: "4-6", 3: "要注意先", 4: "無格付"}.get(grade_int, "1-3")


def _customer_type_label(customer_type_flag: int) -> str:
    return "既存先" if customer_type_flag == 1 else "新規先"


def _build_streamlit_scoring_inputs(
    data: dict,
    *,
    ns: float,
    op: float,
    ep: float,
    ni: float,
    dep: float,
    gp: float,
    bk: float,
    lc: float,
    mach: float,
    oa: float,
    rent: float,
    depr: float,
    rexp: float,
    acq: float,
    contracts: float,
    grade: int,
    customer_type: int,
    industry: str,
    main_bank_text: str,
    competitor_text: str,
    comp_rate: float,
    contract_type: str,
    deal_source: str,
    sales_dept: str,
    lease_asset_score: float,
) -> dict:
    """mobile API の百万円入力を Streamlit/scoring_core の千円入力へ揃える。"""
    industry_major = str(data.get("industry_major") or industry or _DEFAULT_IND)
    industry_sub = str(data.get("industry_sub") or data.get("industry_detail") or "06 総合工事業")
    return {
        "nenshu": ns * 1000,
        "op_profit": op * 1000,
        "ord_profit": ep * 1000,
        "net_income": ni * 1000,
        "dep_expense": dep * 1000,
        "gross_profit": gp * 1000,
        "bank_credit": bk * 1000,
        "lease_credit": lc * 1000,
        "machines": mach * 1000,
        "other_assets": oa * 1000,
        "rent": rent * 1000,
        "depreciation": depr * 1000,
        "rent_expense": rexp * 1000,
        "acquisition_cost": acq * 1000,
        "net_assets": _normalize_numeric(data.get("net_assets", 0)) * 1000,
        "total_assets": _normalize_numeric(data.get("total_assets", 0)) * 1000,
        "contracts": contracts,
        "grade": _grade_label(data.get("grade", grade)),
        "customer_type": _customer_type_label(customer_type),
        "industry_major": industry_major,
        "industry_sub": industry_sub,
        "main_bank": main_bank_text,
        "competitor": competitor_text,
        "competitor_rate": comp_rate,
        "contract_type": contract_type,
        "deal_source": deal_source,
        "sales_dept": sales_dept,
        "asset_score": lease_asset_score,
        "lease_asset_score": lease_asset_score,
        "intuition_score": data.get("intuition_score", data.get("q_intuition_score", 0)),
        "deal_occurrence": data.get("deal_occurrence", ""),
        "num_competitors": data.get("num_competitors", 1 if competitor_text == "競合あり" else 0),
        "asset_type": data.get("asset_type", ""),
    }


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
    _t_start = time.time()
    data = request.get_json(force=True, silent=True) or {}
    try:
        # ── UI から受け取る主要フィールド（百万円単位）──────────────
        gp   = _normalize_numeric(data.get("gross_profit",    0))
        op   = _normalize_numeric(data.get("op_profit",       0))
        ep   = _normalize_numeric(data.get("ord_profit",      0))
        ni   = _normalize_numeric(data.get("net_income",      0))
        dep  = _normalize_numeric(data.get("dep_expense",     0))
        ns   = _normalize_numeric(data.get("nenshu",          0))
        acq  = _normalize_numeric(data.get("acquisition_cost",0))
        lc   = _normalize_numeric(data.get("lease_credit",    0))
        bk   = _normalize_numeric(data.get("bank_credit",     0))
        mach = _normalize_numeric(data.get("machines",        0))
        lt   = _normalize_numeric(data.get("lease_term",      60), 60)
        contracts = _normalize_numeric(data.get("contracts",  0))

        industry      = str(data.get("industry") or data.get("industry_major") or _DEFAULT_IND)
        customer_type = 1 if data.get("customer_type") == "既存先" else 0
        main_bank_text = "メイン先" if data.get("main_bank") == "メイン先" else "非メイン先"
        competitor_text = "競合あり" if data.get("competitor") == "競合あり" else "競合なし"
        main_bank     = 1 if main_bank_text == "メイン先" else 0
        competitor    = 1 if competitor_text == "競合あり" else 0
        comp_rate     = _normalize_numeric(data.get("competitor_rate", 0))
        contract_type = str(data.get("contract_type", _DEFAULT_CONTRACT_T))
        deal_source   = str(data.get("deal_source",   _DEFAULT_DEAL_SRC))
        sales_dept    = str(data.get("sales_dept",    _DEFAULT_SALES_DEPT))
        grade         = _normalize_int(data.get("grade", 4), 4)

    except (TypeError, ValueError) as e:
        return jsonify({"error": f"数値変換エラー: {e}"}), 400

    # P1-002: リースルールチェック用フィールド（全てオプション）
    asset_type_str = str(data.get("asset_type", ""))
    is_re_lease    = bool(data.get("is_re_lease", False))
    insurance_app  = str(data.get("insurance_applicable", "不明"))
    re_lease_ins   = str(data.get("re_lease_insurance", "不明"))

    # ── 派生比率 ──────────────────────────────────────────────────
    depr = _normalize_numeric(data.get("depreciation", 0))
    oa   = _normalize_numeric(data.get("other_assets", 0))
    rent = _normalize_numeric(data.get("rent", 0))
    rexp = _normalize_numeric(data.get("rent_expense", 0))

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

    # ── リース期間別基準金利（feature vector にも使用）───────────────
    base_rate_val, _brt_col = _get_period_rate(lt)

    # ── 定性スコア（0-4 スケール、未入力はモデル中央値）──────────────
    q_history   = _normalize_numeric(data.get("q_history",   _medians.get("q_history",   0.0)))
    q_stability = _normalize_numeric(data.get("q_stability", _medians.get("q_stability", 0.0)))
    q_repayment = _normalize_numeric(data.get("q_repayment", _medians.get("q_repayment", 0.0)))
    q_future    = _normalize_numeric(data.get("q_future",    _medians.get("q_future",    0.0)))
    q_equip     = _normalize_numeric(data.get("q_equip",     _medians.get("q_equip",     0.0)))
    q_mainbk    = _normalize_numeric(data.get("q_mainbk",    _medians.get("q_mainbk",    0.0)))
    q_weighted  = _normalize_numeric(data.get("q_weighted",  _medians.get("q_weighted",  0.0)))
    las         = _normalize_numeric(data.get("lease_asset_score", 100), 100)

    # ── sys_ 特徴量を scoring_core で計算（Streamlit と同等ロジック）──
    _sc_result = None
    if _scoring_core_loaded and _run_quick_scoring is not None:
        try:
            _sc_inputs = _build_streamlit_scoring_inputs(
                data,
                ns=ns, op=op, ep=ep, ni=ni, dep=dep, gp=gp, bk=bk, lc=lc,
                mach=mach, oa=oa, rent=rent, depr=depr, rexp=rexp, acq=acq,
                contracts=contracts, grade=grade, customer_type=customer_type,
                industry=industry, main_bank_text=main_bank_text,
                competitor_text=competitor_text, comp_rate=comp_rate,
                contract_type=contract_type, deal_source=deal_source,
                sales_dept=sales_dept, lease_asset_score=las,
            )
            _sc_result = _run_quick_scoring(_sc_inputs)
        except Exception as _sc_e:
            print(f"[api] scoring_core計算エラー: {_sc_e}")

    sys_score = float(_sc_result["score_borrower"] if _sc_result else _medians.get("sys_score", 57.4))
    sys_dscr  = float(_sc_result["dscr_approx"]    if _sc_result else _medians.get("sys_dscr",   0.63))
    sys_op_m  = float(_sc_result["user_op_margin"] if _sc_result else _medians.get("sys_op_margin", 1.83))
    sys_icr   = float(_sc_result["interest_coverage"] if _sc_result else _medians.get("sys_icr", 10.0))
    sys_appr  = float(_medians.get("sys_approval",  71.0))
    sys_ind   = float(_medians.get("sys_ind_score", 95.9))
    sys_bench = float(_medians.get("sys_bench",     45.5))

    common = dict(
        gp=gp, op=op, ep=ep, ni=ni, dep=dep, depr=depr, ns=ns,
        mach=mach, oa=oa, rent=rent, rexp=rexp, bk=bk, lc=lc, acq=acq, lt=lt,
        contracts=contracts, gpm=gpm, ord_margin=ord_margin, net_margin=net_margin,
        dep_ratio=dep_ratio, bank_to_ns=bank_to_ns, lease_to_ns=lease_to_ns,
        mach_to_ns=mach_to_ns, acq_to_ns=acq_to_ns, op_margin=op_margin,
        dep_to_loan=dep_to_loan, ind_code=ind_code, customer_type=customer_type,
        main_bank=main_bank, competitor=competitor, comp_rate=comp_rate, grade=grade,
        ct_code=ct_code, ds_code=ds_code, sd_code=sd_code, base_rate=base_rate_val,
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

    # 格付けによる金利上乗せ: 1-3先±0 / 4-6先+0.2% / 要注意先+0.5% / 無格付+0.3%
    _grade_premium = {1: 0.0, 2: 0.2, 3: 0.5, 4: 0.3}.get(grade, 0.0)
    recommended_rate += _grade_premium

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
            "base_rate": base_rate_val,
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
    rf_score = int(round(proba * 100))

    # 最終スコアは scoring_core (Streamlit と同一ルート) を使用
    if _sc_result is not None:
        score = int(round(float(_sc_result["score"])))
    else:
        score = rf_score  # フォールバック: scoring_core 未ロード時
    if score >= 71:
        judgment = "承認"
    elif score >= 50:
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
        if _detect_q_risk is None:
            raise RuntimeError("aurion.q_risk is not loaded")
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

    response_payload = {
        "score":             score,
        "probability":       round(proba, 4),
        "rf_score":          rf_score,
        "judgment":          judgment,
        "sys_score_b":       round(sys_score_b, 1),
        "model_pipeline":    model_pipeline,
        "spread_pred":       round(spread_pred, 2),
        "spread_proposed":   round(stealth_spread, 2) if _proposed_rate_raw is not None else None,
        "base_rate":         round(base_rate_val, 2),
        "recommended_rate":  round(recommended_rate, 2),
        "quantum_risk":      _sc_result.get("quantum_risk") if _sc_result else None,
        "score_source":      "scoring_core" if _sc_result else "rf_fallback",
        "rate_range": {
            "low":  round(recommended_rate - 0.3, 2),
            "high": round(recommended_rate + 0.3, 2),
        },
        "warnings":          warnings,
        "rule_check_status": rule_check_status,
        "streamlit": {
            "score": _sc_result.get("score") if _sc_result else None,
            "score_borrower": _sc_result.get("score_borrower") if _sc_result else None,
            "score_base": _sc_result.get("score_base") if _sc_result else None,
            "hantei": _sc_result.get("hantei") if _sc_result else None,
            "approval_line": _sc_result.get("approval_line") if _sc_result else None,
            "quantum_risk": _sc_result.get("quantum_risk") if _sc_result else None,
            "credit_risk_group_score": _sc_result.get("credit_risk_group_score") if _sc_result else None,
            "credit_risk_group_level": _sc_result.get("credit_risk_group_level") if _sc_result else None,
            "score_source": "scoring_core" if _sc_result else "rf_fallback",
            "score_contributions": ((_sc_result.get("score_contributions") or [])[:8] if _sc_result else []),
        },
        "aurion": {
            "q_risk": q_risk_result,
            "quantum_risk": _sc_result.get("quantum_risk") if _sc_result else None,
            "competitor_pressure": stealth_result,
        },
    }
    response_payload["charts"] = _build_mobile_chart_payload(
        response_payload=response_payload,
        sc_result=_sc_result,
        q_risk_result=q_risk_result,
        stealth_result=stealth_result,
    )
    response_payload["insights"] = _build_visual_insights_payload(
        current_case={
            "score": response_payload.get("score"),
            "spread_pred": response_payload.get("spread_pred"),
        }
    )
    response_payload["indicator_analysis"] = _build_indicator_analysis_payload(
        data=data,
        industry_major=str(data.get("industry_major") or industry or _DEFAULT_IND),
        industry_sub=str(data.get("industry_sub") or data.get("industry_detail") or "06 総合工事業"),
    )
    if build_strategy_advice is not None:
        try:
            _advisor_engine = str(data.get("advisor_engine") or os.environ.get("ADVISOR_ENGINE", "gemini")).lower()
            _advisor_builder = (
                build_gemini_strategy_advice
                if _advisor_engine == "gemini" and build_gemini_strategy_advice is not None
                else build_strategy_advice
            )
            _advisor_mode = str(data.get("advisor_mode") or "審査軍師")
            _humor_style = str(data.get("humor_style") or "standard")
            if _advisor_builder is build_gemini_strategy_advice:
                response_payload["advisor"] = _advisor_builder(
                    score_result=response_payload,
                    case=data,
                    mode=_advisor_mode,
                    timeout_seconds=_normalize_numeric(data.get("advisor_timeout", 20), 20),
                    humor_style=_humor_style,
                )
            else:
                response_payload["advisor"] = _advisor_builder(
                    score_result=response_payload,
                    case=data,
                    mode=_advisor_mode,
                    humor_style=_humor_style,
                )
        except Exception as _advisor_e:
            response_payload["advisor"] = {
                "mode": "審査軍師",
                "summary": "軍師AIコメントを生成できませんでした。",
                "strategy": "スコア結果と警戒信号を個別に確認してください。",
                "risk_points": [str(_advisor_e)],
                "recommended_conditions": [],
                "sales_talk": [],
                "decision": {"stance": "unavailable", "confidence": 0.0},
                "metrics": {},
                "disclaimer": "軍師AIは判定を上書きしません。",
            }
    if _append_case_log is not None:
        _snap = dict(response_payload)
        _case_snap = dict(data)
        threading.Thread(
            target=_append_case_log,
            args=(_snap, _case_snap),
            daemon=True,
        ).start()

    _latency = int((time.time() - _t_start) * 1000)
    threading.Thread(
        target=_insert_flask_case,
        args=(dict(response_payload), dict(data), _latency),
        daemon=True,
    ).start()

    return jsonify(response_payload)


_FASTAPI_BASE = os.environ.get("FASTAPI_URL", "http://localhost:8000")


def _proxy(method: str, path: str, *, params=None, body=None):
    """FastAPI への軽量プロキシヘルパー (urllib 標準ライブラリのみ使用)"""
    import urllib.request
    import urllib.parse
    url = f"{_FASTAPI_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()), resp.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}"), e.code


@app.get("/cases")
def list_cases():
    limit = min(int(request.args.get("limit", 30)), 200)
    offset = int(request.args.get("offset", 0))
    sort = request.args.get("sort", "desc")
    body, status = _proxy("GET", "/api/cases", params={"limit": limit, "offset": offset, "sort": sort})
    return jsonify(body), status


@app.patch("/cases/<case_id>/result")
def update_case_result(case_id: str):
    updates = request.get_json(force=True, silent=True) or {}
    # final_status → FastAPI の CaseResultPatch フィールドに変換
    payload = {}
    if "final_status" in updates:
        payload["final_status"] = updates["final_status"]
    if "competitor_rate" in updates:
        payload["competitor_rate"] = updates["competitor_rate"]
    if "loss_reason" in updates:
        payload["loss_reason"] = updates["loss_reason"]
    if "final_result_date" in updates:
        payload["final_result_date"] = updates["final_result_date"]
    body, status = _proxy("PATCH", f"/api/cases/{case_id}/result", body=payload)
    return jsonify(body), status


@app.delete("/cases/<case_id>")
def delete_case_record(case_id: str):
    body, status = _proxy("DELETE", f"/api/cases/{case_id}")
    return jsonify(body), status


@app.post("/advisor/strategy")
def advisor_strategy():
    if build_strategy_advice is None:
        return jsonify({"error": "advisor_strategy is not loaded"}), 503
    data = request.get_json(force=True, silent=True) or {}
    score_result = data.get("score_result") or data.get("result") or data
    case = data.get("case") or data.get("inputs") or {}
    mode = str(data.get("mode") or "審査軍師")
    engine = str(data.get("engine") or os.environ.get("ADVISOR_ENGINE", "gemini")).lower()
    humor_style = str(data.get("humor_style") or "standard")
    try:
        if isinstance(score_result, dict) and "indicator_analysis" not in score_result:
            score_result = dict(score_result)
            score_result["indicator_analysis"] = _build_indicator_analysis_payload(
                data=case if isinstance(case, dict) else {},
                industry_major=str(case.get("industry_major") or case.get("industry") or _DEFAULT_IND),
                industry_sub=str(case.get("industry_sub") or case.get("industry_detail") or "06 総合工事業"),
            )
        if engine == "gemini" and build_gemini_strategy_advice is not None:
            timeout_seconds = _normalize_numeric(data.get("timeout_seconds", 20), 20)
            return jsonify(build_gemini_strategy_advice(
                score_result=score_result,
                case=case,
                mode=mode,
                timeout_seconds=timeout_seconds,
                humor_style=humor_style,
            ))
        return jsonify(build_strategy_advice(score_result=score_result, case=case, mode=mode, humor_style=humor_style))
    except Exception as e:
        return jsonify({"error": f"軍師AI生成エラー: {e}"}), 400


@app.post("/chat")
def chat():
    if build_chat_reply is None:
        return jsonify({"error": "chat_assistant is not loaded"}), 503
    data = request.get_json(force=True, silent=True) or {}
    message = str(data.get("message") or "")
    history = data.get("history") or []
    if not isinstance(history, list):
        history = []
    score_result = data.get("score_result") or data.get("result") or {}
    input_data = data.get("input_data") or {}
    use_obsidian = bool(data.get("use_obsidian", True))
    use_web = bool(data.get("use_web", True))
    humor_style = str(data.get("humor_style") or "standard")
    try:
        timeout_seconds = _normalize_numeric(data.get("timeout_seconds", 30), 30)
        payload = build_chat_reply(
            message=message,
            history=history,
            score_result=score_result,
            use_obsidian=use_obsidian,
            use_web=use_web,
            timeout_seconds=timeout_seconds,
            humor_style=humor_style,
        )
        payload["analysis_graphs"] = _build_chat_analysis_graphs(message, input_data, score_result)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": f"AIチャット生成エラー: {e}"}), 400


@app.post("/api/gunshi/stream")
def gunshi_stream_proxy():
    """FastAPI (port 8000) の SSEエンドポイントにプロキシする。"""
    import requests as _req
    fastapi_url = "http://localhost:8000/api/gunshi/stream"
    try:
        upstream = _req.post(
            fastapi_url,
            json=request.get_json(force=True),
            stream=True,
            timeout=90,
        )
        def generate():
            for chunk in upstream.iter_content(chunk_size=None):
                if chunk:
                    yield chunk
        return Response(
            stream_with_context(generate()),
            content_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 503


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
        "scoring_core_loaded": _scoring_core_loaded,
        "advisor_strategy_loaded": _advisor_loaded,
        "chat_assistant_loaded": _chat_loaded,
        "advisor_engine_default": os.environ.get("ADVISOR_ENGINE", "gemini"),
        "quantum_model_exists": os.path.exists(
            os.path.join(_PROJECT_ROOT, "data", "quantum_model.joblib")
        ),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
