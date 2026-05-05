from __future__ import annotations

import datetime as _dt
import itertools
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_cases import load_all_cases

_DATA_DIR = "/Users/kobayashiisaoryou/clawd/tune_lease_55/data"
MODEL_PATH = os.path.join(_DATA_DIR, "tunnel_optimizer.joblib")

MIN_CASES = 1500
VALID_STATUSES = ("成約", "失注")
SUCCESS_STATUSES = ("成約", "検収", "検収完了")

_MODEL_CACHE: dict[str, Any] = {"mtime": None, "db_mtime": None, "model": None}


def _safe_float(val: Any, default: float = np.nan) -> float:
    try:
        if val is None or val == "":
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _case_lookup(case: dict, *keys: str, default: Any = None) -> Any:
    inputs = case.get("inputs") or {}
    result = case.get("result") or {}
    financials = result.get("financials") or case.get("financials") or {}
    for key in keys:
        if key in case and case.get(key) not in (None, ""):
            return case.get(key)
        if key in inputs and inputs.get(key) not in (None, ""):
            return inputs.get(key)
        if key in result and result.get(key) not in (None, ""):
            return result.get(key)
        if key in financials and financials.get(key) not in (None, ""):
            return financials.get(key)
    return default


def _parse_grade(grade: Any) -> float:
    s = str(grade or "").strip()
    if not s or s in {"0", "無格付"}:
        return 0.0
    if "要注意" in s or "7-9" in s or "③" in s or s == "C":
        return 0.25
    if "4-6" in s or "②" in s or s == "B":
        return 0.7
    if "1-3" in s or "①" in s or s == "A":
        return 1.0
    return 0.5


def _industry_buckets(industry_major: Any) -> dict[str, float]:
    s = str(industry_major or "")
    code = s.split(" ")[0].strip() if s else ""
    return {
        "ind_d": 1.0 if code == "D" else 0.0,
        "ind_e": 1.0 if code == "E" else 0.0,
        "ind_h": 1.0 if code == "H" else 0.0,
        "ind_i": 1.0 if code == "I" else 0.0,
        "ind_m": 1.0 if code == "M" else 0.0,
        "ind_p": 1.0 if code == "P" else 0.0,
        "ind_r": 1.0 if code == "R" else 0.0,
    }


def _missing_count_30(case: dict) -> int:
    barrier_specs = [
        ("nenshu", ("nenshu",), "numeric"),
        ("op_profit", ("op_profit", "rieki"), "numeric"),
        ("ord_profit", ("ord_profit",), "numeric"),
        ("net_income", ("net_income",), "numeric"),
        ("gross_profit", ("gross_profit",), "numeric"),
        ("total_assets", ("total_assets", "assets"), "numeric"),
        ("net_assets", ("net_assets",), "numeric"),
        ("machines", ("machines",), "numeric"),
        ("other_assets", ("other_assets",), "numeric"),
        ("rent", ("rent",), "numeric"),
        ("depreciation", ("depreciation", "dep_expense"), "numeric"),
        ("rent_expense", ("rent_expense",), "numeric"),
        ("bank_credit", ("bank_credit",), "numeric"),
        ("lease_credit", ("lease_credit",), "numeric"),
        ("grade", ("grade",), "categorical"),
        ("main_bank", ("main_bank",), "categorical"),
        ("competitor", ("competitor",), "categorical"),
        ("customer_type", ("customer_type",), "categorical"),
        ("deal_source", ("deal_source",), "categorical"),
        ("lease_term", ("lease_term",), "numeric"),
        ("acceptance_year", ("acceptance_year",), "numeric"),
        ("acquisition_cost", ("acquisition_cost",), "numeric"),
        ("contracts", ("contracts",), "numeric"),
        ("industry_major", ("industry_major",), "categorical"),
        ("industry_sub", ("industry_sub",), "categorical"),
        ("equity_ratio", ("equity_ratio", "user_eq"), "numeric"),
        ("score", ("score",), "numeric"),
        ("score_borrower", ("score_borrower",), "numeric"),
        ("asset_score", ("asset_score",), "numeric"),
        ("final_rate", ("final_rate",), "numeric"),
    ]
    missing = 0
    for _, keys, kind in barrier_specs:
        value = _case_lookup(case, *keys, default=None)
        if value is None:
            missing += 1
            continue
        s = str(value).strip()
        if s in {"", "0", "0.0", "0%", "未設定", "未読取", "無格付", "None"}:
            missing += 1
            continue
        if kind == "numeric":
            try:
                if float(value) <= 0:
                    missing += 1
            except (TypeError, ValueError):
                missing += 1
    return missing


def _feature_row(case: dict) -> dict[str, float]:
    fin = case.get("financials") or (case.get("result") or {}).get("financials") or {}
    sales = _safe_float(_case_lookup(case, "nenshu"), np.nan)
    op_profit = _safe_float(_case_lookup(case, "op_profit", "rieki"), np.nan)
    ord_profit = _safe_float(_case_lookup(case, "ord_profit"), np.nan)
    net_income = _safe_float(_case_lookup(case, "net_income"), np.nan)
    gross_profit = _safe_float(_case_lookup(case, "gross_profit"), np.nan)
    total_assets = _safe_float(_case_lookup(case, "total_assets", "assets"), np.nan)
    net_assets = _safe_float(_case_lookup(case, "net_assets"), np.nan)
    bank_credit = _safe_float(_case_lookup(case, "bank_credit"), np.nan)
    lease_credit = _safe_float(_case_lookup(case, "lease_credit"), np.nan)
    depreciation = _safe_float(_case_lookup(case, "depreciation", "dep_expense"), np.nan)
    rent_expense = _safe_float(_case_lookup(case, "rent_expense"), np.nan)
    acquisition_cost = _safe_float(_case_lookup(case, "acquisition_cost"), np.nan)
    contracts = _safe_float(_case_lookup(case, "contracts"), np.nan)
    final_rate = _safe_float(_case_lookup(case, "final_rate"), np.nan)
    competitor_rate = _safe_float(_case_lookup(case, "competitor_rate"), np.nan)
    score = _safe_float(_case_lookup(case, "score"), np.nan)
    score_borrower = _safe_float(_case_lookup(case, "score_borrower"), np.nan)
    asset_score = _safe_float(_case_lookup(case, "asset_score"), np.nan)
    equity_ratio = _safe_float(_case_lookup(case, "equity_ratio", "user_eq"), np.nan)

    sales_safe = max(float(sales) if np.isfinite(sales) else np.nan, 1.0)
    op_safe = max(abs(float(op_profit)) if np.isfinite(op_profit) else np.nan, 1.0)
    ord_safe = max(sales_safe, 1.0)
    total_safe = max(float(total_assets) if np.isfinite(total_assets) else np.nan, 1.0)
    dep_safe = max(float(depreciation) if np.isfinite(depreciation) else np.nan, 1.0)
    bank_safe = max(float(bank_credit) if np.isfinite(bank_credit) else np.nan, 1.0)
    lease_safe = max(float(lease_credit) if np.isfinite(lease_credit) else np.nan, 1.0)

    missing = _missing_count_30(case)
    barrier_energy = missing / 30.0

    main_bank = str(_case_lookup(case, "main_bank", default="") or "")
    competitor = str(_case_lookup(case, "competitor", default="") or "")
    customer_type = str(_case_lookup(case, "customer_type", default="") or "")
    deal_source = str(_case_lookup(case, "deal_source", default="") or "")
    sales_dept = str(_case_lookup(case, "sales_dept", default="") or "")
    industry_major = str(_case_lookup(case, "industry_major", default="") or "")

    features = {
        "sales_log": math.log1p(max(float(sales) if np.isfinite(sales) else 0.0, 0.0)),
        "op_margin": float(op_profit / sales_safe) if np.isfinite(op_profit) else np.nan,
        "ord_margin": float(ord_profit / sales_safe) if np.isfinite(ord_profit) else np.nan,
        "net_margin": float(net_income / sales_safe) if np.isfinite(net_income) else np.nan,
        "gross_margin": float(gross_profit / sales_safe) if np.isfinite(gross_profit) else np.nan,
        "bank_sales_ratio": float(bank_credit / sales_safe) if np.isfinite(bank_credit) else np.nan,
        "lease_sales_ratio": float(lease_credit / sales_safe) if np.isfinite(lease_credit) else np.nan,
        "debt_sales_ratio": float((bank_credit + lease_credit) / sales_safe) if (np.isfinite(bank_credit) and np.isfinite(lease_credit)) else np.nan,
        "dep_sales_ratio": float(depreciation / sales_safe) if np.isfinite(depreciation) else np.nan,
        "dep_profit_ratio": float(depreciation / op_safe) if np.isfinite(depreciation) else np.nan,
        "equity_ratio": equity_ratio,
        "grade_score": _parse_grade(_case_lookup(case, "grade")),
        "final_rate": final_rate,
        "rate_spread": float(final_rate - competitor_rate) if (np.isfinite(final_rate) and np.isfinite(competitor_rate)) else np.nan,
        "barrier_energy": barrier_energy,
        "completion_ratio": 1.0 - barrier_energy,
        "main_bank_is_main": 1.0 if "メイン先" in main_bank else 0.0,
        "competitor_present": 1.0 if "競合あり" in competitor else 0.0,
        "customer_is_new": 1.0 if "新規" in customer_type else 0.0,
        "deal_source_bank": 1.0 if "銀行紹介" in deal_source else 0.0,
        "score_borrower": score_borrower,
        "asset_score": asset_score,
        "contracts_log": math.log1p(max(float(contracts) if np.isfinite(contracts) else 0.0, 0.0)),
        "sales_dept_utsunomiya": 1.0 if sales_dept == "宇都宮営業部" else 0.0,
        "sales_dept_oyama": 1.0 if sales_dept == "小山営業部" else 0.0,
        "sales_dept_ashikaga": 1.0 if sales_dept == "足利営業部" else 0.0,
        "sales_dept_saitama": 1.0 if sales_dept == "埼玉営業部" else 0.0,
    }
    features.update(_industry_buckets(industry_major))
    return features


def _build_dataset(cases: list[dict]) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    rows = []
    labels = []
    baseline_scores = []
    for case in cases:
        status = case.get("final_status")
        if status not in VALID_STATUSES:
            continue
        rows.append(_feature_row(case))
        labels.append(1 if status in SUCCESS_STATUSES else 0)
        baseline_scores.append(_safe_float(_case_lookup(case, "score_borrower", "score"), np.nan))
    if not rows:
        return pd.DataFrame(), np.array([], dtype=int), pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    return df, np.asarray(labels, dtype=int), pd.Series(baseline_scores, dtype=float)


def _metrics_from_prob(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else 0.5
    acc = accuracy_score(y, pred)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f2 = (5 * prec * rec / (4 * prec + rec)) if (prec + rec) else 0.0
    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(specificity),
        "f2": float(f2),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "tn": float(tn),
        "threshold": float(threshold),
    }


def _best_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, dict[str, float]]:
    best_th = 0.5
    best_metrics = _metrics_from_prob(y, prob, 0.5)
    best_score = -1.0
    for th in np.linspace(0.10, 0.90, 81):
        metrics = _metrics_from_prob(y, prob, float(th))
        score = 0.60 * metrics["auc"] + 0.40 * metrics["f2"]
        if score > best_score:
            best_score = score
            best_th = float(th)
            best_metrics = metrics
    return best_th, best_metrics


def _evaluate_subset(X: pd.DataFrame, y: np.ndarray, subset: list[str], cv: StratifiedKFold) -> dict[str, Any]:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
    ])
    prob = cross_val_predict(pipe, X[subset], y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    th, metrics = _best_threshold(y, prob)
    score = 0.60 * metrics["auc"] + 0.40 * metrics["f2"]
    return {
        "features": list(subset),
        "threshold": th,
        "metrics": metrics,
        "objective": float(score),
        "prob": prob,
    }


def _feature_pool(X: pd.DataFrame, y: np.ndarray) -> list[str]:
    usable = []
    for col in X.columns:
        series = pd.to_numeric(X[col], errors="coerce")
        non_na = series.notna().mean()
        if non_na < 0.25:
            continue
        if series.fillna(0).nunique() <= 1:
            continue
        usable.append(col)
    return usable


@dataclass
class TunnelOptimizerModel:
    feature_names: list[str]
    selected_features: list[str]
    pipeline: Pipeline
    threshold: float
    gamma: float
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    n_cases: int
    n_success: int
    n_failure: int
    trained_at: str
    search_summary: pd.DataFrame


def _fit_pipeline(X: pd.DataFrame, y: np.ndarray, selected: list[str]) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
    ])
    pipe.fit(X[selected], y)
    return pipe


def _tunnel_adjust(prob: np.ndarray, barrier_energy: np.ndarray, gamma: float) -> np.ndarray:
    core = np.clip(prob, 0.02, 0.98)
    barrier = np.clip(barrier_energy, 0.0, 1.0)
    tunnel = np.exp(-(gamma * barrier) / np.maximum(core, 0.05))
    return np.clip(core * tunnel, 0.0, 1.0)


def train_tunnel_model(cases: list[dict] | None = None) -> TunnelOptimizerModel | None:
    cases = cases if cases is not None else load_all_cases()
    X, y, baseline = _build_dataset(cases)
    if X.empty or len(y) < MIN_CASES or len(np.unique(y)) < 2:
        return None

    baseline_prob = baseline.fillna(float(baseline.median() if baseline.notna().any() else 0.5)).clip(0, 1).to_numpy()
    baseline_th, baseline_metrics = _best_threshold(y, baseline_prob)
    baseline_metrics["threshold"] = baseline_th
    baseline_score = 0.60 * baseline_metrics["auc"] + 0.40 * baseline_metrics["f2"]

    pool = _feature_pool(X, y)
    if len(pool) < 5:
        return None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    single_results: list[dict[str, Any]] = []
    for feat in pool:
        single_results.append(_evaluate_subset(X, y, [feat], cv))
    single_results.sort(key=lambda r: (r["objective"], r["metrics"]["auc"], r["metrics"]["recall"]), reverse=True)

    candidate_pool = [r["features"][0] for r in single_results[:18]]
    if len(candidate_pool) < 3:
        candidate_pool = [r["features"][0] for r in single_results[:min(18, len(single_results))]]

    all_results = list(single_results)
    for size in (2, 3):
        for subset in itertools.combinations(candidate_pool, size):
            all_results.append(_evaluate_subset(X, y, list(subset), cv))

    all_results.sort(key=lambda r: (r["objective"], r["metrics"]["auc"], r["metrics"]["recall"]), reverse=True)
    best = all_results[0]

    if best["metrics"]["auc"] < max(0.60, baseline_metrics["auc"] - 0.005):
        return None
    if best["objective"] < baseline_score + 0.01:
        return None

    selected = best["features"]
    pipeline = _fit_pipeline(X, y, selected)
    oof_prob = cross_val_predict(pipeline, X[selected], y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    barrier_energy = X["barrier_energy"].to_numpy(dtype=float)

    best_gamma = 1.0
    best_gamma_score = -1.0
    best_tunnel_prob = oof_prob
    best_tunnel_th = 0.5
    best_tunnel_metrics = _metrics_from_prob(y, oof_prob, 0.5)
    for gamma in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        tunnel_prob = _tunnel_adjust(oof_prob, barrier_energy, gamma)
        th, metrics = _best_threshold(y, tunnel_prob)
        score = 0.60 * metrics["auc"] + 0.40 * metrics["f2"]
        if score > best_gamma_score:
            best_gamma_score = score
            best_gamma = gamma
            best_tunnel_prob = tunnel_prob
            best_tunnel_th = th
            best_tunnel_metrics = metrics

    if best_tunnel_metrics["auc"] < max(0.60, baseline_metrics["auc"] - 0.005):
        return None

    final_pipeline = _fit_pipeline(X, y, selected)
    final_prob = final_pipeline.predict_proba(X[selected])[:, 1]
    final_tunnel_prob = _tunnel_adjust(final_prob, barrier_energy, best_gamma)
    final_th, final_metrics = _best_threshold(y, final_tunnel_prob)
    final_metrics["threshold"] = final_th
    final_metrics["objective"] = 0.60 * final_metrics["auc"] + 0.40 * final_metrics["f2"]

    coef = final_pipeline.named_steps["clf"].coef_.reshape(-1)
    coef_df = pd.DataFrame({
        "feature": selected,
        "coef": coef[:len(selected)],
        "abs_coef": np.abs(coef[:len(selected)]),
    }).sort_values("abs_coef", ascending=False)

    summary = pd.DataFrame(all_results[:40]).assign(
        n_features=lambda d: d["features"].map(len),
        features_text=lambda d: d["features"].map(lambda xs: ", ".join(xs)),
    )

    model = TunnelOptimizerModel(
        feature_names=list(X.columns),
        selected_features=selected,
        pipeline=final_pipeline,
        threshold=best_tunnel_th,
        gamma=best_gamma,
        metrics={
            **final_metrics,
            "objective": float(final_metrics["objective"]),
            "cv_auc": float(best_tunnel_metrics["auc"]),
            "cv_f2": float(best_tunnel_metrics["f2"]),
        },
        baseline_metrics=baseline_metrics,
        n_cases=int(len(y)),
        n_success=int(y.sum()),
        n_failure=int((1 - y).sum()),
        trained_at=_dt.datetime.now().isoformat(timespec="seconds"),
        search_summary=coef_df,
    )

    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
    except Exception:
        pass
    return model


def _db_mtime() -> float:
    from data_cases import DB_PATH
    return os.path.getmtime(DB_PATH) if os.path.exists(DB_PATH) else 0.0


def load_tunnel_model(force_retrain: bool = False) -> TunnelOptimizerModel | None:
    global _MODEL_CACHE
    db_mtime = _db_mtime()
    model_mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0.0
    if not force_retrain and _MODEL_CACHE.get("model") is not None:
        if _MODEL_CACHE.get("mtime") == model_mtime and _MODEL_CACHE.get("db_mtime") == db_mtime:
            return _MODEL_CACHE["model"]
    if not force_retrain and os.path.exists(MODEL_PATH) and model_mtime >= db_mtime:
        try:
            model = joblib.load(MODEL_PATH)
            _MODEL_CACHE = {"mtime": model_mtime, "db_mtime": db_mtime, "model": model}
            return model
        except Exception:
            pass

    model = train_tunnel_model()
    if model is None:
        return None
    try:
        mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    except Exception:
        mtime = None
    _MODEL_CACHE = {"mtime": mtime, "db_mtime": db_mtime, "model": model}
    return model


def score_tunnel_case(case: dict, model: TunnelOptimizerModel | None = None) -> dict[str, Any] | None:
    model = model or load_tunnel_model()
    if model is None:
        return None
    row = _feature_row(case)
    X = pd.DataFrame([row])
    base_prob = float(model.pipeline.predict_proba(X[model.selected_features])[0, 1])
    barrier_energy = float(row.get("barrier_energy", 0.0) or 0.0)
    tunnel_prob = float(_tunnel_adjust(np.array([base_prob]), np.array([barrier_energy]), model.gamma)[0])
    coef = model.pipeline.named_steps["clf"].coef_.reshape(-1)
    values = model.pipeline.named_steps["imputer"].transform(X[model.selected_features])
    scaled = model.pipeline.named_steps["scaler"].transform(values)
    contribs = []
    for feat, val, coef_i in zip(model.selected_features, scaled.reshape(-1), coef[:len(model.selected_features)]):
        contribs.append({
            "feature": feat,
            "value": float(val),
            "coef": float(coef_i),
            "impact": float(val * coef_i),
        })
    contribs = sorted(contribs, key=lambda x: abs(x["impact"]), reverse=True)
    return {
        "proxy_score": round(tunnel_prob * 100, 1),
        "base_probability": round(base_prob * 100, 1),
        "tunnel_probability": round(tunnel_prob * 100, 1),
        "barrier_energy": round(barrier_energy, 4),
        "barrier_count_30": int(round(barrier_energy * 30)),
        "gamma": round(float(model.gamma), 3),
        "selected_features": list(model.selected_features),
        "top_contributions": contribs[:8],
        "metrics": model.metrics,
        "baseline_metrics": model.baseline_metrics,
        "trained_at": model.trained_at,
        "n_cases": model.n_cases,
        "n_success": model.n_success,
        "n_failure": model.n_failure,
        "threshold": model.threshold,
    }


def build_tunnel_correction(case: dict, model: TunnelOptimizerModel | None = None) -> dict[str, Any] | None:
    scored = score_tunnel_case(case, model=model)
    if scored is None:
        return None
    return {
        "source": "tunnel",
        "proxy": scored,
        "weighted_score": scored["proxy_score"],
        "combined_score": scored["proxy_score"],
        "rank": None,
        "rank_text": "トンネル補完",
        "rank_desc": "30項目の欠損障壁を考慮し、総当たりで選ばれた補助因子から成約到達確率を算出したスコアです。",
        "items": {
            feat: {
                "label": feat,
                "value": None,
                "weight": 0,
                "level_label": f"{feat}",
            } for feat in scored["selected_features"][:3]
        },
    }
