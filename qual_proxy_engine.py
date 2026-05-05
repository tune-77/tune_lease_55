from __future__ import annotations

import datetime as _dt
import math
import os
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_cases import load_all_cases

_DATA_DIR = "/Users/kobayashiisaoryou/clawd/tune_lease_55/data"
MODEL_PATH = os.path.join(_DATA_DIR, "qual_proxy_model.joblib")

VALID_STATUSES = ("成約", "失注")
SUCCESS_STATUSES = ("成約", "検収", "検収完了")

CORE_FEATURES = [
    "dep_profit_ratio",
    "grade_score",
    "bank_sales_ratio",
]

BARRIER_FIELDS = [
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
    ("user_eq", ("equity_ratio", "user_eq"), "numeric"),
    ("score", ("score",), "numeric"),
    ("score_borrower", ("score_borrower",), "numeric"),
    ("asset_score", ("asset_score",), "numeric"),
    ("final_rate", ("final_rate",), "numeric"),
]

FEATURES = CORE_FEATURES + [
    "lease_sales_ratio",
    "debt_sales_ratio",
    "op_margin",
    "equity_ratio",
    "sales_log",
    "qual_missing_count",
    "barrier_count_30",
    "has_depreciation",
    "has_bank_credit",
    "has_grade",
    "op_nonpos",
    "dep_x_grade",
    "bank_x_grade",
    "dep_x_bank",
    "margin_x_grade",
]

BLEND_WEIGHT_PROB = 0.7
BLEND_WEIGHT_CLUSTER = 0.3

_MODEL_CACHE: dict[str, Any] = {"mtime": None, "db_mtime": None, "model": None}


def _safe_float(val: Any, default: float = 0.0) -> float:
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


def _grade_score(grade: Any) -> float:
    s = str(grade or "").strip()
    if not s or s == "0":
        return 0.0
    if "無格付" in s:
        return 0.0
    if "要注意" in s or "7-9" in s or "③" in s or "C" == s:
        return 0.25
    if "4-6" in s or "②" in s or "B" == s:
        return 0.7
    if "1-3" in s or "①" in s or "A" == s:
        return 1.0
    return 0.5


def _is_missing(case: dict, keys: tuple[str, ...], kind: str) -> bool:
    value = _case_lookup(case, *keys, default=None)
    if value is None:
        return True
    s = str(value).strip()
    if s in ("", "0", "0.0", "0%", "未設定", "未読取", "無格付", "None"):
        return True
    if kind == "numeric":
        try:
            return float(value) <= 0
        except (TypeError, ValueError):
            return True
    return False


def _build_feature_row(case: dict) -> dict[str, float]:
    sales = max(_safe_float(_case_lookup(case, "nenshu", default=0.0)), 0.0)
    op_profit = _safe_float(_case_lookup(case, "op_profit", "rieki", default=0.0))
    depreciation = max(_safe_float(_case_lookup(case, "depreciation", "dep_expense", default=0.0)), 0.0)
    bank_credit = max(_safe_float(_case_lookup(case, "bank_credit", default=0.0)), 0.0)
    lease_credit = max(_safe_float(_case_lookup(case, "lease_credit", default=0.0)), 0.0)
    equity_ratio = _safe_float(_case_lookup(case, "equity_ratio", "user_eq", default=0.0))
    grade = _case_lookup(case, "grade", default="")

    sales_safe = max(sales, 1.0)
    op_abs_safe = max(abs(op_profit), 1.0)
    dep_profit_ratio = depreciation / op_abs_safe
    dep_profit_ratio = float(np.clip(dep_profit_ratio, 0.0, 20.0))
    bank_sales_ratio = bank_credit / sales_safe
    bank_sales_ratio = float(np.clip(bank_sales_ratio, 0.0, 20.0))
    lease_sales_ratio = lease_credit / sales_safe
    lease_sales_ratio = float(np.clip(lease_sales_ratio, 0.0, 20.0))
    debt_sales_ratio = (bank_credit + lease_credit) / sales_safe
    debt_sales_ratio = float(np.clip(debt_sales_ratio, 0.0, 30.0))
    op_margin = op_profit / sales_safe
    op_margin = float(np.clip(op_margin, -5.0, 5.0))
    sales_log = math.log1p(sales)
    grade_score = _grade_score(grade)
    has_depreciation = 1.0 if depreciation > 0 else 0.0
    has_bank_credit = 1.0 if bank_credit > 0 else 0.0
    has_grade = 1.0 if str(grade or "").strip() not in ("", "0") else 0.0
    op_nonpos = 1.0 if op_profit <= 0 else 0.0
    qual_missing_count = float(3 - int(has_depreciation > 0) - int(has_bank_credit > 0) - int(has_grade > 0))
    barrier_count_30 = float(sum(1 for _, keys, kind in BARRIER_FIELDS if _is_missing(case, keys, kind)))

    return {
        "dep_profit_ratio": dep_profit_ratio,
        "grade_score": grade_score,
        "bank_sales_ratio": bank_sales_ratio,
        "lease_sales_ratio": lease_sales_ratio,
        "debt_sales_ratio": debt_sales_ratio,
        "op_margin": op_margin,
        "equity_ratio": equity_ratio,
        "sales_log": sales_log,
        "qual_missing_count": qual_missing_count,
        "barrier_count_30": barrier_count_30,
        "has_depreciation": has_depreciation,
        "has_bank_credit": has_bank_credit,
        "has_grade": has_grade,
        "op_nonpos": op_nonpos,
        "dep_x_grade": dep_profit_ratio * grade_score,
        "bank_x_grade": bank_sales_ratio * grade_score,
        "dep_x_bank": dep_profit_ratio * bank_sales_ratio,
        "margin_x_grade": op_margin * grade_score,
    }


def _build_dataset(cases: list[dict]) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    labels = []
    for case in cases:
        status = case.get("final_status")
        if status not in VALID_STATUSES:
            continue
        rows.append(_build_feature_row(case))
        labels.append(1 if status in SUCCESS_STATUSES else 0)
    if not rows:
        return pd.DataFrame(columns=FEATURES), np.array([], dtype=int)
    df = pd.DataFrame(rows).fillna(0.0)
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0
    return df[FEATURES].astype(float), np.array(labels, dtype=int)


def _fit_precision(success_core: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    core = np.asarray(success_core, dtype=float)
    if core.ndim != 2 or len(core) == 0:
        raise ValueError("success_core must be a 2D array with samples")
    mean = core.mean(axis=0)
    if len(core) >= max(10, core.shape[1] + 2):
        try:
            est = MinCovDet(support_fraction=0.8, random_state=42).fit(core)
            cov = est.covariance_
        except Exception:
            cov = np.cov(core, rowvar=False)
    else:
        cov = np.cov(core, rowvar=False)
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    precision = np.linalg.pinv(cov)
    dist = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", core - mean, precision, core - mean), 0.0))
    scale = float(np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return mean, precision, scale


@dataclass
class ProxyQualModel:
    feature_names: list[str]
    core_features: list[str]
    scaler: StandardScaler
    classifier: LogisticRegression
    core_mean: np.ndarray
    core_precision: np.ndarray
    distance_scale: float
    metrics: dict[str, float]
    n_cases: int
    n_success: int
    n_failure: int
    trained_at: str


def _compute_core_distance(core_X: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    delta = core_X - mean
    dist_sq = np.einsum("ij,jk,ik->i", delta, precision, delta)
    return np.sqrt(np.maximum(dist_sq, 0.0))


def train_proxy_model(cases: list[dict] | None = None, force: bool = False) -> ProxyQualModel | None:
    cases = cases if cases is not None else load_all_cases()
    X, y = _build_dataset(cases)
    if X.empty or len(y) < 1500 or len(np.unique(y)) < 2:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    core_idx = [FEATURES.index(f) for f in CORE_FEATURES]
    success_mask = y == 1
    success_core = X_scaled[success_mask][:, core_idx]
    if len(success_core) == 0:
        return None
    core_mean, core_precision, distance_scale = _fit_precision(success_core)
    core_dist = _compute_core_distance(X_scaled[:, core_idx], core_mean, core_precision)
    X_aug = np.column_stack([X_scaled, core_dist])

    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_aug, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except Exception:
        X_train, X_test, y_train, y_test = X_aug, X_aug, y, y

    clf_eval = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    clf_eval.fit(X_train, y_train)
    test_prob = clf_eval.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else 0.5,
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
    }
    if metrics["auc"] < 0.55 or metrics["accuracy"] < 0.55:
        return None

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    clf.fit(X_aug, y)

    model = ProxyQualModel(
        feature_names=list(FEATURES),
        core_features=list(CORE_FEATURES),
        scaler=scaler,
        classifier=clf,
        core_mean=core_mean,
        core_precision=core_precision,
        distance_scale=distance_scale,
        metrics=metrics,
        n_cases=int(len(y)),
        n_success=int(success_mask.sum()),
        n_failure=int((~success_mask).sum()),
        trained_at=_dt.datetime.now().isoformat(timespec="seconds"),
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


def load_proxy_model(force_retrain: bool = False) -> ProxyQualModel | None:
    global _MODEL_CACHE
    db_mtime = _db_mtime()
    model_mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0.0
    if not force_retrain:
        if _MODEL_CACHE.get("model") is not None:
            if _MODEL_CACHE.get("mtime") == model_mtime and _MODEL_CACHE.get("db_mtime") == db_mtime:
                return _MODEL_CACHE["model"]
        if os.path.exists(MODEL_PATH) and model_mtime >= db_mtime:
            try:
                model = joblib.load(MODEL_PATH)
                _MODEL_CACHE = {"mtime": model_mtime, "db_mtime": db_mtime, "model": model}
                return model
            except Exception:
                pass

    model = train_proxy_model(force=force_retrain)
    if model is None:
        return None
    try:
        mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    except Exception:
        mtime = None
    _MODEL_CACHE = {"mtime": mtime, "db_mtime": db_mtime, "model": model}
    return model


def score_proxy_case(case: dict, model: ProxyQualModel | None = None) -> dict[str, Any] | None:
    model = model or load_proxy_model()
    if model is None:
        return None

    row = _build_feature_row(case)
    df = pd.DataFrame([row], columns=model.feature_names)
    X_scaled = model.scaler.transform(df.values)
    core_idx = [model.feature_names.index(f) for f in model.core_features]
    core = X_scaled[:, core_idx]
    distance = float(_compute_core_distance(core, model.core_mean, model.core_precision)[0])
    aug = np.column_stack([X_scaled, np.array([distance], dtype=float)])
    probability = float(model.classifier.predict_proba(aug)[0, 1])
    similarity = float(np.exp(-distance / max(model.distance_scale, 1e-6)))
    barrier_count = float(row.get("barrier_count_30", 0.0) or 0.0)
    barrier_energy = barrier_count / max(len(BARRIER_FIELDS), 1)
    tunnel_component = float(np.exp(-barrier_energy / max(similarity, 0.05)))
    proxy_score = float(np.clip((BLEND_WEIGHT_PROB * probability + BLEND_WEIGHT_CLUSTER * tunnel_component) * 100, 0, 100))

    coef = model.classifier.coef_.reshape(-1)
    feature_names = model.feature_names + ["success_cluster_distance"]
    contribs = []
    for name, value, coef_i in zip(feature_names, aug.reshape(-1), coef):
        contribs.append({
            "feature": name,
            "value": float(value),
            "coef": float(coef_i),
            "impact": float(value * coef_i),
        })
    contribs = sorted(contribs, key=lambda x: abs(x["impact"]), reverse=True)

    return {
        "proxy_score": round(proxy_score, 1),
        "probability": round(probability * 100, 1),
        "cluster_similarity": round(similarity * 100, 1),
        "cluster_distance": round(distance, 4),
        "barrier_count_30": round(barrier_count, 1),
        "barrier_energy": round(barrier_energy, 4),
        "tunnel_component": round(tunnel_component * 100, 1),
        "distance_scale": round(float(model.distance_scale), 4),
        "feature_values": row,
        "top_contributions": contribs[:5],
        "metrics": model.metrics,
        "trained_at": model.trained_at,
        "n_cases": model.n_cases,
        "n_success": model.n_success,
        "n_failure": model.n_failure,
    }


def build_proxy_correction(case: dict, model: ProxyQualModel | None = None) -> dict[str, Any] | None:
    scored = score_proxy_case(case, model=model)
    if scored is None:
        return None
    return {
        "source": "proxy",
        "proxy": scored,
        "weighted_score": scored["proxy_score"],
        "combined_score": scored["proxy_score"],
        "rank": None,
        "rank_text": "代理定性",
        "rank_desc": "定性未入力時に、成功クラスター距離と教師あり補正で作った代理定性スコアです。",
        "items": {
            "dep_profit_ratio": {
                "label": "減価償却 / 営業利益",
                "value": scored["feature_values"]["dep_profit_ratio"],
                "weight": 40,
                "level_label": f"{scored['feature_values']['dep_profit_ratio']:.2f}",
            },
            "grade_score": {
                "label": "格付",
                "value": scored["feature_values"]["grade_score"],
                "weight": 30,
                "level_label": f"{scored['feature_values']['grade_score']:.2f}",
            },
            "bank_sales_ratio": {
                "label": "売上に対する銀行借入",
                "value": scored["feature_values"]["bank_sales_ratio"],
                "weight": 30,
                "level_label": f"{scored['feature_values']['bank_sales_ratio']:.2f}",
            },
        },
    }
