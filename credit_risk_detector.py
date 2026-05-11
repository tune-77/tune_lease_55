"""Credit-risk group detector learned from excluded grade cases."""

from __future__ import annotations

import json
import os
import sqlite3
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from grade_normalizer import is_excluded_grade, normalize_grade

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.join(_SCRIPT_DIR, "data", "lease_data.db")

_NUM_FEATURES = [
    ("売上高", "nenshu"),
    ("売上高総利益", "gross_profit"),
    ("営業利益", "op_profit"),
    ("経常利益", "ord_profit"),
    ("当期利益", "net_income"),
    ("減価償却", "depreciation"),
    ("機械・装置", "machines"),
    ("その他(有形固定資産)", "other_assets"),
    ("賃借料", "rent"),
    ("賃借料(経費)", "rent_expense"),
    ("銀行借入", "bank_credit"),
    ("リース残高", "lease_credit"),
    ("メイン非メイン", "main_bank_binary"),
]
_NUM_COLS = [label for label, _ in _NUM_FEATURES]
_CAT_COLS = ["業種", "格付", "ソース", "取引区分"]
_ALL_COLS = _NUM_COLS + _CAT_COLS


def _safe_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _money_million(value: Any) -> float:
    return _safe_float(value) / 1000.0


def _case_to_features(case: dict[str, Any]) -> dict[str, Any]:
    inp = case.get("inputs") or case
    row: dict[str, Any] = {}
    for label, key in _NUM_FEATURES:
        if key == "main_bank_binary":
            val = case.get("main_bank") or inp.get("main_bank") or ""
            row[label] = 1.0 if val == "メイン先" else 0.0
        else:
            row[label] = _money_million(inp.get(key, case.get(key)))
    row["業種"] = case.get("industry_major") or inp.get("industry_major") or "未設定"
    row["格付"] = normalize_grade(inp.get("grade") or case.get("grade"))
    row["ソース"] = inp.get("deal_source") or case.get("deal_source") or "未設定"
    row["取引区分"] = case.get("customer_type") or inp.get("customer_type") or "既存先"
    return row


def _load_training_rows(db_path: str) -> tuple[pd.DataFrame | None, np.ndarray | None, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    stats = {"normal": 0, "excluded": 0}
    if not os.path.exists(db_path):
        return None, None, stats

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        normal_rows = conn.execute(
            """
            SELECT data
            FROM past_cases
            WHERE final_status IN ('成約', '検収', '検収完了', '失注')
            """
        ).fetchall()
        for r in normal_rows:
            try:
                rows.append(_case_to_features(json.loads(r["data"] or "{}")))
                labels.append(0)
                stats["normal"] += 1
            except Exception:
                continue

        has_excluded = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='excluded_grade_cases'"
        ).fetchone()
        if has_excluded:
            excluded_rows = conn.execute("SELECT data FROM excluded_grade_cases").fetchall()
            for r in excluded_rows:
                try:
                    rows.append(_case_to_features(json.loads(r["data"] or "{}")))
                    labels.append(1)
                    stats["excluded"] += 1
                except Exception:
                    continue

    if not rows:
        return None, None, stats
    return pd.DataFrame(rows, columns=_ALL_COLS), np.array(labels, dtype=int), stats


@lru_cache(maxsize=4)
def _fit_detector(db_path: str, db_mtime: float):
    X, y, stats = _load_training_rows(db_path)
    if X is None or y is None or stats["excluded"] < 10 or stats["normal"] < 50 or len(set(y.tolist())) < 2:
        return None, stats

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), _NUM_COLS),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="未設定")),
                        ("onehot", encoder),
                    ]
                ),
                _CAT_COLS,
            ),
        ]
    )
    model = Pipeline(
        [
            ("pre", pre),
            (
                "lr",
                LogisticRegression(
                    C=0.5,
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model, stats


def detect_credit_risk_group(inputs: dict[str, Any], db_path: str = _DB_PATH) -> dict[str, Any]:
    """Return similarity score to the excluded credit-risk company group."""
    raw_grade = inputs.get("grade")
    if is_excluded_grade(raw_grade):
        return {
            "score": 100.0,
            "level": "excluded_grade",
            "flag": True,
            "reasons": ["格付が 8-3 / 9 / 10 に該当するため、信用リスクDATA群として検知"],
            "training_normal": 0,
            "training_excluded": 0,
            "available": True,
        }

    try:
        db_mtime = os.path.getmtime(db_path)
        model, stats = _fit_detector(db_path, db_mtime)
    except Exception as exc:
        return {
            "score": 0.0,
            "level": "unavailable",
            "flag": False,
            "reasons": [f"信用リスク群モデルを利用できません: {exc}"],
            "training_normal": 0,
            "training_excluded": 0,
            "available": False,
        }

    if model is None:
        return {
            "score": 0.0,
            "level": "unavailable",
            "flag": False,
            "reasons": ["信用リスク群の教師データが不足しています"],
            "training_normal": stats.get("normal", 0),
            "training_excluded": stats.get("excluded", 0),
            "available": False,
        }

    X = pd.DataFrame([_case_to_features(inputs)], columns=_ALL_COLS)
    prob = float(model.predict_proba(X)[0][1])
    score = round(prob * 100.0, 1)
    if score >= 70:
        level = "high"
    elif score >= 45:
        level = "watch"
    else:
        level = "normal"

    reasons = [f"除外格付DATA群との類似スコア {score:.1f}/100"]
    if level == "high":
        reasons.append("信用リスク群に近い財務・属性パターンです")
    elif level == "watch":
        reasons.append("信用リスク群との中程度の類似があります")

    return {
        "score": score,
        "level": level,
        "flag": level in {"high", "watch"},
        "reasons": reasons,
        "training_normal": stats.get("normal", 0),
        "training_excluded": stats.get("excluded", 0),
        "available": True,
    }
