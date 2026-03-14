# -*- coding: utf-8 -*-
"""
sumaho10 用: 1件分の入力辞書から学習モデル（業種別ハイブリッド）で予測し、
既存確率・AI確率・ハイブリッド確率・判定・Top5理由を返す。
モデルが無い・エラー時は None を返す（本システムのみで動作させるため）。
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

warnings.filterwarnings("ignore")

# 円単位で渡す想定
def _ensure_float(v: Any) -> float:
    if v is None: return 0.0
    try: return float(v)
    except (TypeError, ValueError): return 0.0


def map_industry_major_to_scoring(selected_major: str) -> str:
    """
    sumaho の業種（例: "E 製造業", "D 建設業"）を
    学習モデルの業種ラベルに変換する。
    """
    if not selected_major or not isinstance(selected_major, str):
        return "サービス業"
    major = selected_major.strip()
    code = major.split()[0] if " " in major else major[:1]
    # 製造業, 建設業, サービス業, 卸売業, 小売業
    if code == "E":
        return "製造業"
    if code == "D":
        return "建設業"
    if code == "H":
        return "建設業"  # 運輸は建設業で近似
    if code == "P":
        return "サービス業"  # 医療・福祉
    if code in ("I", "K", "M", "R"):
        return "サービス業"
    if "卸売" in major:
        return "卸売業"
    if "小売" in major:
        return "小売業"
    return "サービス業"


def predict_one(
    revenue: float,
    total_assets: float,
    equity: float,
    operating_profit: float,
    net_income: float,
    machinery_equipment: float,
    other_fixed_assets: float,
    depreciation: float,
    rent_expense: float,
    industry: str,
    base_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    1件分の財務データで学習モデルを実行する。

    Args:
        revenue, total_assets, equity, operating_profit, net_income,
        machinery_equipment, other_fixed_assets, depreciation, rent_expense: 円単位
        industry: 学習モデル用業種（"製造業", "建設業", "サービス業", "卸売業", "小売業"）
        base_path: モデルディレクトリ（未指定時は scoring/models/industry_specific）

    Returns:
        {
            "legacy_prob": float,
            "ai_prob": float,
            "hybrid_prob": float,
            "decision": "承認" | "否決",
            "top5_reasons": ["feature: value", ...]
        } または エラー時 None
    """
    try:
        import numpy as np
        import pandas as pd
        import joblib
        from .feature_engineering_custom import CustomFinancialFeatures
        from .industry_hybrid_model import IndustrySpecificHybridModel
        from .model import CreditScoringModel
    except Exception as e:
        return None

    if base_path is None:
        base_path = str(Path(__file__).resolve().parent / "models" / "industry_specific")
    base = Path(base_path)
    if not base.exists():
        return None
    for f in ("industry_coefficients.pkl", "industry_intercepts.pkl", "unified_ai_model.pkl", "scaler.pkl", "label_encoder.pkl"):
        if not (base / f).exists():
            return None

    revenue = _ensure_float(revenue)
    total_assets = _ensure_float(total_assets)
    equity = _ensure_float(equity)
    if total_assets <= 0 or equity < 0:
        return None

    operating_profit = _ensure_float(operating_profit)
    net_income = _ensure_float(net_income)
    machinery_equipment = _ensure_float(machinery_equipment)
    other_fixed_assets = _ensure_float(other_fixed_assets)
    depreciation = _ensure_float(depreciation)
    rent_expense = _ensure_float(rent_expense)

    try:
        engine = CustomFinancialFeatures()
        industry_model = IndustrySpecificHybridModel()
        industry_model.industry_coefficients = joblib.load(base / "industry_coefficients.pkl")
        industry_model.industry_intercepts = joblib.load(base / "industry_intercepts.pkl")
        unified_ai = CreditScoringModel(model_type="lightgbm")
        unified_ai.load_model(str(base / "unified_ai_model.pkl"))
        scaler = joblib.load(base / "scaler.pkl")
        label_encoder = joblib.load(base / "label_encoder.pkl")
    except Exception:
        return None

    row = {
        "revenue": max(revenue, 1),
        "total_assets": total_assets,
        "equity": equity,
        "operating_profit": operating_profit,
        "net_income": net_income,
        "machinery_equipment": machinery_equipment,
        "other_fixed_assets": other_fixed_assets,
        "depreciation": depreciation,
        "rent_expense": rent_expense,
    }
    X = pd.DataFrame([row])
    features = engine.calculate_all_features(X)
    features = engine.calculate_risk_flags(features)
    features["industry"] = industry

    legacy_prob = float(industry_model.predict_by_industry(features)[0])
    important = engine.get_important_features_for_lease()
    scale_cols = [f for f in important if f in features.columns]
    X_scale = features[scale_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = pd.DataFrame(
        scaler.transform(X_scale),
        columns=scale_cols,
        index=X_scale.index,
    )
    X_scaled["industry_encoded"] = label_encoder.transform(features["industry"])
    if hasattr(unified_ai.model, "feature_importances_") and unified_ai.feature_importance is None:
        unified_ai.feature_importance = pd.DataFrame({
            "feature": X_scaled.columns.tolist(),
            "importance": unified_ai.model.feature_importances_,
        }).sort_values("importance", ascending=False)
    ai_prob = float(unified_ai.predict_proba(X_scaled)[0])
    hybrid_prob = 0.3 * legacy_prob + 0.7 * ai_prob
    decision = "承認" if hybrid_prob < 0.5 else "否決"

    top5_reasons: List[str] = []
    if unified_ai.feature_importance is not None and len(unified_ai.feature_importance) > 0:
        for f in unified_ai.feature_importance.head(5)["feature"].tolist():
            if f in features.columns:
                v = features[f].iloc[0]
                if pd.notna(v):
                    top5_reasons.append(f"{f}: {v:.2f}")

    return {
        "legacy_prob": round(legacy_prob, 4),
        "ai_prob": round(ai_prob, 4),
        "hybrid_prob": round(hybrid_prob, 4),
        "decision": decision,
        "top5_reasons": top5_reasons,
    }
