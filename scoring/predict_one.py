# -*- coding: utf-8 -*-
"""
sumaho10 用: 1件分の入力辞書から学習モデル（RandomForest + 業種別回帰）で予測し、
既存確率・AI確率・ハイブリッド確率・判定・Top5理由を返す。
RF bundle が無い場合のみ、旧 LGBM bundle にフォールバックする。
"""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

warnings.filterwarnings("ignore")

_logger = logging.getLogger(__name__)

# ── モデルキャッシュ（Streamlit内外どちらでも動作）──────────────────────────
_LEGACY_MODEL_CACHE: "dict[str, dict]" = {}
_RF_BUNDLE_CACHE: "dict[str, dict]" = {}


def _load_legacy_models(base_path: str) -> dict | None:
    """旧 LGBM bundle を初回のみロードしてキャッシュする。"""
    if base_path in _LEGACY_MODEL_CACHE:
        return _LEGACY_MODEL_CACHE[base_path]

    import joblib
    from .feature_engineering_custom import CustomFinancialFeatures
    from .industry_hybrid_model import IndustrySpecificHybridModel
    from .model import CreditScoringModel

    base = Path(base_path)
    if not base.exists():
        return None
    try:
        engine = CustomFinancialFeatures()
        industry_model = IndustrySpecificHybridModel()
        industry_model.industry_coefficients = joblib.load(base / "industry_coefficients.pkl")
        industry_model.industry_intercepts = joblib.load(base / "industry_intercepts.pkl")
        unified_ai = CreditScoringModel(model_type="lightgbm")
        unified_ai.load_model(str(base / "unified_ai_model.pkl"))
        scaler = joblib.load(base / "scaler.pkl")
        label_encoder = joblib.load(base / "label_encoder.pkl")
    except Exception as e:
        _logger.error("legacy model load failed: %s", e, exc_info=True)
        return None

    _LEGACY_MODEL_CACHE[base_path] = {
        "engine": engine,
        "industry_model": industry_model,
        "unified_ai": unified_ai,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }
    return _LEGACY_MODEL_CACHE[base_path]


def _load_rf_bundle(base_path: str | None = None) -> dict | None:
    """RandomForest bundle を初回のみロードしてキャッシュする。"""
    candidates = []
    env_path = (os.environ.get("LEASE_SCORING_RF_BUNDLE") or "").strip()
    if env_path:
        candidates.append(Path(env_path))
    if base_path:
        bp = Path(base_path)
        if bp.is_file():
            candidates.append(bp)
        else:
            candidates.append(bp / "ml_rf_v4.pkl")
    candidates.append(Path(__file__).resolve().parent.parent / "data" / "ml_rf_v4.pkl")
    candidates.append(Path(__file__).resolve().parent / "data" / "ml_rf_v4.pkl")

    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in _RF_BUNDLE_CACHE:
            return _RF_BUNDLE_CACHE[key]
        if not path.exists():
            continue
        try:
            import joblib
            bundle = joblib.load(path)
            if isinstance(bundle, dict) and "model" in bundle:
                _RF_BUNDLE_CACHE[key] = bundle
                return bundle
        except Exception as e:
            _logger.error("RF bundle load failed (%s): %s", path, e, exc_info=True)
    return None


# 円単位で渡す想定
def _ensure_float(v: Any) -> float:
    if v is None: return 0.0
    try: return float(v)
    except (TypeError, ValueError): return 0.0


def _safe_get_float(context: dict | None, key: str, default: float | None = None) -> float | None:
    if not context:
        return default
    if key not in context:
        return default
    try:
        v = context.get(key)
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_get_int(context: dict | None, key: str, default: int | None = None) -> int | None:
    if not context:
        return default
    if key not in context:
        return default
    try:
        v = context.get(key)
        if v is None:
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_get_str(context: dict | None, key: str, default: str = "") -> str:
    if not context:
        return default
    v = context.get(key)
    if v is None:
        return default
    return str(v)


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
    context: Optional[Dict[str, Any]] = None,
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
    except Exception as e:
        _logger.warning("predict_one: numpy/pandas が利用できません: %s", e)
        return None

    revenue = _ensure_float(revenue)
    total_assets = _ensure_float(total_assets)
    equity = _ensure_float(equity)
    if total_assets <= 0 or equity < 0:
        _logger.debug("predict_one: 入力値不正 total_assets=%s equity=%s", total_assets, equity)
        return None

    operating_profit = _ensure_float(operating_profit)
    net_income = _ensure_float(net_income)
    machinery_equipment = _ensure_float(machinery_equipment)
    other_fixed_assets = _ensure_float(other_fixed_assets)
    depreciation = _ensure_float(depreciation)
    rent_expense = _ensure_float(rent_expense)
    rf_bundle = _load_rf_bundle(base_path)
    legacy_models = _load_legacy_models(base_path or str(Path(__file__).resolve().parent / "models" / "industry_specific"))

    if rf_bundle is None and legacy_models is None:
        _logger.info("predict_one: RF/LGBM モデルがどちらもロードできません")
        return None

    # RF bundle 用の特徴量を組み立てる。training 時と同じ列を優先し、足りない値は imputer で補完する。
    ctx = context or {}
    features_map: dict[str, Any] = {}

    # 財務・比率
    gross_profit = _safe_get_float(ctx, "gross_profit", _ensure_float(ctx.get("gross_profit") if ctx else None))
    op_profit = _safe_get_float(ctx, "op_profit", operating_profit)
    ord_profit = _safe_get_float(ctx, "ord_profit", net_income)
    dep_expense = _safe_get_float(ctx, "dep_expense", _ensure_float(ctx.get("dep_expense") if ctx else None))
    gpm = _safe_get_float(ctx, "gpm")
    ord_margin = _safe_get_float(ctx, "ord_margin")
    net_margin = _safe_get_float(ctx, "net_margin")
    dep_ratio = _safe_get_float(ctx, "dep_ratio")
    bank_to_ns = _safe_get_float(ctx, "bank_to_ns")
    lease_to_ns = _safe_get_float(ctx, "lease_to_ns")
    mach_to_ns = _safe_get_float(ctx, "mach_to_ns")
    acq_to_ns = _safe_get_float(ctx, "acq_to_ns")
    op_margin = _safe_get_float(ctx, "op_margin")
    dep_to_loan = _safe_get_float(ctx, "dep_to_loan")
    acquisition_cost = _safe_get_float(ctx, "acquisition_cost")
    lease_term = _safe_get_float(ctx, "lease_term")
    contracts = _safe_get_float(ctx, "contracts")
    lease_asset_score = _safe_get_float(ctx, "lease_asset_score")
    base_rate = _safe_get_float(ctx, "base_rate")
    winning_spread = _safe_get_float(ctx, "winning_spread")

    # RF bundle の特徴量は 53 列。足りないものは None のままでも imputer が補完できる。
    features_map.update({
        "gross_profit": gross_profit,
        "op_profit": op_profit,
        "ord_profit": ord_profit,
        "net_income": net_income,
        "dep_expense": dep_expense,
        "depreciation": depreciation,
        "nenshu": revenue,
        "machines": machinery_equipment,
        "other_assets": other_fixed_assets,
        "rent": _safe_get_float(ctx, "rent"),
        "rent_expense": rent_expense,
        "bank_credit": _safe_get_float(ctx, "bank_credit"),
        "lease_credit": _safe_get_float(ctx, "lease_credit"),
        "gpm": gpm,
        "ord_margin": ord_margin,
        "net_margin": net_margin,
        "dep_ratio": dep_ratio,
        "bank_to_ns": bank_to_ns,
        "lease_to_ns": lease_to_ns,
        "mach_to_ns": mach_to_ns,
        "acq_to_ns": acq_to_ns,
        "op_margin": op_margin,
        "dep_to_loan": dep_to_loan,
        "acquisition_cost": acquisition_cost,
        "lease_term": lease_term,
        "contracts": contracts,
        "lease_asset_score": lease_asset_score,
        "industry": _safe_get_str(ctx, "industry", industry),
        "customer_type": 1 if _safe_get_str(ctx, "customer_type", "既存先") == "既存先" else 0,
        "main_bank": 1 if _safe_get_str(ctx, "main_bank", "非メイン先").startswith("メイン先") else 0,
        "competitor": 1 if _safe_get_str(ctx, "competitor", "競合なし") == "競合あり" else 0,
        "competitor_rate": _safe_get_float(ctx, "competitor_rate"),
        "grade": {"①1-3 (優良)": 1, "①1-3（優良）": 1, "②4-6 (標準)": 2, "②4-6（標準）": 2, "③7-9 (注意)": 3, "③7-9（注意）": 3, "④無格付": 4}.get(_safe_get_str(ctx, "grade", ""), 4),
        "contract_type": _safe_get_str(ctx, "contract_type", "一般"),
        "deal_source": _safe_get_str(ctx, "deal_source", "銀行紹介"),
        "sales_dept": _safe_get_str(ctx, "sales_dept", "未設定"),
        "base_rate": base_rate,
        "q_history": _safe_get_float(ctx, "q_history"),
        "q_stability": _safe_get_float(ctx, "q_stability"),
        "q_repayment": _safe_get_float(ctx, "q_repayment"),
        "q_future": _safe_get_float(ctx, "q_future"),
        "q_equip": _safe_get_float(ctx, "q_equip"),
        "q_mainbk": _safe_get_float(ctx, "q_mainbk"),
        "q_weighted": _safe_get_float(ctx, "q_weighted"),
        "sys_score": _safe_get_float(ctx, "sys_score"),
        "sys_score_b": _safe_get_float(ctx, "sys_score_b"),
        "sys_dscr": _safe_get_float(ctx, "sys_dscr"),
        "sys_op_margin": _safe_get_float(ctx, "sys_op_margin"),
        "sys_icr": _safe_get_float(ctx, "sys_icr"),
        "sys_approval": _safe_get_float(ctx, "sys_approval"),
        "sys_ind_score": _safe_get_float(ctx, "sys_ind_score"),
        "sys_bench": _safe_get_float(ctx, "sys_bench"),
        "winning_spread": winning_spread,
    })

    legacy_prob = None
    if legacy_models is not None:
        try:
            engine = legacy_models["engine"]
            industry_model = legacy_models["industry_model"]
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
            features["industry"] = _safe_get_str(ctx, "industry", industry)
            legacy_prob = float(industry_model.predict_by_industry(features)[0])
        except Exception as e:
            _logger.warning("predict_one: legacy model inference failed: %s", e)

    rf_approval_prob = None
    top5_reasons: List[str] = []
    if rf_bundle is not None:
        try:
            import numpy as np
            import pandas as pd

            model = rf_bundle["model"]
            feature_names = rf_bundle["feature_names"]
            imputer = rf_bundle["imputer"]
            encoders = rf_bundle["encoders"]

            def _enc_safe(key: str, val: str) -> int:
                le = encoders[key]
                default = le.classes_[0]
                if val not in le.classes_:
                    val = default
                return int(le.transform([val])[0])

            row = []
            for f in feature_names:
                if f == "industry":
                    row.append(_enc_safe("industry", _safe_get_str(ctx, "industry", industry)))
                elif f == "contract_type":
                    row.append(_enc_safe("contract_type", _safe_get_str(ctx, "contract_type", "一般")))
                elif f == "deal_source":
                    row.append(_enc_safe("deal_source", _safe_get_str(ctx, "deal_source", "銀行紹介")))
                elif f == "sales_dept":
                    row.append(_enc_safe("sales_dept", _safe_get_str(ctx, "sales_dept", "未設定")))
                else:
                    v = features_map.get(f)
                    row.append(np.nan if v is None else v)

            X_raw = np.array([row], dtype=float)
            X_imp = imputer.transform(X_raw)
            rf_approval_prob = float(model.predict_proba(X_imp)[0, 1])

            if hasattr(model, "feature_importances_"):
                fi = list(zip(feature_names, model.feature_importances_.tolist()))
                for f, _ in sorted(fi, key=lambda x: x[1], reverse=True)[:5]:
                    idx = feature_names.index(f)
                    val = row[idx]
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    top5_reasons.append(f"{f}: {float(val):.2f}" if isinstance(val, (int, float)) else f"{f}: {val}")
        except Exception as e:
            _logger.warning("predict_one: RF inference failed, fallback to legacy path: %s", e)

    if rf_approval_prob is None:
        # RF bundle が使えない場合は旧 LGBM bundle へフォールバック
        if legacy_models is None:
            return None
        models = legacy_models
        engine = models["engine"]
        industry_model = models["industry_model"]
        unified_ai = models["unified_ai"]
        scaler = models["scaler"]
        label_encoder = models["label_encoder"]

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
        features["industry"] = _safe_get_str(ctx, "industry", industry)

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
        rf_approval_prob = float(unified_ai.predict_proba(X_scaled)[0])
        if unified_ai.feature_importance is not None and len(unified_ai.feature_importance) > 0:
            for f in unified_ai.feature_importance.head(5)["feature"].tolist():
                if f in features.columns:
                    v = features[f].iloc[0]
                    if pd.notna(v):
                        top5_reasons.append(f"{f}: {v:.2f}")

        # 旧 bundle は「失注確率」を返していたので、そのまま risk に変換
        ai_prob = float(rf_approval_prob)
        hybrid_prob = 0.3 * (legacy_prob or 0.0) + 0.7 * ai_prob
        decision = "承認" if hybrid_prob < 0.5 else "否決"
        return {
            "legacy_prob": round(legacy_prob or 0.0, 4),
            "ai_prob": round(ai_prob, 4),
            "hybrid_prob": round(hybrid_prob, 4),
            "decision": decision,
            "top5_reasons": top5_reasons,
        }

    # RF bundle は「承認確率」を返すので、既存インターフェースに合わせてデフォルト率へ変換
    ai_prob = 1.0 - rf_approval_prob
    if legacy_prob is None:
        legacy_prob = ai_prob
    hybrid_prob = 0.3 * legacy_prob + 0.7 * ai_prob
    decision = "承認" if rf_approval_prob >= 0.5 else "否決"

    return {
        "legacy_prob": round(legacy_prob, 4),
        "ai_prob": round(ai_prob, 4),
        "hybrid_prob": round(hybrid_prob, 4),
        "decision": decision,
        "top5_reasons": top5_reasons,
    }
