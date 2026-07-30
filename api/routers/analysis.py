"""高度分析ルーター (REV-234 Phase4)"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["analysis"])

# =============================================================================
# 高度分析 API (Phase 15.3: 計数分析・ログ・マスタ)
# =============================================================================

# ── 履歴分析・ダッシュボード
@router.get("/api/analysis/contract_drivers")
def api_contract_drivers():
    from analysis_regression import run_contract_driver_analysis
    res = run_contract_driver_analysis()
    if res is None:
        raise HTTPException(status_code=400, detail="Not enough data (minimum 5 closed cases required).")
    
    # JSON直列化のための細かな変換処理
    # closed_cases は大きすぎるので概要だけ返す
    return {
        "closed_count": res.get("closed_count"),
        "avg_financials": res.get("avg_financials"),
        "tag_ranking": res.get("tag_ranking"),
        "top3_drivers": res.get("top3_drivers"),
        "qualitative_summary": {
            "avg_weighted": res.get("qualitative_summary", {}).get("avg_weighted"),
            "n_with_qual": res.get("qualitative_summary", {}).get("n_with_qual"),
            "rank_distribution": res.get("qualitative_summary", {}).get("rank_distribution"),
        } if res.get("qualitative_summary") else None
    }

# ── 定量要因分析
def _get_gemini_api_key() -> str:
    # 共通実装へ委譲（環境変数 → secrets.toml 直接パース、Streamlit非依存）
    from api.secret_access import get_gemini_api_key

    return get_gemini_api_key()


def _metric_line(name: str, result: Dict[str, Any]) -> str:
    acc = result.get(f"accuracy_{name}")
    auc = result.get(f"auc_{name}")
    acc_text = f"{acc:.3f}" if isinstance(acc, (int, float)) else "N/A"
    auc_text = f"{auc:.3f}" if isinstance(auc, (int, float)) else "N/A"
    return f"{name.upper()}: accuracy={acc_text}, auc={auc_text}"


def _top_factor_text(items: Any, limit: int = 5, absolute: bool = False) -> str:
    if not isinstance(items, list):
        return "N/A"
    cleaned = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            cleaned.append((str(item[0]), float(item[1])))
        except Exception:
            continue
    if absolute:
        cleaned.sort(key=lambda x: abs(x[1]), reverse=True)
    else:
        cleaned.sort(key=lambda x: x[1], reverse=True)
    return ", ".join(f"{name}={value:.3f}" for name, value in cleaned[:limit]) or "N/A"


def _generate_quantitative_gemini_comment(result: Dict[str, Any]) -> Dict[str, str]:
    api_key = _get_gemini_api_key()
    if not api_key:
        return {
            "text": "Gemini APIキーが未設定のため、AI所見を生成できませんでした。",
            "source": "unavailable",
        }

    prompt = f"""
あなたはリース審査の分析担当です。次の定量要因・ML分析結果を読み、営業担当向けに2〜3行の日本語で要点を書いてください。
断定しすぎず、ロジスティック回帰・ランダムフォレスト・LGBM・アンサンブルの複合的な見方にしてください。箇条書きは禁止です。

件数: {result.get("n_cases")}、成約: {result.get("n_positive")}、失注: {result.get("n_negative")}
モデル指標: {_metric_line("lr", result)} / {_metric_line("rf", result)} / {_metric_line("lgb", result)}
アンサンブル: accuracy={result.get("accuracy_ensemble") if isinstance(result.get("accuracy_ensemble"), (int, float)) else "N/A"}, auc={result.get("auc_ensemble") if isinstance(result.get("auc_ensemble"), (int, float)) else "N/A"}, LR比率={result.get("ensemble_alpha") if isinstance(result.get("ensemble_alpha"), (int, float)) else "N/A"}
現時点の最良モデル: {result.get("best_auc_model") or "N/A"} / AUC={result.get("best_auc_value") if isinstance(result.get("best_auc_value"), (int, float)) else "N/A"}
ロジスティック回帰の主な係数: {_top_factor_text(result.get("lr_coef"), absolute=True)}
RandomForestの主な重要度: {_top_factor_text(result.get("rf_importance"))}
LGBMの主な重要度: {_top_factor_text(result.get("lgb_importance"))}
""".strip()

    try:
        import requests

        url = _gemini_generate_url()
        response = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            headers={"x-goog-api-key": api_key},
            timeout=25,
        )
        response.raise_for_status()
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return {"text": text, "source": "gemini"}
    except Exception as e:
        return {
            "text": f"Gemini APIへの接続に失敗したため、AI所見を生成できませんでした: {e}",
            "source": "error",
        }


@router.get("/api/analysis/quantitative")
def api_quantitative():
    from analysis_regression import run_quantitative_contract_analysis
    res = run_quantitative_contract_analysis()
    if res is None:
        raise HTTPException(status_code=400, detail="Not enough data (minimum 50 cases required).")
    res["gemini_comment"] = _generate_quantitative_gemini_comment(res)
    return res

# ── 定性要因分析
@router.post("/api/analysis/qualitative")
def api_qualitative():
    from analysis_regression import run_qualitative_contract_analysis
    from category_config import QUALITATIVE_SCORING_CORRECTION_ITEMS
    res = run_qualitative_contract_analysis(QUALITATIVE_SCORING_CORRECTION_ITEMS)
    if res is None:
        raise HTTPException(status_code=400, detail="Not enough data.")
    return res



@router.get("/api/logs/app")
def get_app_log_lines():
    log_path = "streamlit.log"
    if not os.path.exists(log_path):
        return {"logs": []}
    with open(log_path, "r", encoding="utf-8") as f:
        # Return last 100 lines
        lines = f.readlines()
        return {"logs": [line.strip() for line in lines[-100:]]}



# ── 競合関係グラフ
@router.get("/api/analysis/competitor_graph")
def api_competitor_graph():
    from components.graph_view import build_graph_data
    try:
        data = build_graph_data()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ── システム管理・分析系API v2
@router.get("/api/analysis/auto_optimizer_status")
def get_auto_optimizer_status():
    from auto_optimizer import get_training_status
    try:
        return get_training_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/model_review/status")
def get_model_review_status_api():
    from model_review_hooks import get_model_review_hook_status
    try:
        return get_model_review_hook_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/model_review/run")
def run_model_review_hooks_api():
    from model_review_hooks import run_model_review_hooks
    try:
        res = run_model_review_hooks(force=True)
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/analysis/pdca_reflection")
def get_pdca_reflection():
    from llm_pdca_reflection import load_pdca_rules
    try:
        return load_pdca_rules()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/analysis/math_proposals")
def get_math_proposals():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prop_path = os.path.join(base_dir, "data", "math_proposals.json")
    if not os.path.exists(prop_path):
        return {"proposals": []}
    try:
        with open(prop_path, "r", encoding="utf-8") as f:
            props = json.load(f)
        return {"proposals": props}
    except Exception as e:
        return {"proposals": []}

@router.get("/api/analysis/coeff_history")
def get_coeff_history():
    from data_cases import load_coeff_history
    try:
        return {"history": load_coeff_history()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/analysis/app_logs")
def get_app_logs(lines: int = 100):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(base_dir, "logs", "app.log"),
        os.path.join(base_dir, "app.log"),
        os.path.join(base_dir, "streamlit.log"),
    ]
    log_path = next((p for p in paths if os.path.exists(p)), None)
    if not log_path:
        return {"logs": "Log file not found."}
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return {"logs": "".join(all_lines[-lines:]), "path": log_path}
    except Exception as e:
        return {"logs": f"Error reading log: {e}"}




@router.post("/api/analysis/run_auto_optimization")
def run_auto_opt():
    from auto_optimizer import run_auto_optimization
    try:
        res = run_auto_optimization(force=True)
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/analysis/run_pdca")
def run_pdca_api(max_cases: int = 20):
    from llm_pdca_reflection import run_monthly_pdca_reflection
    try:
        res = run_monthly_pdca_reflection(force=True, max_cases=max_cases)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/api/analysis/status_summary")
def get_analysis_status_summary():
    from auto_optimizer import get_training_status
    try:
        s = get_training_status()
        # 追加情報の統合（今何件、あと何件）
        res = {
            "auto_opt": {
                "count": s["count"],
                "min": 50,
                "rem": s["next_trigger"] if s["count"] < 50 else max(0, 20 - (s["count"] - s["last_trained_count"]))
            },
            "quant_ml": {
                "count": s["count"],
                "min": 50,
                "rem": max(0, 50 - s["count"])
            },
            "qual_contrib": {
                "count": s["count"],
                "min": 50,
                "rem": max(0, 50 - s["count"])
            }
        }
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


