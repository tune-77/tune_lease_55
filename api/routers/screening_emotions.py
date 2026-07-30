"""ネットワークリスク・最新審査・感情エンドポイントルーター (REV-234 Phase13)"""
from __future__ import annotations
import logging as _lg
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.db_connection import get_connection
from lease_news_digest import get_latest_lease_news_focus, record_lease_news_view

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_OBSIDIAN_VAULT_PATH: str = os.environ.get("OBSIDIAN_VAULT_PATH", "") or os.environ.get("OBSIDIAN_VAULT", "")
_logger = _lg.getLogger(__name__)
router = APIRouter(tags=["screening-emotions"])


# ── サプライチェーン波及リスク ────────────────────────────────────────────────

@router.get("/api/analysis/network_risk")
def api_network_risk(industry: str = ""):
    """業種コードまたは業種名からサプライチェーン波及リスクを計算する"""
    import sys as _sys
    _root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    try:
        from components.graph_risk import GraphRiskEngine
        engine = GraphRiskEngine()
        result = engine.calculate_network_risk(industry)
        sim = engine.run_scenario_simulation(industry, n_simulations=300)
        return {
            "network_risk_pct": round(result.get("network_risk_score", 0.05) * 100, 1),
            "base_risk_pct": round(result.get("base_risk", 0.05) * 100, 1),
            "impacted_by": result.get("impacted_by", [])[:5],
            "sim_mean_pct": round(sim.get("mean_risk", 0.05) * 100, 1),
            "sim_var95_pct": round(sim.get("max_risk_95", 0.05) * 100, 1),
            "target_industry": result.get("target_industry", industry),
        }
    except Exception as e:
        return {"network_risk_pct": 5.0, "base_risk_pct": 5.0, "impacted_by": [],
                "sim_mean_pct": 5.0, "sim_var95_pct": 10.0, "target_industry": industry,
                "error": str(e)}


# ── 最新審査データ ─────────────────────────────────────────────────────────────

@router.get("/api/latest-screening")
def get_latest_screening():
    """
    直近の審査データを返す。
    screening_records の最新スコア + past_cases の最新フォーム入力値を合成して、
    debate ページの初期値として使用する。
    """
    import json as _json
    import os as _os

    defaults = {
        "score": 52,
        "company_name": "",
        "industry_major": "製造業",
        "nenshu": 0,
        "op_margin_pct": 0,
        "equity_ratio": 0,
        "bank_credit": 0,
        "lease_credit": 0,
        "asset_name": "",
        "lease_amount": 0,
        "news_focus": [],
        "news_focus_summary": "",
        "news_focus_tag_summary": "",
        "news_focus_note_path": "",
        "news_focus_note_date": "",
    }

    def _first_non_empty(*values):
        for value in values:
            if value not in (None, "", [], {}):
                return value
        return None

    def _to_million(value):
        try:
            return round(float(value) / 1000, 2)
        except Exception:
            return 0

    def _safe_float(value):
        try:
            return float(value)
        except Exception:
            return 0.0

    try:
        with get_connection() as conn:

            # screening_records から最新スコアを取得
            try:
                sr = conn.execute(
                    "SELECT total_score, input_snapshot FROM screening_records ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if sr:
                    defaults["score"] = round(float(sr["total_score"]), 1)
                    if sr["input_snapshot"]:
                        snap = _json.loads(sr["input_snapshot"])
                        for key in ("company_name", "industry_major", "asset_name"):
                            if snap.get(key):
                                defaults[key] = snap[key]
            except Exception:
                pass

            # past_cases から最新のフォーム入力値を取得（千円→百万円 変換）
            try:
                pc = conn.execute(
                    "SELECT data FROM past_cases ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if pc and pc["data"]:
                    d = _json.loads(pc["data"])

                    inputs = d.get("inputs") if isinstance(d.get("inputs"), dict) else {}
                    result = d.get("result") if isinstance(d.get("result"), dict) else {}

                    company_name = _first_non_empty(d.get("company_name"), inputs.get("company_name"))
                    if company_name:
                        defaults["company_name"] = company_name

                    industry_major = _first_non_empty(
                        d.get("selected_major"),
                        d.get("industry_major"),
                        inputs.get("industry_major"),
                    )
                    if industry_major:
                        defaults["industry_major"] = industry_major

                    nenshu_raw = _safe_float(_first_non_empty(inputs.get("nenshu"), d.get("nenshu")))
                    op_profit_raw = _safe_float(_first_non_empty(inputs.get("op_profit"), inputs.get("rieki"), d.get("rieki")))
                    net_assets_raw = _safe_float(_first_non_empty(inputs.get("net_assets"), d.get("net_assets")))
                    total_assets_raw = _safe_float(_first_non_empty(inputs.get("total_assets"), d.get("total_assets")))

                    if nenshu_raw > 0:
                        defaults["nenshu"] = _to_million(nenshu_raw)
                    if nenshu_raw > 0:
                        defaults["op_margin_pct"] = round(op_profit_raw / nenshu_raw * 100, 1)
                    if total_assets_raw > 0:
                        defaults["equity_ratio"] = round(net_assets_raw / total_assets_raw * 100, 1)

                    bank_credit_raw = _safe_float(_first_non_empty(inputs.get("bank_credit"), d.get("bank_credit")))
                    lease_credit_raw = _safe_float(_first_non_empty(inputs.get("lease_credit"), d.get("lease_credit")))
                    acquisition_cost_raw = _safe_float(_first_non_empty(inputs.get("acquisition_cost"), d.get("acquisition_cost")))
                    if bank_credit_raw:
                        defaults["bank_credit"] = _to_million(bank_credit_raw)
                    if lease_credit_raw:
                        defaults["lease_credit"] = _to_million(lease_credit_raw)
                    if acquisition_cost_raw:
                        defaults["lease_amount"] = _to_million(acquisition_cost_raw)

                    asset_name = _first_non_empty(
                        inputs.get("asset_name"),
                        inputs.get("selected_asset_id"),
                        d.get("asset_name"),
                    )
                    if asset_name:
                        defaults["asset_name"] = asset_name

                    if result.get("score") is not None:
                        try:
                            defaults["score"] = round(float(result["score"]), 1)
                        except Exception:
                            pass
            except Exception:
                pass

    except Exception as e:
        _logger.error("get_latest_screening DB error: %s", e)

    try:
        focus = get_latest_lease_news_focus()
        if focus.available:
            defaults["news_focus"] = list(focus.focus_lines)
            defaults["news_focus_summary"] = focus.headline
            defaults["news_focus_tag_summary"] = focus.tag_summary
            defaults["news_focus_note_path"] = focus.note_path
            defaults["news_focus_note_date"] = focus.note_date
            try:
                record_lease_news_view(focus.note_date or "", focus.note_path, focus.tag_summary)
            except Exception as _view_err:
                print(f"[API] lease news view metric failed: {_view_err}")
    except Exception as e:
        print(f"[API] latest lease news focus load failed: {e}")

    return defaults


# ── 感情時系列 エンドポイント（REV-075）───────────────────────────────────────

@router.post("/api/intelligence/emotions/record")
def record_emotion_history_api():
    """現在の感情スコアをDBに保存する（1日1回、当日分が既にあればスキップ）。"""
    try:
        from lease_intelligence_mind import (
            _derive_complex_emotions,
            load_lease_intelligence_mind,
        )
        from lease_news_digest import find_vault
        from api.database import record_emotion_snapshot

        vault = find_vault()
        if not vault:
            raise HTTPException(status_code=503, detail="Obsidian Vaultが見つかりません")
        state = load_lease_intelligence_mind(vault)
        emotions = _derive_complex_emotions(state.get("mood", {}))
        scores = {e["key"]: float(e["score"]) for e in emotions}
        dominant = emotions[0]["key"] if emotions else ""
        row_id, inserted = record_emotion_snapshot(scores, dominant)
        return {"id": row_id, "inserted": inserted, "scores": scores}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/intelligence/emotions/history")
def get_emotion_history_api(days: int = 30):
    """過去N日分の7軸感情スコアを時系列で返す。"""
    try:
        from api.database import get_emotion_history
        return {"days": days, "history": get_emotion_history(days)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/intelligence/emotions/summary")
def get_emotion_summary_api(days: int = 30):
    """期間内の各軸の平均・最大・最小・標準偏差を返す。"""
    try:
        from api.database import get_emotion_summary
        return get_emotion_summary(days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── REV-076: 感情レーダーチャート フィードバック ─────────────────────────────

class EmotionFeedbackRequest(BaseModel):
    rating: str  # 'good' | 'needs_improvement'
    comment: Optional[str] = None
    emotion_category: Optional[str] = None


def _append_emotion_feedback_to_obsidian(rating: str, comment: Optional[str], emotion_category: Optional[str]) -> dict:
    """フィードバックを Obsidian の感情可視化フィードバック.md に追記する。"""
    import datetime as _dt
    vault_raw = _OBSIDIAN_VAULT_PATH or os.environ.get("OBSIDIAN_VAULT") or os.environ.get("OBSIDIAN_VAULT_PATH") or ""
    if not vault_raw:
        return {"status": "skipped", "reason": "obsidian_vault_not_configured"}
    vault = Path(vault_raw).expanduser().resolve()
    if not (vault / ".obsidian").exists():
        return {"status": "skipped", "reason": "obsidian_vault_not_found"}

    now = _dt.datetime.now()
    rel = Path("Projects") / "tune_lease_55" / "Lease Intelligence" / "感情可視化フィードバック.md"
    path = (vault / rel).resolve()
    if vault not in path.parents and path != vault:
        return {"status": "error", "reason": "unsafe_path"}
    path.parent.mkdir(parents=True, exist_ok=True)

    rating_label = "👍 わかりやすい" if rating == "good" else "📝 意見あり"
    lines = [f"\n## {now.strftime('%Y-%m-%d %H:%M')} — {rating_label}"]
    if emotion_category:
        lines.append(f"- 感情軸: {emotion_category}")
    if comment:
        lines.append(f"- コメント: {comment}")

    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return {"status": "ok", "path": str(rel)}


@router.post("/api/intelligence/emotions/feedback")
def post_emotion_feedback(req: EmotionFeedbackRequest):
    if req.rating not in ("good", "needs_improvement"):
        raise HTTPException(status_code=422, detail="rating は 'good' または 'needs_improvement' で指定してください")
    from api.database import save_emotion_feedback
    record_id = save_emotion_feedback(req.rating, req.comment, req.emotion_category)
    obsidian_result = _append_emotion_feedback_to_obsidian(req.rating, req.comment, req.emotion_category)
    return {"status": "saved", "id": record_id, "obsidian": obsidian_result}


@router.get("/api/intelligence/emotions/feedback")
def get_emotion_feedback(resolved: Optional[bool] = None):
    from api.database import get_emotion_feedbacks
    items = get_emotion_feedbacks(resolved=resolved)
    return {"items": items, "total": len(items)}
