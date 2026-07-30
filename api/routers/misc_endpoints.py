from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from api.cloudrun_writeback import record_cloudrun_input_event
from api.schemas import WorkLogRequest, WorkLogResponse

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)

router = APIRouter(tags=["misc"])

@router.get("/api/judgment-feedback/summary")
def judgment_feedback_summary_api():
    from judgment_feedback import get_judgment_feedback_summary

    return get_judgment_feedback_summary()


@router.get("/api/prompt-feedback/summary")
def prompt_feedback_summary_api():
    from prompt_feedback_metrics import DEFAULT_LOG_PATH, build_summary, load_jsonl

    rows = load_jsonl(DEFAULT_LOG_PATH)
    return {
        "source": str(DEFAULT_LOG_PATH),
        "summary": build_summary(rows),
    }


@router.get("/api/operational-trust/summary")
def operational_trust_summary_api():
    from operational_trust import build_operational_trust_summary

    vault = None
    try:
        from lease_news_digest import find_vault

        found = find_vault()
        vault = Path(found) if found else None
    except Exception:
        vault = None
    return build_operational_trust_summary(Path(_REPO_ROOT), vault=vault)


class JudgmentFeedbackReviewRequest(BaseModel):
    review_status: str


class JudgmentFeedbackCreateRequest(BaseModel):
    case_id: str
    model_decision: str
    human_decision: str
    reason: str
    source: str = "debate"
    score: Optional[float] = None
    input_snapshot: dict = Field(default_factory=dict)
    evidence_snapshot: dict = Field(default_factory=dict)


@router.post("/api/judgment-feedback")
def create_judgment_feedback_api(req: JudgmentFeedbackCreateRequest, background_tasks: BackgroundTasks):
    from judgment_feedback import record_judgment_feedback

    result = record_judgment_feedback(
        case_id=req.case_id,
        model_decision=req.model_decision,
        human_decision=req.human_decision,
        reason=req.reason,
        source=req.source,
        score=req.score,
        input_snapshot=req.input_snapshot,
        evidence_snapshot=req.evidence_snapshot,
    )
    if not result.get("success"):
        raise HTTPException(status_code=422, detail=result.get("error"))
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="judgment_feedback_created",
        surface="judgment_feedback",
        payload={**req.model_dump(), "record_id": result.get("record_id")},
    )
    return result


@router.get("/api/judgment-feedback/candidates")
def judgment_feedback_candidates_api(approved_only: bool = False):
    from judgment_feedback import load_judgment_training_candidates

    return {
        "items": load_judgment_training_candidates(approved_only=approved_only),
        "approved_only": approved_only,
    }


@router.post("/api/judgment-feedback/{record_id}/review")
def review_judgment_feedback_api(record_id: int, req: JudgmentFeedbackReviewRequest):
    from judgment_feedback import review_judgment_feedback

    result = review_judgment_feedback(record_id, req.review_status)
    if not result.get("success"):
        raise HTTPException(status_code=422, detail=result.get("error"))
    return result


# ── screening_outcomes エンドポイント（追加のみ、既存ルート不変）──────────────────

class OutcomeCreateRequest(BaseModel):
    case_id: str = Field(..., description="案件 ID（past_cases.id 等）")
    actual_status: str = Field(
        default="unknown",
        description="unknown / normal / late_30 / late_90 / default / completed",
    )
    screening_id: Optional[int] = Field(default=None, description="screening_records.id への参照")
    contract_date: Optional[str] = Field(default=None, description="成約日（YYYY-MM-DD）")
    scheduled_end_date: Optional[str] = Field(default=None, description="リース満了予定日")
    delinquent: int = Field(default=0, description="0=正常, 1=延滞・デフォルト")
    loss_given_default: Optional[float] = Field(default=None, description="実損額（円）")
    notes: Optional[str] = Field(default=None, description="備考")


class OutcomeResponse(BaseModel):
    id: int
    case_id: str
    screening_id: Optional[int]
    contract_date: Optional[str]
    scheduled_end_date: Optional[str]
    actual_status: str
    delinquent: int
    loss_given_default: Optional[float]
    checked_at: str
    notes: Optional[str]
    created_at: str
    updated_at: str


@router.post("/api/outcomes", response_model=OutcomeResponse)
def create_outcome(req: OutcomeCreateRequest):
    """審査後の追跡結果（支払状況等）を登録する。"""
    try:
        from api.add_outcomes_table import insert_outcome, get_outcome
        new_id = insert_outcome(
            case_id=req.case_id,
            actual_status=req.actual_status,
            screening_id=req.screening_id,
            contract_date=req.contract_date,
            scheduled_end_date=req.scheduled_end_date,
            delinquent=req.delinquent,
            loss_given_default=req.loss_given_default,
            notes=req.notes,
        )
        row = get_outcome(new_id)
        if row is None:
            raise HTTPException(status_code=500, detail="insert succeeded but row not found")
        return OutcomeResponse(**row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/outcomes", response_model=List[OutcomeResponse])
def get_outcomes(
    screening_id: Optional[int] = None,
    case_id: Optional[str] = None,
    actual_status: Optional[str] = None,
    limit: int = 100,
):
    """審査後追跡結果の一覧を取得する。screening_id / case_id / actual_status で絞り込み可能。"""
    try:
        from api.add_outcomes_table import list_outcomes
        rows = list_outcomes(
            screening_id=screening_id,
            case_id=case_id,
            actual_status=actual_status,
            limit=limit,
        )
        return [OutcomeResponse(**r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Fluid Pipeline エンドポイント（追記のみ、既存ルート不変）────────────────────

@router.post("/api/fluid/trigger")
def fluid_trigger(triggered_by: str = "manual"):
    """ドリフト検知→再学習→PDCA反省パイプラインを手動でバックグラウンド起動する。"""
    try:
        from api.fluid_pipeline import trigger_fluid_pipeline
        result = trigger_fluid_pipeline(triggered_by=triggered_by)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/fluid/status")
def fluid_status():
    """Fluid Pipeline の現在状態を返す。"""
    try:
        from api.fluid_pipeline import get_fluid_status
        return get_fluid_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/drift-stats")
def get_drift_stats():
    """スコアリングドリフト監視用統計を返す（REV-008）。"""
    import json as _json
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT timestamp, score, final_status, data FROM past_cases WHERE score IS NOT NULL ORDER BY timestamp ASC"
            ).fetchall()
    except Exception as e:
        logger.error("get_drift_stats DB error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    monthly: dict = {}
    score_by_status: dict = {"成約": [], "失注": []}
    all_scores: list = []

    for row in rows:
        ts = (row["timestamp"] or "")[:7]  # YYYY-MM
        score = row["score"]
        status = row["final_status"] or "未登録"
        all_scores.append(score)
        if ts:
            if ts not in monthly:
                monthly[ts] = {"month": ts, "count": 0, "won": 0, "lost": 0, "score_sum": 0.0, "score_won": [], "score_lost": []}
            monthly[ts]["count"] += 1
            monthly[ts]["score_sum"] += score
            if status == "成約":
                monthly[ts]["won"] += 1
                monthly[ts]["score_won"].append(score)
                score_by_status["成約"].append(score)
            elif status == "失注":
                monthly[ts]["lost"] += 1
                monthly[ts]["score_lost"].append(score)
                score_by_status["失注"].append(score)

    monthly_list = []
    for m in sorted(monthly.keys()):
        d = monthly[m]
        total_decided = d["won"] + d["lost"]
        monthly_list.append({
            "month": m,
            "count": d["count"],
            "won": d["won"],
            "lost": d["lost"],
            "win_rate": round(d["won"] / total_decided * 100, 1) if total_decided > 0 else None,
            "avg_score": round(d["score_sum"] / d["count"], 1) if d["count"] > 0 else None,
            "avg_score_won": round(sum(d["score_won"]) / len(d["score_won"]), 1) if d["score_won"] else None,
            "avg_score_lost": round(sum(d["score_lost"]) / len(d["score_lost"]), 1) if d["score_lost"] else None,
        })

    avg_won = round(sum(score_by_status["成約"]) / len(score_by_status["成約"]), 1) if score_by_status["成約"] else None
    avg_lost = round(sum(score_by_status["失注"]) / len(score_by_status["失注"]), 1) if score_by_status["失注"] else None
    separation = round(avg_won - avg_lost, 1) if avg_won is not None and avg_lost is not None else None

    bins = [0] * 10
    for s in all_scores:
        idx = min(9, int(s // 10))
        bins[idx] += 1
    score_dist = [{"range": f"{i*10}–{i*10+9}", "count": bins[i]} for i in range(10)]

    drift_alert = separation is not None and separation < 5.0

    return {
        "monthly": monthly_list,
        "summary": {
            "total": len(all_scores),
            "won_count": len(score_by_status["成約"]),
            "lost_count": len(score_by_status["失注"]),
            "avg_score_won": avg_won,
            "avg_score_lost": avg_lost,
            "separation": separation,
            "drift_alert": drift_alert,
        },
        "score_dist": score_dist,
    }


class CounterfactualRequest(BaseModel):
    case_id: str
    target_score: float = 70.0


@router.post("/api/counterfactual/analyze")
def analyze_counterfactual(req: CounterfactualRequest):
    """Counterfactual Explanation（REV-009）。指定案件の審査通過に必要な最小変更を計算する。"""
    import json as _json
    from scoring_core import run_quick_scoring

    # 案件データ取得
    try:
        with get_connection() as conn:
            row = conn.execute("SELECT data FROM past_cases WHERE id = ?", (req.case_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Case not found")
        case = _json.loads(row["data"] or "{}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("analyze_counterfactual DB error case_id=%s: %s", req.case_id, e)
        raise HTTPException(status_code=500, detail=str(e))

    inputs = case.get("inputs", {})
    result = case.get("result", {})
    if not inputs:
        raise HTTPException(status_code=422, detail="この案件には入力データが保存されていません")

    current_score = float(result.get("score") or run_quick_scoring(inputs).get("score", 0))
    target = req.target_score

    def score_with(overrides: dict) -> float:
        merged = {**inputs, **overrides}
        try:
            return float(run_quick_scoring(merged).get("score", 0))
        except Exception:
            return 0.0

    # 主要パラメータの現在値
    nenshu = max(1.0, float(inputs.get("nenshu") or 1))
    op_profit = float(inputs.get("op_profit") or 0)
    ord_profit = float(inputs.get("ord_profit") or 0)
    net_income = float(inputs.get("net_income") or 0)
    net_assets = float(inputs.get("net_assets") or 0)
    total_assets = max(1.0, float(inputs.get("total_assets") or 1))
    bank_credit = float(inputs.get("bank_credit") or 0)
    grade = str(inputs.get("grade") or "②4-6 (標準)")

    current_op_margin = op_profit / nenshu * 100
    current_eq_ratio = net_assets / total_assets * 100

    analyses = []

    # 1. 営業利益率改善 (op_profit増加)
    if current_score < target:
        for mult in [1.2, 1.5, 2.0, 3.0, 5.0, 10.0]:
            s = score_with({"op_profit": op_profit * mult})
            if s >= target:
                new_margin = (op_profit * mult) / nenshu * 100
                analyses.append({
                    "param": "op_profit",
                    "label": "営業利益率",
                    "current_display": f"{current_op_margin:.1f}%",
                    "required_display": f"{new_margin:.1f}%",
                    "current_value": current_op_margin,
                    "required_value": new_margin,
                    "change_pct": (mult - 1) * 100,
                    "achieved_score": round(s, 1),
                    "difficulty": "易" if mult <= 1.5 else "中" if mult <= 3.0 else "難",
                    "note": f"営業利益を現在の {mult:.0f}倍（+{(mult-1)*100:.0f}%）にする必要があります",
                })
                break

    # 2. 自己資本比率改善 (net_assets増加)
    if current_score < target:
        for add_ratio in [5, 10, 20, 35, 50]:
            new_net_assets = (current_eq_ratio + add_ratio) / 100 * total_assets
            s = score_with({"net_assets": new_net_assets})
            if s >= target:
                new_eq = new_net_assets / total_assets * 100
                analyses.append({
                    "param": "net_assets",
                    "label": "自己資本比率",
                    "current_display": f"{current_eq_ratio:.1f}%",
                    "required_display": f"{new_eq:.1f}%",
                    "current_value": current_eq_ratio,
                    "required_value": new_eq,
                    "change_pct": add_ratio,
                    "achieved_score": round(s, 1),
                    "difficulty": "易" if add_ratio <= 10 else "中" if add_ratio <= 25 else "難",
                    "note": f"自己資本比率を {add_ratio}pt 改善する必要があります",
                })
                break

    # 3. 売上高改善 (nenshu増加)
    if current_score < target:
        for mult in [1.3, 1.5, 2.0, 3.0]:
            s = score_with({"nenshu": nenshu * mult, "op_profit": op_profit * mult,
                            "ord_profit": ord_profit * mult, "net_income": net_income * mult})
            if s >= target:
                analyses.append({
                    "param": "nenshu",
                    "label": "売上高（按分増加）",
                    "current_display": f"{nenshu/1000:.1f}百万円",
                    "required_display": f"{nenshu*mult/1000:.1f}百万円",
                    "current_value": nenshu,
                    "required_value": nenshu * mult,
                    "change_pct": (mult - 1) * 100,
                    "achieved_score": round(s, 1),
                    "difficulty": "中" if mult <= 1.5 else "難",
                    "note": f"売上高を {mult:.1f}倍に成長させる必要があります（利益率維持想定）",
                })
                break

    # 4. 格付改善
    grade_map = [
        ("s", "S格（超優良）"),
        ("①", "①格（優良）"),
        ("②", "②格（標準）"),
        ("③", "③格（注意）"),
        ("④", "④格（要注意）"),
    ]
    if current_score < target:
        for gk, gl in grade_map:
            if grade.lower().startswith(gk):
                continue
            # 現在より良い格付のみ試す
            s = score_with({"grade": gk})
            if s >= target:
                analyses.append({
                    "param": "grade",
                    "label": "社内格付",
                    "current_display": grade,
                    "required_display": gl,
                    "current_value": 0,
                    "required_value": 0,
                    "change_pct": None,
                    "achieved_score": round(s, 1),
                    "difficulty": "中",
                    "note": f"格付を {grade} → {gl} に改善する必要があります",
                })
                break

    # 複合提案（op_profit + net_assets を半分ずつ改善）
    if current_score < target and len(analyses) < 2:
        for op_mult, eq_add in [(1.3, 5), (1.5, 10), (2.0, 15), (2.5, 20)]:
            new_na = (current_eq_ratio + eq_add) / 100 * total_assets
            s = score_with({"op_profit": op_profit * op_mult, "net_assets": new_na})
            if s >= target:
                analyses.append({
                    "param": "combined",
                    "label": "複合改善（利益率＋自己資本）",
                    "current_display": f"利益率{current_op_margin:.1f}% / 自己資本{current_eq_ratio:.1f}%",
                    "required_display": f"利益率{op_profit*op_mult/nenshu*100:.1f}% / 自己資本{(current_eq_ratio+eq_add):.1f}%",
                    "current_value": 0,
                    "required_value": 0,
                    "change_pct": None,
                    "achieved_score": round(s, 1),
                    "difficulty": "中",
                    "note": f"営業利益率+{(op_mult-1)*100:.0f}% かつ 自己資本比率+{eq_add}pt の複合改善",
                })
                break

    # スコア感度（各パラメータを±stepで変化させたスコア列）
    def op_sensitivity():
        data = []
        for pct in range(-50, 151, 10):
            v = op_profit * (1 + pct / 100) if op_profit != 0 else pct * nenshu / 10000
            s = score_with({"op_profit": v})
            data.append({
                "pct_change": pct,
                "op_margin": round(v / nenshu * 100, 2) if nenshu > 0 else 0,
                "score": round(s, 1),
            })
        return data

    def eq_sensitivity():
        data = []
        for add in range(-30, 51, 5):
            new_na = (current_eq_ratio + add) / 100 * total_assets
            s = score_with({"net_assets": max(0, new_na)})
            data.append({
                "eq_ratio": round(current_eq_ratio + add, 1),
                "score": round(s, 1),
            })
        return data

    return {
        "case_id": req.case_id,
        "current_score": round(current_score, 1),
        "target_score": target,
        "gap": round(target - current_score, 1),
        "current_metrics": {
            "op_margin": round(current_op_margin, 2),
            "eq_ratio": round(current_eq_ratio, 2),
            "nenshu": nenshu,
            "op_profit": op_profit,
            "net_assets": net_assets,
            "total_assets": total_assets,
            "grade": grade,
        },
        "counterfactuals": analyses,
        "op_sensitivity": op_sensitivity(),
        "eq_sensitivity": eq_sensitivity(),
    }


class RateEngineRequest(BaseModel):
    score: float
    term_months: int = 60
    asset_id: str = "other"
    grade: str = "②"
    lease_amount: float = 10000000
    year_month: str = ""


@router.post("/api/rate-engine/propose")
def propose_lease_rate(req: RateEngineRequest):
    """動的金利提案エンジン（REV-002）。借手スコア・物件種別・期間から最適金利を提案する。"""
    import datetime
    from base_rate_master import get_base_rate_by_term

    year_month = req.year_month or datetime.date.today().strftime("%Y-%m")
    term_months = max(12, min(120, req.term_months))
    term_years = term_months / 12

    base_rate = get_base_rate_by_term(year_month, term_months)
    if base_rate is None:
        for i in range(1, 7):
            prev_date = datetime.date.today().replace(day=1) - datetime.timedelta(days=30 * i)
            base_rate = get_base_rate_by_term(prev_date.strftime("%Y-%m"), term_months)
            if base_rate is not None:
                break
    if base_rate is None:
        base_rate = 2.0

    _asset_spreads: dict[tuple[str, int], float] = {
        ("medical", 1): 0.25, ("medical", 3): 0.30, ("medical", 5): 0.35, ("medical", 7): 0.40,
        ("it", 1): 0.35, ("it", 3): 0.50, ("it", 5): 0.65, ("it", 7): 0.75,
        ("pc", 1): 0.35, ("pc", 3): 0.50, ("pc", 5): 0.65, ("pc", 7): 0.75,
        ("vehicle", 1): 0.28, ("vehicle", 3): 0.32, ("vehicle", 5): 0.38, ("vehicle", 7): 0.45,
        ("car", 1): 0.28, ("car", 3): 0.32, ("car", 5): 0.38, ("car", 7): 0.45,
        ("machinery", 1): 0.30, ("machinery", 3): 0.38, ("machinery", 5): 0.45, ("machinery", 7): 0.52,
        ("construction", 1): 0.32, ("construction", 3): 0.40, ("construction", 5): 0.48, ("construction", 7): 0.55,
        ("solar", 1): 0.28, ("solar", 3): 0.35, ("solar", 5): 0.42, ("solar", 7): 0.50,
        ("other", 1): 0.32, ("other", 3): 0.40, ("other", 5): 0.50, ("other", 7): 0.58,
    }
    valid_terms = [1, 3, 5, 7]
    nearest_t = min(valid_terms, key=lambda t: abs(t - term_years))
    asset_id_lower = req.asset_id.lower()
    matched_prefix = next((k for (k, _) in _asset_spreads if asset_id_lower.startswith(k) and k != "other"), None)
    asset_spread = _asset_spreads.get((matched_prefix or "other", nearest_t), 0.45)

    _grade_spreads = {"s": -0.10, "①": -0.10, "a": 0.10, "②": 0.25, "b": 0.25,
                      "③": 0.55, "c": 0.55, "④": 0.90, "d": 0.90}
    grade_lower = req.grade.strip().lower()
    grade_spread = next((v for k, v in _grade_spreads.items() if grade_lower.startswith(k)), 0.30)

    score = max(0.0, min(100.0, req.score))
    if score >= 90: risk_adj = -0.10
    elif score >= 80: risk_adj = -0.05
    elif score >= 70: risk_adj = 0.00
    elif score >= 60: risk_adj = 0.15
    elif score >= 50: risk_adj = 0.30
    else: risk_adj = 0.50

    proposed_rate = round(max(0.5, base_rate + asset_spread + grade_spread + risk_adj), 4)

    monthly_rate = proposed_rate / 100 / 12
    amount = max(1.0, req.lease_amount)
    if monthly_rate > 0:
        monthly_payment = amount * monthly_rate / (1 - (1 + monthly_rate) ** (-term_months))
    else:
        monthly_payment = amount / term_months
    total_payment = monthly_payment * term_months
    total_interest = total_payment - amount

    sensitivity = []
    for s in range(max(0, int(score) - 30), min(101, int(score) + 35), 5):
        if s >= 90: r = -0.10
        elif s >= 80: r = -0.05
        elif s >= 70: r = 0.00
        elif s >= 60: r = 0.15
        elif s >= 50: r = 0.30
        else: r = 0.50
        sensitivity.append({
            "score": s,
            "rate": round(base_rate + asset_spread + grade_spread + r, 4),
            "is_current": abs(s - score) < 5,
        })

    return {
        "year_month": year_month,
        "proposed_rate": proposed_rate,
        "breakdown": {
            "base_rate": round(base_rate, 4),
            "asset_spread": round(asset_spread, 4),
            "grade_spread": round(grade_spread, 4),
            "risk_adjustment": round(risk_adj, 4),
        },
        "monthly_payment": round(monthly_payment),
        "total_payment": round(total_payment),
        "total_interest": round(total_interest),
        "term_months": term_months,
        "lease_amount": amount,
        "sensitivity": sensitivity,
    }


@router.get("/api/umap/embeddings")
def get_umap_embeddings():
    """UMAP 2D散布図用の学習データ埋め込みを返す（フロントエンドで一度キャッシュして使用）。"""
    embed_path = os.path.join(_REPO_ROOT, "data", "umap_embeddings.json")
    if not os.path.exists(embed_path):
        raise HTTPException(status_code=404, detail="umap_embeddings.json が見つかりません。train_umap_anomaly.py を実行してください。")
    with open(embed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@router.post("/api/work-log", response_model=WorkLogResponse)
def save_work_log(req: WorkLogRequest):
    """Codexスタイルの作業ログをmemory/とObsidianに保存する。"""
    import datetime as _dt
    import sys as _sys
    from pathlib import Path as _Path

    MEMORY_DIR = _Path.home() / ".claude" / "projects" / "-Users-kobayashiisaoryou-clawd-tune-lease-55" / "memory"
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pr_suffix = f"（PR #{req.pr}）" if req.pr else ""
    body_lines = [
        f"## 作業: {req.title}{pr_suffix}",
        "",
        "### 何をしたか",
        req.what,
    ]
    if req.why_hard:
        body_lines += ["", "### なぜ大変だったか", req.why_hard]
    if req.next_time:
        body_lines += ["", "### 次回どう切り分けるか", req.next_time]
    if req.lesson:
        body_lines += ["", "### 教訓", req.lesson]

    tag_str = ", ".join(req.tags)
    now_str = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    mem_content = (
        f"---\nname: work_log_{ts}\ndescription: 作業ログ: {req.title}\n"
        f"metadata:\n  type: project\n---\n\n"
        f"---\ndate: {now_str}\ntype: work_log\ntags: [{tag_str}]\n---\n\n"
        + "\n".join(body_lines) + "\n"
    )
    mem_path = MEMORY_DIR / f"work_log_{ts}.md"
    mem_path.write_text(mem_content, encoding="utf-8")

    try:
        from mobile_app.obsidian_bridge import append_work_log
        obs_result = append_work_log(
            title=req.title, what=req.what, why_hard=req.why_hard,
            next_time=req.next_time, lesson=req.lesson, pr=req.pr, tags=req.tags,
        )
    except Exception as e:
        obs_result = {"status": "error", "reason": str(e)}

    return WorkLogResponse(memory_path=str(mem_path), obsidian=obs_result)



# ── 世界認識 通知ステータス ────────────────────────────────────────────────
_WORLD_VIEW_MIND_PATH = Path(_REPO_ROOT) / "data" / "mind.json"
_WORLD_VIEW_NOTIFIED_PATH = Path(_REPO_ROOT) / "data" / "world_view_notified.json"


def _wv_load_mind() -> dict:
    try:
        d = json.loads(_WORLD_VIEW_MIND_PATH.read_text(encoding="utf-8"))
        return d.get("world_view", {}) if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _wv_load_notified() -> dict:
    try:
        return json.loads(_WORLD_VIEW_NOTIFIED_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"acked_at": ""}


@router.get("/api/world-view-status")
def get_world_view_status():
    world_view = _wv_load_mind()
    updated_at = str(world_view.get("updated_at", "")).strip()
    acked_at = str(_wv_load_notified().get("acked_at", "")).strip()
    has_update = bool(updated_at and updated_at > acked_at)
    return {
        "has_update": has_update,
        "updated_at": updated_at,
        "summary": str(world_view.get("summary", "")).strip(),
    }


@router.post("/api/world-view-ack")
def post_world_view_ack():
    world_view = _wv_load_mind()
    updated_at = str(world_view.get("updated_at", "")).strip()
    _WORLD_VIEW_NOTIFIED_PATH.write_text(
        json.dumps({"acked_at": updated_at}, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"status": "acked", "acked_at": updated_at}
