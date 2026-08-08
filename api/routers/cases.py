"""
api/routers/cases.py  – REV-234 Phase 5
Cases エンドポイント（一覧・業種別・営業部別成約率・詳細・進捗スタンプ）

Extracted from api/main.py.  以下のエンドポイントを含む:
  GET  /api/cases                       – list_cases
  GET  /api/cases/industry-winrate      – get_industry_winrate
  GET  /api/cases/sales-dept-winrate    – get_sales_dept_winrate
  GET  /api/cases/{case_id}             – get_case_detail
  POST /api/cases/progress-stamp        – stamp_case_progress
  POST /api/deal/closure-probability    – calc_deal_closure_probability

注: patch / delete / register / pending の各エンドポイントは
    main.py 内部ヘルパー (_git_push_db 等) への依存が深いため main.py に残す。
"""
from __future__ import annotations

import logging
import os
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.db_connection import current_backend, get_connection
from api.schemas import DealClosureRequest, DealClosureResponse
from scoring.deal_closure_engine import build_features, build_features_from_deltas, compute_closure_likelihood

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cases"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _db_available() -> bool:
    """Cloud SQL ではローカル SQLite ファイルがなくても DB 利用可能とみなす。"""
    from runtime_paths import get_db_path  # type: ignore[import]
    return current_backend() == "postgresql" or os.path.exists(get_db_path())


SERVICE_GENERAL_LABEL = "サービス業全般"
_SERVICE_GENERAL_ALIASES = {
    "91 職業紹介・労働者派遣業",
    "R サービス業(他に分類されないもの)",
}


def _normalize_industry_for_stats(industry: str) -> str:
    label = str(industry or "").strip()
    if label in _SERVICE_GENERAL_ALIASES:
        return SERVICE_GENERAL_LABEL
    return label


def _compute_case_closure_probability(case_data: dict) -> float | None:
    reg = case_data.get("registration_date")
    est = case_data.get("estimate_sent_date")
    resp = case_data.get("customer_response_date")
    if not (reg and est and resp):
        return None
    features = build_features(registration_date=reg, estimate_sent_date=est, customer_response_date=resp)
    prob = compute_closure_likelihood(features, has_cash_data=bool(case_data.get("has_cash_data", True)))
    return float(prob)


# ── Models ────────────────────────────────────────────────────────────────────

class CaseProgressStampRequest(BaseModel):
    case_id: str
    event_type: str  # estimate_sent | customer_response
    occurred_at: str | None = None  # YYYY-MM-DD


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/api/cases")
def list_cases(limit: int = 30, offset: int = 0, sort: str = "desc"):
    """過去案件一覧 (limit/offset/sort 対応)"""
    import json  # noqa: F401

    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    order = "DESC" if sort.lower() != "asc" else "ASC"
    rows = []
    try:
        with get_connection() as conn:
            res = conn.execute(
                f"SELECT id, timestamp, industry_sub, score, final_status, "
                f"json_extract(data,'$.company_name') AS company_name, "
                f"json_extract(data,'$.company_no')   AS company_no, "
                f"json_extract(data,'$.judgment')     AS judgment, "
                f"COALESCE(json_extract(data,'$._source'), 'past_cases') AS source "
                f"FROM past_cases ORDER BY timestamp {order} LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            for r in res:
                rows.append(dict(r))
    except Exception as e:
        logger.error("list_cases DB error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return rows


@router.get("/api/cases/pending")
def get_pending_cases():
    """未登録案件一覧。

    main.py 側にも互換エンドポイントがあるが、この router の
    `/api/cases/{case_id}` より後に登録されるため、静的パスをここで先に受ける。
    """
    rows = []

    try:
        with get_connection() as conn:
            res = conn.execute(
                "SELECT id, timestamp, industry_sub, score, data "
                "FROM past_cases "
                "WHERE COALESCE(NULLIF(final_status, ''), '未登録') IN ('未登録', '稟議中', 'スコアリングのみ') "
                "ORDER BY timestamp DESC LIMIT 50"
            ).fetchall()
            for r in res:
                try:
                    data = json.loads(r["data"] or "{}")
                except Exception:
                    data = {}
                inputs = data.get("inputs") if isinstance(data.get("inputs"), dict) else {}
                result = data.get("result") if isinstance(data.get("result"), dict) else {}
                rows.append({
                    "id": str(r["id"]),
                    "company_no": data.get("company_no") or inputs.get("company_no") or "",
                    "company_name": data.get("company_name") or inputs.get("company_name") or "名称未設定",
                    "timestamp": r["timestamp"],
                    "score": r["score"] if r["score"] not in (None, "") else result.get("score", result.get("score_base")),
                    "hantei": result.get("hantei") or data.get("hantei") or "",
                    "industry": r["industry_sub"] or data.get("industry_sub") or inputs.get("industry_sub") or data.get("industry_major") or inputs.get("industry_major") or "",
                    "registration_date": data.get("registration_date") or (r["timestamp"] or "")[:10],
                    "estimate_sent_date": data.get("estimate_sent_date") or (r["timestamp"] or "")[:10],
                    "final_result_date": data.get("final_result_date"),
                    "_source": "past_cases",
                })
    except Exception as e:
        logger.error("get_pending_cases DB error: %s", e)

    rows.extend(_list_cloudrun_score_pending_cases(limit=50))
    rows.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    return rows[:80]


def _list_cloudrun_score_pending_cases(limit: int = 50) -> list[dict]:
    try:
        from api.cloudrun_pending_cases import cloudrun_score_pending_item
        from api.routers.feedback_loop import (
            _CLOUDRUN_RETURN_DB,
            _cloudrun_return_table_exists,
            _connect_cloudrun_return_db,
            _ensure_cloudrun_return_review_schema,
        )

        if not _CLOUDRUN_RETURN_DB.exists():
            return []
        with _connect_cloudrun_return_db() as conn:
            _ensure_cloudrun_return_review_schema(conn)
            if not _cloudrun_return_table_exists(conn, "cloudrun_score_inputs"):
                return []
            rows = conn.execute(
                """
                SELECT *
                  FROM cloudrun_score_inputs
                 WHERE COALESCE(NULLIF(return_review_status, ''), 'candidate') != 'rejected'
                   AND COALESCE(return_registered_case_id, '') = ''
                 ORDER BY COALESCE(NULLIF(created_at, ''), '1970-01-01') DESC, id DESC
                 LIMIT ?
                """,
                (max(1, min(int(limit or 50), 200)),),
            ).fetchall()
        return [cloudrun_score_pending_item(dict(row)) for row in rows]
    except Exception as exc:
        logger.warning("cloudrun score pending list skipped: %s", exc)
        return []


@router.get("/api/cases/industry-winrate")
def get_industry_winrate():
    """業種別成約率を past_cases から集計して返す（REV-055/117~119）。"""
    if not _db_available():
        return []
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT industry_sub, final_status, COUNT(*) FROM past_cases "
            "WHERE final_status IS NOT NULL AND final_status != '' "
            "GROUP BY industry_sub, final_status"
        )
        rows = cur.fetchall()
    _SUCCESS = {"成約", "検収完了"}
    _FAILURE = {"失注"}
    agg: dict = {}
    for industry, status, cnt in rows:
        if not industry or industry == "0":
            continue
        industry = _normalize_industry_for_stats(industry)
        d = agg.setdefault(industry, {"won": 0, "lost": 0})
        if status in _SUCCESS:
            d["won"] += cnt
        elif status in _FAILURE:
            d["lost"] += cnt
    result = []
    total_won = sum(v["won"] for v in agg.values())
    total_lost = sum(v["lost"] for v in agg.values())
    total_all = total_won + total_lost
    overall_rate = round(total_won / total_all * 100, 1) if total_all > 0 else 0
    for industry, d in agg.items():
        total = d["won"] + d["lost"]
        if total == 0:
            continue
        rate = round(d["won"] / total * 100, 1)
        result.append({
            "industry": industry,
            "won": d["won"],
            "lost": d["lost"],
            "total": total,
            "win_rate": rate,
            "diff": round(rate - overall_rate, 1),
        })
    result.sort(key=lambda x: x["total"], reverse=True)
    return {"items": result, "overall_rate": overall_rate, "total_won": total_won, "total_lost": total_lost}


@router.get("/api/cases/sales-dept-winrate")
def get_sales_dept_winrate():
    """営業部別成約率を集計して返す（REV-112）。"""
    if not _db_available():
        return {"items": [], "overall_rate": 0.0, "total_won": 0, "total_lost": 0}
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT sales_dept,
                   SUM(CASE WHEN final_status IN ('成約','検収完了') THEN 1 ELSE 0 END) as won,
                   SUM(CASE WHEN final_status = '失注' THEN 1 ELSE 0 END) as lost,
                   COUNT(*) as total,
                   ROUND(AVG(score), 1) as avg_score
            FROM past_cases
            WHERE sales_dept NOT IN ('', '0', '未設定')
              AND final_status IN ('成約','検収完了','失注')
            GROUP BY sales_dept
            ORDER BY total DESC
        """)
        rows = cur.fetchall()
    total_won = sum(r[1] for r in rows)
    total_lost = sum(r[2] for r in rows)
    overall_rate = round(total_won / (total_won + total_lost) * 100, 1) if (total_won + total_lost) > 0 else 0.0
    result = []
    for dept, won, lost, total, avg_score in rows:
        rate = round(won / (won + lost) * 100, 1) if (won + lost) > 0 else 0.0
        result.append({
            "dept": dept,
            "won": won,
            "lost": lost,
            "total": total,
            "win_rate": rate,
            "avg_score": avg_score or 0.0,
            "diff": round(rate - overall_rate, 1),
        })
    return {"items": result, "overall_rate": overall_rate, "total_won": total_won, "total_lost": total_lost}


@router.get("/api/cases/{case_id}")
def get_case_detail(case_id: str):
    """案件の全データを返す（result + inputs を含む）"""
    import json

    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT data FROM past_cases WHERE id = ?", (case_id,)
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Case not found")
        payload = json.loads(row["data"] or "{}")
        payload.setdefault("_source", "past_cases")
        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_case_detail DB error case_id=%s: %s", case_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/cases/progress-stamp")
def stamp_case_progress(req: CaseProgressStampRequest):
    from data_cases import load_all_cases, update_case  # type: ignore[import]
    import datetime

    cases = load_all_cases()
    target = None
    for c in cases:
        if c.get("id") == req.case_id or c.get("company_no") == req.case_id or c.get("company_name") == req.case_id:
            target = c
            break
    if not target:
        raise HTTPException(status_code=404, detail="Case not found")

    stamp_date = req.occurred_at or datetime.datetime.now().strftime("%Y-%m-%d")
    if req.event_type == "estimate_sent":
        key = "estimate_sent_date"
    elif req.event_type == "customer_response":
        key = "customer_response_date"
    else:
        raise HTTPException(status_code=422, detail="event_type must be estimate_sent or customer_response")

    if not update_case(target.get("id"), {key: stamp_date}):
        raise HTTPException(status_code=500, detail="Failed to update timestamp")

    target[key] = stamp_date
    if not target.get("registration_date"):
        target["registration_date"] = str(target.get("timestamp", ""))[:10] or stamp_date

    prob = _compute_case_closure_probability(target)
    if prob is not None:
        update_case(target.get("id"), {
            "predicted_closure_probability": prob,
            "predicted_closure_probability_percent": round(prob * 100.0, 2),
        })

    return {
        "status": "success",
        "case_id": target.get("id"),
        "event_type": req.event_type,
        "stamped_at": stamp_date,
        "closure_probability": prob,
        "closure_probability_percent": round(prob * 100.0, 2) if prob is not None else None,
    }


@router.post("/api/deal/closure-probability", response_model=DealClosureResponse)
def calc_deal_closure_probability(req: DealClosureRequest):
    try:
        if req.delta_send is not None and req.delta_response is not None:
            features = build_features_from_deltas(req.delta_send, req.delta_response)
        elif req.registration_date and req.estimate_sent_date and req.customer_response_date:
            features = build_features(
                registration_date=req.registration_date,
                estimate_sent_date=req.estimate_sent_date,
                customer_response_date=req.customer_response_date,
            )
        else:
            raise ValueError("Either (delta_send & delta_response) or all 3 dates are required")
        prob = compute_closure_likelihood(features, has_cash_data=req.has_cash_data)
        return DealClosureResponse(
            closure_probability=prob,
            closure_probability_percent=round(prob * 100.0, 2),
            delta_send=features.delta_send,
            delta_response=features.delta_response,
            model_note="Trajectory-likelihood prototype (residue-inspired), preserving existing score pipeline.",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
