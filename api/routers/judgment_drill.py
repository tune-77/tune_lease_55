"""
判断ドリル API ルーター (REV-234)
Cloud Run では FastAPI コンテナ側が CSV を読み書きする。
Next.js の route.ts はこのルーターを FASTAPI_URL 経由で呼ぶ。
"""
import csv as _csv
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/judgment-drill", tags=["judgment-drill"])

_DRILL_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "judgment_drills" / "judgment_drill_1000_20260725.csv"

_DRILL_EDITABLE = {
    "credit_score_20", "repayment_source_score_20", "asset_exit_score_20",
    "plan_specificity_score_20", "uncertainty_control_score_20", "total_score_100",
    "user_decision", "heaviest_issue", "additional_checks", "ringi_sentence",
    "score_decision_gap_note", "ai_feedback_outcome", "ai_feedback_note",
    "okf_candidate_tags", "presentation_use_ok", "reviewer", "judged_at", "status",
}


def _load_drill_rows() -> list[dict]:
    if not _DRILL_CSV.exists():
        return []
    with _DRILL_CSV.open(newline="", encoding="utf-8") as f:
        return list(_csv.DictReader(f))


def _save_drill_rows(rows: list[dict]) -> None:
    if not rows:
        return
    _DRILL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with _DRILL_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@router.get("/cases")
def get_judgment_drill_cases(status: Optional[str] = None, limit: int = 50, offset: int = 0):
    rows = _load_drill_rows()
    if status:
        rows = [r for r in rows if r.get("status") == status]
    total = len(rows)
    page = rows[offset: offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "cases": page}


@router.get("/cases/{case_id}")
def get_judgment_drill_case(case_id: str):
    rows = _load_drill_rows()
    for row in rows:
        if row.get("case_id") == case_id:
            return row
    raise HTTPException(status_code=404, detail=f"case_id={case_id} not found")


class JudgmentDrillUpdate(BaseModel):
    fields: dict


@router.patch("/cases/{case_id}")
def update_judgment_drill_case(case_id: str, req: JudgmentDrillUpdate):
    rows = _load_drill_rows()
    found = False
    for row in rows:
        if row.get("case_id") == case_id:
            for k, v in req.fields.items():
                if k in _DRILL_EDITABLE:
                    row[k] = v
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail=f"case_id={case_id} not found")
    _save_drill_rows(rows)
    return {"ok": True, "case_id": case_id}


@router.get("/stats")
def get_judgment_drill_stats():
    rows = _load_drill_rows()
    total = len(rows)
    judged = sum(1 for r in rows if r.get("status") == "judged")
    pending = sum(1 for r in rows if r.get("status") == "pending_user_judgment")
    decisions: dict[str, int] = {}
    for r in rows:
        d = r.get("user_decision", "")
        if d:
            decisions[d] = decisions.get(d, 0) + 1
    return {"total": total, "judged": judged, "pending": pending, "decisions": decisions}
