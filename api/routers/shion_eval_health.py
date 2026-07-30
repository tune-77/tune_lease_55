"""
api/routers/shion_eval_health.py  – REV-234 Phase 6
Extracted from api/main.py.

Endpoints:
  GET  /api/shion-eval-health        – get_shion_eval_health
  POST /api/shion-eval-health/check  – post_shion_eval_health_check
  POST /api/shion-eval-health/repair – post_shion_eval_health_repair
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["shion-eval-health"])

_SCRIPT_DIR = str(Path(__file__).resolve().parent.parent)   # api/
_REPO_ROOT = str(Path(_SCRIPT_DIR).parent)                  # repo root


# ── Models ────────────────────────────────────────────────────────────────────

class ShionEvalHealthCheckRequest(BaseModel):
    case_id: str
    reply: str = ""
    memory_debug: dict = Field(default_factory=dict)
    knowledge_refs: list = Field(default_factory=list)
    daily_clinic_used: Optional[bool] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/api/shion-eval-health")
def get_shion_eval_health():
    try:
        from shion_eval_health import build_shion_eval_health_payload  # type: ignore[import]
        return build_shion_eval_health_payload(Path(_REPO_ROOT))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/shion-eval-health/check")
def post_shion_eval_health_check(req: ShionEvalHealthCheckRequest):
    try:
        from shion_eval_health import case_by_id, evaluate_shion_trace  # type: ignore[import]
        case = case_by_id(req.case_id)
        return evaluate_shion_trace(
            case,
            reply=req.reply,
            memory_debug=req.memory_debug,
            knowledge_refs=req.knowledge_refs,
            daily_clinic_used=req.daily_clinic_used,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"eval case not found: {req.case_id}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/shion-eval-health/repair")
def post_shion_eval_health_repair():
    try:
        from shion_eval_health import repair_operation_loop_health  # type: ignore[import]
        return repair_operation_loop_health(Path(_REPO_ROOT))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
