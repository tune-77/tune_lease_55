"""
api/routers/chronicle.py  – REV-234 Phase 6
Extracted from api/main.py.

Endpoints:
  GET  /api/chronicle/summary                – get_chronicle_summary_api
  GET  /api/chronicle/history                – get_chronicle_history_api
  GET  /api/chronicle/snapshots              – get_chronicle_snapshots_api
  POST /api/chronicle/rollback               – post_chronicle_rollback_api
  POST /api/chronicle/simulation/round       – run_simulation_round_api
  GET  /api/chronicle/simulation/history     – get_simulation_history_api
  GET  /api/chronicle/simulation/archaia_log – get_archaia_log_api
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["chronicle"])

_SCRIPT_DIR = str(Path(__file__).resolve().parent.parent)   # api/
_REPO_ROOT = str(Path(_SCRIPT_DIR).parent)                  # repo root


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_approval_rates_standalone(cases: list):
    """civilization_chronicle.py の _compute_approval_rates をインライン化（streamlit非依存）。"""
    from scoring_core import APPROVAL_LINE as _APPROVAL_LINE  # type: ignore[import]
    scored = [
        c for c in cases
        if c.get("result") and c["result"].get("score") is not None
    ]
    if len(scored) < 10:
        return None, None
    all_scores = [c["result"]["score"] for c in scored]
    baseline_rate = sum(1 for s in all_scores if s >= _APPROVAL_LINE) / len(all_scores)
    recent = all_scores[-30:]
    recent_rate = sum(1 for s in recent if s >= _APPROVAL_LINE) / len(recent)
    return baseline_rate, recent_rate


# ── Models ────────────────────────────────────────────────────────────────────

class RollbackRequest(BaseModel):
    snapshot_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/api/chronicle/summary")
def get_chronicle_summary_api():
    from data_cases import load_all_cases  # type: ignore[import]
    try:
        cases = load_all_cases()
        baseline, recent = _compute_approval_rates_standalone(cases)
        return {
            "baseline_rate": baseline or 0,
            "recent_rate": recent or 0,
            "drift": abs((recent or 0) - (baseline or 0)),
            "warn_threshold": 0.15,
            "total_cases": len(cases)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/chronicle/history")
def get_chronicle_history_api():
    from data_cases import load_coeff_history  # type: ignore[import]
    try:
        history = load_coeff_history()
        return {"history": history[:100]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/chronicle/snapshots")
def get_chronicle_snapshots_api():
    """ガバナンスのスナップショット一覧を返す。data_cases.py のロジックをインライン化。"""
    snap_path = os.path.join(_REPO_ROOT, "data", "governance_snapshots.json")
    try:
        if not os.path.exists(snap_path):
            return {"snapshots": []}
        with open(snap_path, "r", encoding="utf-8") as f:
            snaps = json.load(f)
        return {"snapshots": list(reversed(snaps)) if snaps else []}
    except Exception as e:
        return {"snapshots": [], "error": str(e)}


@router.post("/api/chronicle/rollback")
def post_chronicle_rollback_api(req: RollbackRequest):
    from data_cases import load_governance_snapshots, save_coeff_overrides  # type: ignore[import]
    try:
        snaps = load_governance_snapshots()
        target = next((s for s in snaps if s.get("id") == req.snapshot_id), None)
        if not target:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        overrides = target.get("overrides", {})
        comment = f"Rollback to {req.snapshot_id} (via API)"
        success = save_coeff_overrides(overrides, comment=comment)
        return {"status": "success" if success else "failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/chronicle/simulation/round")
def run_simulation_round_api():
    from novel_simulation import run_simulation_round  # type: ignore[import]
    try:
        res = run_simulation_round()
        if "error" in res:
            raise HTTPException(status_code=400, detail=res["error"])
        return res
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/chronicle/simulation/history")
def get_simulation_history_api(limit: int = 20):
    from novel_simulation import get_round_history  # type: ignore[import]
    try:
        return {"history": get_round_history(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/chronicle/simulation/archaia_log")
def get_archaia_log_api(limit: int = 30):
    from novel_simulation import get_archaia_log  # type: ignore[import]
    try:
        return {"logs": get_archaia_log(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
