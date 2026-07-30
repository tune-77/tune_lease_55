"""
api/routers/recipes.py  – REV-234 Phase 6
Extracted from api/main.py.

Endpoints:
  GET  /api/recipes/status                    – get_recipes_status
  GET  /api/recipes/pending                   – get_pending_recipes
  POST /api/recipes/{recipe_id}/approve       – approve_recipe
  POST /api/recipes/{recipe_id}/approve-and-apply – approve_and_apply_recipe
  POST /api/recipes/{recipe_id}/reject        – reject_recipe
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["recipes"])

_SCRIPT_DIR = str(Path(__file__).resolve().parent.parent)   # api/
_REPO_ROOT = str(Path(_SCRIPT_DIR).parent)                  # repo root

_RECIPES_ROOT = Path(_REPO_ROOT) / "data" / "recipes"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _candidate_report_dirs() -> list[Path]:
    bundle_root = Path(os.environ.get("CLOUDRUN_BUNDLE_DIR") or (Path(_REPO_ROOT) / ".cloudrun_bundle"))
    dirs = [Path(_REPO_ROOT) / "reports", bundle_root / "reports"]
    unique: list[Path] = []
    seen: set[str] = set()
    for directory in dirs:
        key = str(directory)
        if key not in seen:
            seen.add(key)
            unique.append(directory)
    return unique


def _latest_improvement_report_path() -> Path | None:
    for reports_dir in _candidate_report_dirs():
        latest = reports_dir / "latest.json"
        if latest.exists():
            return latest
    candidates: list[Path] = []
    for reports_dir in _candidate_report_dirs():
        candidates.extend(reports_dir.glob("improvement_report_*.json"))
    candidates = sorted(candidates)
    return candidates[-1] if candidates else None


def _recipe_count(dirname: str) -> int:
    path = _RECIPES_ROOT / dirname
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("*.json") if item.is_file())


def _recipe_risk_level(recipe: dict) -> str:
    safety = recipe.get("safety", "none")
    files = recipe.get("files", [])
    total_changes = sum(len(f.get("changes", [])) for f in files)
    if total_changes >= 10 or safety == "tsc":
        return "medium"
    return "low"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/api/recipes/status")
def get_recipes_status():
    latest_path = _latest_improvement_report_path()
    latest: dict = {}
    if latest_path and latest_path.exists():
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception:
            latest = {}
    codex_queue = latest.get("codex_auto_queue") if isinstance(latest.get("codex_auto_queue"), dict) else {}
    return {
        "pending_count": _recipe_count("pending"),
        "approved_count": _recipe_count("approved"),
        "applied_count": _recipe_count("applied"),
        "rejected_count": _recipe_count("rejected"),
        "codex_auto_queue": {
            "status": codex_queue.get("status", ""),
            "queued_count": codex_queue.get("queued_count", 0),
            "safe_count": codex_queue.get("safe_count", 0),
            "maybe_count": codex_queue.get("maybe_count", 0),
            "manual_or_blocked_count": codex_queue.get("manual_or_blocked_count", 0),
        },
        "note": "今回の修正案を適用待ちへ送る操作です。実適用は scripts/apply_recipe.py が適用待ちを処理します。",
    }


@router.get("/api/recipes/pending")
def get_pending_recipes():
    pending_dir = _RECIPES_ROOT / "pending"
    recipes: list[dict] = []
    if not pending_dir.exists():
        return {"recipes": recipes}
    for path in sorted(pending_dir.glob("*.json")):
        try:
            recipe = json.loads(path.read_text(encoding="utf-8"))
            recipe["id"] = path.stem
            recipe.setdefault("risk_level", _recipe_risk_level(recipe))
            recipes.append(recipe)
        except Exception:
            pass
    return {"recipes": recipes}


@router.post("/api/recipes/{recipe_id}/approve")
def approve_recipe(recipe_id: str):
    src = _RECIPES_ROOT / "pending" / f"{recipe_id}.json"
    if not src.exists():
        raise HTTPException(status_code=404, detail="Recipe not found")
    dst_dir = _RECIPES_ROOT / "approved"
    dst_dir.mkdir(parents=True, exist_ok=True)
    src.rename(dst_dir / f"{recipe_id}.json")
    return {"status": "approved", "id": recipe_id}


@router.post("/api/recipes/{recipe_id}/approve-and-apply")
def approve_and_apply_recipe(recipe_id: str):
    if os.environ.get("K_SERVICE"):
        raise HTTPException(
            status_code=409,
            detail="レシピの自動適用はローカルの作業ツリーへ修正を入れる処理のため、Cloud Run上では実行できません。",
        )
    src = _RECIPES_ROOT / "pending" / f"{recipe_id}.json"
    if not src.exists():
        raise HTTPException(status_code=404, detail="Recipe not found")
    try:
        from datetime import datetime as _dt, timezone as _timezone
        from scripts.apply_recipe import _check_git_clean, _process_recipe  # type: ignore[import]

        _check_git_clean()
        recipe = json.loads(src.read_text(encoding="utf-8"))
        recipe["approved_at"] = _dt.now(_timezone.utc).isoformat()
        dst_dir = _RECIPES_ROOT / "approved"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{recipe_id}.json"
        dst.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
        src.unlink()
        status, message = _process_recipe(dst)
        return {"status": status, "id": recipe_id, "message": message}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/recipes/{recipe_id}/reject")
def reject_recipe(recipe_id: str):
    src = _RECIPES_ROOT / "pending" / f"{recipe_id}.json"
    if not src.exists():
        raise HTTPException(status_code=404, detail="Recipe not found")
    dst_dir = _RECIPES_ROOT / "rejected"
    dst_dir.mkdir(parents=True, exist_ok=True)
    src.rename(dst_dir / f"{recipe_id}.json")
    return {"status": "rejected", "id": recipe_id}
