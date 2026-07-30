"""紫苑自己分析・内面状態ルーター (REV-234 Phase3)"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["shion-meta"])

@router.get("/api/shion/self-analysis")
def get_shion_self_analysis(refresh: bool = False):
    """紫苑の自己分析を取得する（24時間キャッシュ）。"""
    from api.shion_self_analysis import get_shion_self_analysis as _get_analysis
    try:
        return _get_analysis(force_refresh=refresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/shion/inner-state")
def get_shion_inner_state():
    """紫苑の「内面状態（経験ループ・実践知・人間フィードバック）」を一括して取得するデバッグ用エンドポイント。"""
    try:
        from api.shion_experience_loop import load_experience_state
        state = load_experience_state()

        from api.shion_practical_knowledge import load_learned_practical_map, _iter_scenes, _public_scene
        learned_map = load_learned_practical_map()
        merged_scenes = _iter_scenes(learned_map)
        public_scenes = [_public_scene(s) for s in merged_scenes]

        learned_sources = []
        for s in public_scenes:
            for src in s.get("learned_sources") or []:
                if src not in learned_sources:
                    learned_sources.append(src)

        feedback_summary = {}
        routes_to_check = ["relationship_ux", "environment_continuity", "lease_judgment", "implementation"]
        for r in routes_to_check:
            feedback_summary[r] = _summarize_human_response_feedback(r)

        return {
            "experience_count": state.get("experience_count", 0),
            "current_focus": state.get("current_focus", ""),
            "self_narrative": state.get("self_narrative", ""),
            "mood": state.get("mood", {}),
            "confidence": state.get("confidence", {}),
            "recent_experiences": state.get("recent_experiences", []),
            "next_response_bias": state.get("next_response_bias", []),
            "open_questions": state.get("open_questions", []),
            "practical_scenes": public_scenes,
            "learned_sources": learned_sources,
            "human_feedback_summary": feedback_summary,
            "updated_at": state.get("updated_at", "")
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class PromoteKeypointRequest(BaseModel):
    text: str
    case_summary: str = ""
    role: str = ""


@router.post("/api/shion/promote-keypoint")
def promote_keypoint(req: PromoteKeypointRequest):
    """討論結果から抽出した判断基準を Obsidian vault の mind.json に候補として追記する。

    vault が無い環境（Cloud Run 等）では 503 で捨てずにローカルの保留キュー
    （data/shion_promoted_keypoints_pending.jsonl、gitignore対象）へ退避し、
    vault 環境での同期を待つ。

    注意:
    マルチエージェント討論の出力は、その場の仮説・視点であって紫苑中核の
    確定信念ではない。ここでは active 直入れせず、needs_review の
    judgment_asset_candidate として保存する。
    """
    import json as _json
    from pathlib import Path
    from datetime import date, datetime, timezone
    from api.shion_memory_taxonomy import infer_applies_when

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text が空です")

    new_entry: dict = {
        "fact": text,
        "source": "debate",
        "source_detail": "multi_agent_screening",
        "case": req.case_summary,
        "date": date.today().isoformat(),
        "memory_type": "judgment_memory",
        "asset_stage": "judgment_asset_candidate",
        "status": "needs_review",
        "review_required": True,
        "confidence": 0.62,
        "applies_when": infer_applies_when(text),
    }
    if req.role:
        new_entry["role"] = req.role

    vault_mind = None
    try:
        from lease_news_digest import find_vault
        vault = find_vault()
        if vault:
            candidate = (
                Path(vault)
                / "Projects"
                / "tune_lease_55"
                / "Lease Intelligence"
                / "mind.json"
            )
            if candidate.exists():
                vault_mind = candidate
    except Exception:
        vault_mind = None

    if vault_mind is not None:
        try:
            with vault_mind.open(encoding="utf-8") as f:
                data = _json.load(f)

            keypoints: list = data.get("conversation_keypoints") or []
            keypoints.append(new_entry)

            # 120件上限（古いものから削除）
            _LIMIT = 120
            if len(keypoints) > _LIMIT:
                keypoints = keypoints[-_LIMIT:]

            data["conversation_keypoints"] = keypoints

            # 書き戻し（一時ファイル経由でアトミックに）
            tmp = vault_mind.with_suffix(".json.tmp")
            tmp.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(vault_mind)

            return {"success": True, "storage": "vault", "total_keypoints": len(keypoints)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"mind.json 書き込みエラー: {e}")

    # vault 不在（Cloud Run 等）: 保留キューへ退避してキーポイントの消失を防ぐ
    try:
        pending_path = Path(__file__).parent.parent / "data" / "shion_promoted_keypoints_pending.jsonl"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        queued_entry = {**new_entry, "queued_at": datetime.now(timezone.utc).isoformat()}
        with pending_path.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(queued_entry, ensure_ascii=False) + "\n")
        return {
            "success": True,
            "storage": "local_pending",
            "note": "Obsidian vault 不在のため保留キューへ保存しました。vault 環境で mind.json へ同期してください。",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保留キュー書き込みエラー: {e}")


@router.get("/api/shion/central-synthesis")
def get_central_synthesis():
    """セントラルの最新状態を返す（world_view.commentary を読んで返す）。"""
    import json as _json
    from pathlib import Path

    try:
        from lease_news_digest import find_vault
        vault = find_vault()
        if not vault:
            return {"error": "Obsidian vault が見つかりません", "commentary": {}}

        vault_mind = (
            Path(vault)
            / "Projects"
            / "tune_lease_55"
            / "Lease Intelligence"
            / "mind.json"
        )
        if not vault_mind.exists():
            return {"error": "mind.json が見つかりません", "commentary": {}}

        data = _json.loads(vault_mind.read_text(encoding="utf-8"))
        world_view = data.get("world_view") or {}
        commentary = world_view.get("commentary") or {}

        return {
            "commentary": commentary,
            "world_view_summary": world_view.get("summary", ""),
            "total_keypoints": len(data.get("conversation_keypoints") or []),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


