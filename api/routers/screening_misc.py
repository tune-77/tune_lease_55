"""
api/routers/screening_misc.py  – REV-234 Phase 7
マルチエージェント審査・知識管理・会話履歴 エンドポイント群
Extracted from api/main.py.

Endpoints:
  POST   /api/crystallize-now               – crystallize_now
  POST   /api/reindex-knowledge             – reindex_knowledge
  POST   /api/reload-feedback               – reload_feedback
  POST   /api/multi-agent-screening         – multi_agent_screening
  POST   /api/multi-agent-screening/stream  – multi_agent_screening_stream
  POST   /api/business-plan/validate        – validate_business_plan_endpoint
  GET    /api/conversation-history          – get_conversation_history
  DELETE /api/conversation-history/{id}     – delete_conversation_history
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.schemas import BusinessPlanCheckRequest  # type: ignore[import]

router = APIRouter(tags=["screening-misc"])

# =============================================================================
# マルチエージェント審査 API (石橋 vs 風林火山 + 軍師調停)
# =============================================================================

class MultiAgentRequest(BaseModel):
    score: float
    company_name: str = ""
    industry_major: str = ""
    industry_sub: str = ""
    prefecture: str = ""
    nenshu: float = 0
    op_margin_pct: float = 0
    equity_ratio: float = 0
    bank_credit: float = 0
    lease_credit: float = 0
    asset_name: str = ""
    lease_amount: float = 0
    session_id: str = ""
    participants: Optional[dict] = None


@router.post("/api/crystallize-now")
def crystallize_now():
    """
    知識結晶化バッチを手動で即時実行する（スケジュールを待たずに呼び出せる）。
    同期実行して結果を返す。
    """
    from api.scheduler import run_crystallization_batch
    try:
        result = run_crystallization_batch()
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/reindex-knowledge")
def reindex_knowledge():
    """
    Obsidian ナレッジを手動で再インデックス化する。
    Vault 更新後に呼び出す。バックグラウンドで実行し即座に 202 を返す。
    """
    from api.knowledge.indexer import start_background_indexing
    start_background_indexing()
    return {"status": "indexing_started", "message": "バックグラウンドでインデックス化を開始しました"}


@router.post("/api/reload-feedback")
def reload_feedback():
    """
    Obsidian Feedback/ フォルダを再読み込みして ChromaDB を更新する。
    フィードバックファイルを追加・編集した後に呼び出す。バックグラウンドで実行し即座に返す。
    """
    from api.knowledge.feedback_watcher import start_background_feedback_loading
    start_background_feedback_loading()
    return {"status": "reload_started", "message": "フィードバックの再読み込みを開始しました"}


@router.post("/api/multi-agent-screening")
def multi_agent_screening(req: MultiAgentRequest):
    """
    マルチエージェント討論審査。

    スコアが承認ライン（scoring_core.APPROVAL_LINE、既定71）以上/40以下は統合派単独高速処理、
    その間の境界帯は懐疑派・楽観派が2ラウンド討論後に統合派が裁定。
    """
    from api.multi_agent_screening import run_debate_screening
    try:
        result = run_debate_screening(req.model_dump())
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        # 主経路（作り込んだ討論）が失敗 → 独立実装の ADK マルチエージェント討論へ
        # フォールバックし、審査を止めない。フォールバックも失敗したら従来通り 500。
        try:
            from api.shion_debate_adk import run_debate_adk_fallback
            print("[WARNING] run_debate_screening failed, fallback to ADK debate")
            return run_debate_adk_fallback(req.model_dump())
        except Exception as fe:
            print(f"[WARNING] ADK debate fallback also failed: {fe}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/multi-agent-screening/stream")
def multi_agent_screening_stream(req: MultiAgentRequest):
    """
    マルチエージェント討論審査（SSE ストリーミング版）。

    ラウンドごとの途中経過を配信する。イベントは
    {"type": "start" | "round1" | "round2" | "result" | "core_candidates" | "error", ...payload}。
    result まで受信すれば審査は完了。core_candidates は結果表示後に遅延配信される。
    """
    from api.multi_agent_screening import iter_debate_screening
    params = req.model_dump()

    def event_generator():
        try:
            for stage, payload in iter_debate_screening(params):
                yield f"data: {json.dumps({'type': stage, **payload}, ensure_ascii=False)}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            # 主経路が失敗 → ADK マルチエージェント討論のフォールバック結果を1件配信。
            try:
                from api.shion_debate_adk import run_debate_adk_fallback
                result = run_debate_adk_fallback(params)
                yield f"data: {json.dumps({'type': 'result', **result}, ensure_ascii=False)}\n\n"
            except Exception:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@router.post("/api/business-plan/validate")
def validate_business_plan_endpoint(req: BusinessPlanCheckRequest):
    """
    事業計画チェック（簡易版）。

    直近実績と計画値の整合性を機械チェックし、Gemini が利用可能なら
    講評・顧客への確認質問を付ける。LLM 不通でも機械チェックだけで応答する。
    """
    from api.business_plan_check import validate_business_plan
    try:
        return validate_business_plan(req.model_dump())
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/conversation-history")
def get_conversation_history(company_name: str, limit: int = 5):
    """企業名で過去の討論履歴を取得する。"""
    if not company_name:
        raise HTTPException(status_code=422, detail="company_name は必須です")
    try:
        from api.database import get_conversation_history
        history = get_conversation_history(company_name, limit=min(limit, 20))
        return {"company_name": company_name, "count": len(history), "sessions": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/conversation-history/{session_id}")
def delete_conversation_history(session_id: str):
    """session_id に紐づく会話履歴を削除する。"""
    try:
        from api.database import delete_conversation_session
        deleted = delete_conversation_session(session_id)
        return {"deleted": deleted, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))