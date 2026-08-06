"""REV-222: 関係性スコア参照・フィードバック"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


@router.get("/api/relationship/state")
def get_relationship_state_endpoint():
    """関係性スコアの現在状態を返す（デバッグ・フロント参照用）。"""
    try:
        from api.shion_relationship import get_relationship_state
        return get_relationship_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RelationshipFeedbackRequest(BaseModel):
    feedback_type: str = "neutral"  # "positive" | "negative" | "neutral"
    topic_depth: str = "normal"     # "shallow" | "normal" | "deep"


@router.post("/api/relationship/feedback")
def post_relationship_feedback(req: RelationshipFeedbackRequest):
    """
    フロントから明示的なフィードバックを受け取り関係性スコアを更新する。
    チャット画面の「良かった／残念」ボタンなどから呼ぶ。
    """
    valid_fb = {"positive", "negative", "neutral"}
    valid_depth = {"shallow", "normal", "deep"}
    if req.feedback_type not in valid_fb:
        raise HTTPException(status_code=422, detail=f"feedback_type must be one of {valid_fb}")
    if req.topic_depth not in valid_depth:
        raise HTTPException(status_code=422, detail=f"topic_depth must be one of {valid_depth}")
    try:
        from api.shion_relationship import record_interaction
        state = record_interaction(
            feedback_type=req.feedback_type,   # type: ignore[arg-type]
            topic_depth=req.topic_depth,       # type: ignore[arg-type]
        )
        return {"status": "ok", "score": state["score"], "trend": state["trend"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
