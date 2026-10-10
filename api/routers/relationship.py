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
            # REV-598 画面の「良かった」ボタンは発言のお礼より少し強く数える
            feedback_type="positive_button" if req.feedback_type == "positive" else req.feedback_type,  # type: ignore[arg-type]
            topic_depth=req.topic_depth,       # type: ignore[arg-type]
        )
        return {"status": "ok", "score": state["score"], "trend": state["trend"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/relationship/user-affect")
def get_user_affect_endpoint(user_id: str = "default"):
    """REV-465: 相手ごとに記憶した最近の様子（減衰後）を返す（確認・デバッグ用）。"""
    try:
        from api.user_affect_memory import build_user_affect_memory_block, recall_user_affect
        recall = recall_user_affect(user_id[:100])
        return {**recall.to_payload(), "prompt_block": build_user_affect_memory_block(recall)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/relationship/mutual-prediction")
def get_mutual_prediction_endpoint(user_id: str = "default"):
    """REV-472: 紫苑が相手について立てた予想・答え合わせ・気になること（確認・デバッグ用）。"""
    try:
        from api.shion_mutual_prediction import get_user_summary
        return get_user_summary(user_id[:100])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
