"""Privacy-bounded lease-intelligence activity endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

from lease_intelligence_activity import record_user_activity, suggest_related_feature

router = APIRouter(prefix="/api/lease-intelligence", tags=["lease-intelligence"])


class LeaseIntelligenceActivityRequest(BaseModel):
    surface: str
    action: str = "page_view"
    event_id: str = ""


@router.post("/activity")
def record_lease_intelligence_activity_api(req: LeaseIntelligenceActivityRequest):
    """Record a privacy-bounded explicit in-app activity event."""
    recorded = record_user_activity(
        surface=req.surface,
        action=req.action,
        event_id=req.event_id,
    )
    return {
        "recorded": recorded,
        "privacy": "Stores only surface, action, timestamp, and a dedupe id.",
    }


@router.get("/related-suggestion")
def get_lease_intelligence_related_suggestion_api():
    """直近の利用状況から、関連するが未使用の機能を最大1件提案する（REV-237）。"""
    return {"suggestion": suggest_related_feature()}
