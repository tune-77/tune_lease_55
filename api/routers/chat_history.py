"""Generic chat-history endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.get("/history")
def get_chat_history(user_id: str = "default", limit: int = 50, since: Optional[str] = None):
    """汎用チャット履歴を取得する。"""
    try:
        from api.chat_memory import get_recent_messages

        messages = get_recent_messages(user_id, limit=min(limit, 200), since=since)
        return {"user_id": user_id, "count": len(messages), "messages": messages}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/history")
def delete_chat_history(user_id: str = "default"):
    """汎用チャット履歴を全削除する。"""
    try:
        from api.chat_memory import delete_history

        deleted = delete_history(user_id)
        return {"deleted": deleted, "user_id": user_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
