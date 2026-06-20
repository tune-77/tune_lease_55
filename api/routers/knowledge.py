from __future__ import annotations

import datetime
import json
import os
import threading
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FEEDBACK_LOG_PATH = os.path.join(_REPO_ROOT, "data", "rag_feedback_log.jsonl")
_feedback_log_lock = threading.Lock()


class RagFeedbackRequest(BaseModel):
    query: str
    doc_id: str
    obsidian_ref: str
    rating: Literal["good", "bad"]
    surface: str = "next_chat_rag"


@router.post("/knowledge/feedback")
def post_rag_feedback(req: RagFeedbackRequest) -> dict:
    entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "query": req.query,
        "doc_id": req.doc_id,
        "obsidian_ref": req.obsidian_ref,
        "rating": req.rating,
        "surface": req.surface,
    }
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with _feedback_log_lock:
        with open(_FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    return {"status": "ok"}
