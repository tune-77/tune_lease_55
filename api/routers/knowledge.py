from __future__ import annotations

import datetime
import json
import os
import threading
from typing import Literal

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from api.cloudrun_writeback import record_cloudrun_input_event

router = APIRouter()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FEEDBACK_LOG_PATH = os.path.join(_REPO_ROOT, "data", "rag_feedback_log.jsonl")
_HIT_LOG_PATH = os.path.join(_REPO_ROOT, "data", "rag_hit_log.jsonl")
_feedback_log_lock = threading.Lock()
_hit_log_lock = threading.Lock()


class RagFeedbackRequest(BaseModel):
    query: str
    doc_id: str
    obsidian_ref: str
    rating: Literal["good", "bad"]
    surface: str = "next_chat_rag"


@router.post("/knowledge/feedback")
def post_rag_feedback(req: RagFeedbackRequest, background_tasks: BackgroundTasks) -> dict:
    now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    entry = {
        "ts": now_ts,
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
    hit_entry = {
        "ts": now_ts,
        "doc_id": req.doc_id,
        "obsidian_ref": req.obsidian_ref,
        "rating": req.rating,
        "surface": req.surface,
        "hit_type": "feedback_confirmed",
    }
    hit_line = json.dumps(hit_entry, ensure_ascii=False) + "\n"
    with _hit_log_lock:
        with open(_HIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(hit_line)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="rag_feedback",
        surface=req.surface,
        payload=entry,
    )
    return {"status": "ok"}
