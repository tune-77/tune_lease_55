from __future__ import annotations

import datetime
import hashlib
import json
import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from api.cloudrun_writeback import record_cloudrun_input_event
from runtime_paths import get_data_path

router = APIRouter()

DEFAULT_USAGE_LOG = Path(get_data_path("shion_memory_usage_log.jsonl"))
_log_lock = threading.Lock()

MemoryFeedbackOutcome = Literal["helped", "neutral", "challenged"]


class ShionMemoryFeedbackRequest(BaseModel):
    memory_ids: list[str] = Field(default_factory=list)
    outcome: MemoryFeedbackOutcome
    route: str = ""
    surface: str = "next_chat"
    question: str = ""
    response_hash: str = ""
    comment: str = ""
    user_id: str = "default"


def _clean_memory_ids(memory_ids: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in memory_ids:
        mid = str(raw or "").strip()
        if not mid or mid in seen:
            continue
        seen.add(mid)
        cleaned.append(mid[:80])
    return cleaned[:12]


def append_shion_memory_feedback(
    req: ShionMemoryFeedbackRequest,
    *,
    usage_log: Path = DEFAULT_USAGE_LOG,
) -> dict:
    refs = _clean_memory_ids(req.memory_ids)
    if not refs:
        raise HTTPException(status_code=422, detail="memory_ids is required")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    question = str(req.question or "").strip()
    response_hash = str(req.response_hash or "").strip()
    seed = "\n".join([now, "|".join(refs), req.outcome, question[:160], response_hash])
    entry = {
        "event_id": hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16],
        "ts": now,
        "source": "explicit_memory_feedback",
        "surface": str(req.surface or "next_chat")[:120],
        "route": str(req.route or "feedback")[:80],
        "refs": refs,
        "ref_count": len(refs),
        "memory_feedback": req.outcome,
        "question": question[:160],
        "response_hash": response_hash[:96],
        "comment": str(req.comment or "").strip()[:300],
        "user_id": str(req.user_id or "default")[:80],
    }
    usage_log.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        with usage_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return entry


@router.post("/api/shion-memory-feedback")
def post_shion_memory_feedback(req: ShionMemoryFeedbackRequest, background_tasks: BackgroundTasks) -> dict:
    entry = append_shion_memory_feedback(req)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="shion_memory_feedback",
        surface=entry["surface"],
        payload={**entry, "schema_version": 1},
    )
    return {"status": "ok", "feedback": entry}
