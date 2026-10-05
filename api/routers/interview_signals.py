"""面談メモ・現場メモの定性シグナル（REV-470）。審査画面の参考表示専用で、スコア・判定には使わない。"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.interview_signals import (
    build_interview_signals_prompt_block,
    extract_interview_signals,
    interview_signals_enabled,
)

router = APIRouter(prefix="/api/screening", tags=["screening-interview-signals"])


class InterviewSignalsRequest(BaseModel):
    text: str = Field(default="", max_length=8000)


@router.post("/interview-signals")
def interview_signals(req: InterviewSignalsRequest) -> dict:
    if not interview_signals_enabled():
        return {"enabled": False, "signals": []}
    result = extract_interview_signals(req.text)
    return {"enabled": True, **result, "prompt_block": build_interview_signals_prompt_block(result)}
