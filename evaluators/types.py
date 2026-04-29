from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class EvaluatorParams(BaseModel):
    epsilon: float = Field(0.0, ge=0.0, le=1.0)
    score_min: float = 70.0
    exploit_engine: str = "quantum"
    exploration_pool: list[str] = ["mahalanobis", "quantum_sim"]
    fallback_engine: str = "mahalanobis"


class SelectionContext(BaseModel):
    base_score: float


class EvaluatorResult(BaseModel):
    name: str
    version: str = "unknown"
    risk: float = Field(ge=0.0, le=100.0)
    verdict: str = ""
    anomalies: list[dict[str, Any]] = []
    contributions: dict[str, float] = {}
    explanation: str = ""
