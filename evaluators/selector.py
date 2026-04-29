from __future__ import annotations

import logging
import random
from .registry import get
from .types import EvaluatorParams, EvaluatorResult

logger = logging.getLogger(__name__)


class EpsilonGreedySelector:
    def __init__(self, params: EvaluatorParams):
        self.params = params

    def _choose(self) -> str:
        if random.random() < self.params.epsilon:
            return random.choice(self.params.exploration_pool)
        return self.params.exploit_engine

    def evaluate(self, case: dict, base_score: float) -> tuple[str, EvaluatorResult] | tuple[None, None]:
        if base_score < self.params.score_min:
            return None, None
        chosen = self._choose()
        try:
            ev = get(chosen)()
            if not ev.is_ready():
                raise RuntimeError(f"{chosen} not ready")
            return chosen, ev.predict(case)
        except Exception as e:
            logger.warning("selector fallback: %s", e)
            fb_name = self.params.fallback_engine
            fb = get(fb_name)()
            return f"fallback:{chosen}->{fb_name}", fb.predict(case)
