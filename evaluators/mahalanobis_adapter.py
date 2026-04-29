from __future__ import annotations

import os
import numpy as np
from .base import BaseEvaluator
from .types import EvaluatorResult


class MahalanobisAdapter(BaseEvaluator):
    name = "mahalanobis"
    version = "v1"

    def __init__(self, model_path: str = "data/mahalanobis_model.joblib"):
        self.model_path = model_path
        self._model = None

    def _load(self):
        if self._model is None:
            from mahalanobis_engine import MahalanobisScorer
            self._model = MahalanobisScorer.load(self.model_path)
        return self._model

    def is_ready(self) -> bool:
        return os.path.exists(self.model_path)

    def predict(self, case: dict) -> EvaluatorResult:
        model = self._load()
        vals = [case.get("inputs", case).get(f, 0.0) for f in model.feature_names]
        score, _, _, contribs = model.get_analysis(np.array(vals, dtype=float))
        risk = float(max(0.0, min(100.0, 100.0 - float(score))))
        return EvaluatorResult(
            name=self.name,
            version=self.version,
            risk=risk,
            verdict="要確認" if risk >= 40 else "妥当",
            anomalies=[],
            contributions={f: float(c) for f, c in zip(model.feature_names, contribs)},
            explanation="mahalanobis fallback",
        )
