from __future__ import annotations

import os
from .base import BaseEvaluator
from .types import EvaluatorResult


class QuantumAdapter(BaseEvaluator):
    name = "quantum"
    version = "v1"

    def __init__(self, model_path: str = "data/quantum_model.joblib"):
        self.model_path = model_path
        self._gate = None

    def _load(self):
        if self._gate is None:
            from quantum_analysis_module import QuantumGate
            self._gate = QuantumGate.load_cached(self.model_path)
        return self._gate

    def is_ready(self) -> bool:
        return os.path.exists(self.model_path)

    def predict(self, case: dict) -> EvaluatorResult:
        gate = self._load()
        qr = gate.predict({"inputs": case.get("inputs", case)})
        return EvaluatorResult(
            name=self.name,
            version=self.version,
            risk=float(qr.get("quantum_risk", 0.0)),
            verdict=str(qr.get("verdict", "")),
            anomalies=qr.get("pair_anomalies", []) or [],
            contributions=qr.get("pair_contributions", {}) or {},
            explanation=str(qr.get("explanation", "")),
        )
