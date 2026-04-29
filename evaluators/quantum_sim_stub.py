from __future__ import annotations

from .base import BaseEvaluator
from .types import EvaluatorResult


class QuantumSimStub(BaseEvaluator):
    name = "quantum_sim"
    version = "stub"

    def is_ready(self) -> bool:
        return False

    def predict(self, case: dict) -> EvaluatorResult:
        raise RuntimeError("quantum_sim stub is not ready")
