from __future__ import annotations

from abc import ABC, abstractmethod
from .types import EvaluatorResult


class BaseEvaluator(ABC):
    name: str = "base"
    version: str = "unknown"

    @abstractmethod
    def is_ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def predict(self, case: dict) -> EvaluatorResult:
        raise NotImplementedError
