from __future__ import annotations

from .base import BaseEvaluator
from .mahalanobis_adapter import MahalanobisAdapter
from .quantum_adapter import QuantumAdapter
from .quantum_sim_stub import QuantumSimStub

_registry: dict[str, type[BaseEvaluator]] = {
    "mahalanobis": MahalanobisAdapter,
    "quantum": QuantumAdapter,
    "quantum_sim": QuantumSimStub,
}


def register(name: str, factory: type[BaseEvaluator]) -> None:
    _registry[name] = factory


def get(name: str) -> type[BaseEvaluator]:
    return _registry[name]


def available() -> list[str]:
    return sorted(_registry.keys())
