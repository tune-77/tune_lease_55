from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from scipy.stats import norm


@dataclass
class DealTrajectoryFeatures:
    delta_send: int
    delta_response: int
    delta_send_z: float
    delta_response_z: float


BASELINE_SEND_MEAN = 2.0
BASELINE_SEND_STD = 1.5
BASELINE_RESPONSE_MEAN = 3.0
BASELINE_RESPONSE_STD = 2.0
MAX_REASONABLE_DAYS = 120


def _parse_date(v: Optional[str]) -> Optional[date]:
    if not v:
        return None
    return date.fromisoformat(v)


def _clean_delta(days: int) -> int:
    if days < 0:
        return 0
    return min(days, MAX_REASONABLE_DAYS)


def _z_score(value: int, mean: float, std: float) -> float:
    safe_std = std if std > 1e-6 else 1.0
    return (value - mean) / safe_std


def build_features(registration_date: str, estimate_sent_date: str, customer_response_date: str) -> DealTrajectoryFeatures:
    reg = _parse_date(registration_date)
    est = _parse_date(estimate_sent_date)
    resp = _parse_date(customer_response_date)
    if not (reg and est and resp):
        raise ValueError("registration_date, estimate_sent_date, customer_response_date are required in YYYY-MM-DD format")

    delta_send = _clean_delta((est - reg).days)
    delta_response = _clean_delta((resp - est).days)

    return DealTrajectoryFeatures(
        delta_send=delta_send,
        delta_response=delta_response,
        delta_send_z=_z_score(delta_send, BASELINE_SEND_MEAN, BASELINE_SEND_STD),
        delta_response_z=_z_score(delta_response, BASELINE_RESPONSE_MEAN, BASELINE_RESPONSE_STD),
    )




def build_features_from_deltas(delta_send: int, delta_response: int) -> DealTrajectoryFeatures:
    clean_send = _clean_delta(int(delta_send))
    clean_response = _clean_delta(int(delta_response))
    return DealTrajectoryFeatures(
        delta_send=clean_send,
        delta_response=clean_response,
        delta_send_z=_z_score(clean_send, BASELINE_SEND_MEAN, BASELINE_SEND_STD),
        delta_response_z=_z_score(clean_response, BASELINE_RESPONSE_MEAN, BASELINE_RESPONSE_STD),
    )

def compute_closure_likelihood(features: DealTrajectoryFeatures, has_cash_data: bool) -> float:
    send_speed = norm.cdf(-features.delta_send_z)
    response_speed = norm.cdf(-features.delta_response_z)

    velocity_weight = 0.55 if has_cash_data else 0.75
    smoothness_weight = 1.0 - velocity_weight

    likelihood = smoothness_weight * send_speed + velocity_weight * response_speed

    if (not has_cash_data) and features.delta_response <= 1:
        likelihood += 0.18

    return max(0.0, min(1.0, likelihood))
