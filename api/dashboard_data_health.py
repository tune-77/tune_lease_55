"""Privacy-preserving health checks for dashboard aggregate data."""

from __future__ import annotations

from typing import Any


def evaluate_dashboard_data_health(payload: Any) -> tuple[bool, str]:
    """Validate dashboard aggregates without returning their contents."""
    if not isinstance(payload, dict):
        return False, "invalid_payload"

    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        return False, "missing_analysis"

    closed_count = analysis.get("closed_count")
    if isinstance(closed_count, bool) or not isinstance(closed_count, int):
        return False, "invalid_closed_count"
    if closed_count < 0:
        return False, "invalid_closed_count"

    if not isinstance(payload.get("recent_cases"), list):
        return False, "missing_recent_cases"

    if closed_count > 0:
        if not isinstance(analysis.get("avg_financials"), dict):
            return False, "missing_avg_financials"
        top3_drivers = analysis.get("top3_drivers")
        if not isinstance(top3_drivers, list) or not top3_drivers:
            return False, "missing_top3_drivers"

    return True, "ok"
