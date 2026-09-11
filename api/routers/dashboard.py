"""Dashboard read-only endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["dashboard"])


@router.get("/api/dashboard/data-health")
def get_dashboard_data_health():
    """Return only dashboard health metadata, never case or aggregate contents."""
    try:
        from api.dashboard_data_health import evaluate_dashboard_data_health
        from data_cases import load_dashboard_stats_cache, refresh_dashboard_stats_cache

        payload = load_dashboard_stats_cache()
        if payload is None:
            payload = refresh_dashboard_stats_cache()
        healthy, reason = evaluate_dashboard_data_health(payload)
        return {"healthy": healthy, "reason": reason}
    except Exception:
        logger.exception("dashboard data health check failed")
        return {"healthy": False, "reason": "health_check_error"}
