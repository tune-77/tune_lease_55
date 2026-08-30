#!/usr/bin/env python3
"""Fail when the live dashboard health endpoint reports unusable data."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_PATH = "/api/dashboard/data-health"


def validate_health_response(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "response is not a JSON object"
    if payload.get("healthy") is not True:
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason:
            reason = "unknown"
        return False, f"dashboard data unhealthy: {reason}"
    if payload.get("reason") != "ok":
        return False, "healthy response has an invalid reason"
    return True, "ok"


def fetch_health_response(base_url: str, path: str, timeout: float) -> Any:
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "tune-lease-dashboard-health/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    args = parser.parse_args()

    attempts = max(1, args.attempts)
    last_reason = "unknown failure"
    for attempt in range(1, attempts + 1):
        try:
            payload = fetch_health_response(args.base_url, args.path, args.timeout)
            healthy, reason = validate_health_response(payload)
            if healthy:
                print("[OK] dashboard data healthy")
                return 0
            last_reason = reason
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_reason = f"{type(exc).__name__}: {exc}"

        print(f"[WARN] dashboard health attempt {attempt}/{attempts} failed: {last_reason}")
        if attempt < attempts:
            time.sleep(max(0.0, args.retry_delay))

    print(f"[FAIL] dashboard data unhealthy after {attempts} attempts: {last_reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
