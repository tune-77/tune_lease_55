"""Security header middleware."""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


def get_trusted_hosts() -> list[str]:
    """Return the explicit Host allowlist for local and managed runtimes."""
    hosts = {"localhost", "127.0.0.1", "::1", "testserver"}
    if os.environ.get("K_SERVICE", "").strip():
        hosts.add("*.run.app")
    if os.environ.get("PUBLIC_TUNNEL", "").strip().lower() in {"1", "true", "yes", "on"}:
        hosts.add("*.trycloudflare.com")
    for raw_host in os.environ.get("TRUSTED_HOSTS", "").split(","):
        if raw_host.strip():
            hosts.add(raw_host.strip())
    for raw_origin in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(","):
        hostname = urlsplit(raw_origin.strip()).hostname
        if hostname:
            hosts.add(hostname)
    return sorted(hosts)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response
