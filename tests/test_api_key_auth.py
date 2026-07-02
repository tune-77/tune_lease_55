"""ApiKeyAuthMiddleware の検証。

starlette が無い環境では自動 skip する。ミドルウェアは api/api_key_auth.py に
独立しているため、重い api.main を読み込まずにテストできる。
"""
import pytest

pytest.importorskip("starlette")
from starlette.applications import Starlette  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from api.api_key_auth import ApiKeyAuthMiddleware  # noqa: E402


def _build_client(monkeypatch, api_key: str) -> TestClient:
    if api_key:
        monkeypatch.setenv("API_ACCESS_KEY", api_key)
    else:
        monkeypatch.delenv("API_ACCESS_KEY", raising=False)

    async def healthz(request):
        return JSONResponse({"ok": True})

    async def secret(request):
        return JSONResponse({"data": "sensitive"})

    app = Starlette(routes=[
        Route("/healthz", healthz),
        Route("/api/secret", secret),
    ])
    app.add_middleware(ApiKeyAuthMiddleware)
    return TestClient(app)


def test_disabled_when_key_unset(monkeypatch):
    client = _build_client(monkeypatch, "")
    assert client.get("/api/secret").status_code == 200


def test_blocks_without_key_when_enabled(monkeypatch):
    client = _build_client(monkeypatch, "s3cret")
    assert client.get("/api/secret").status_code == 401


def test_allows_with_correct_key(monkeypatch):
    client = _build_client(monkeypatch, "s3cret")
    assert client.get("/api/secret", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get(
        "/api/secret", headers={"Authorization": "Bearer s3cret"}
    ).status_code == 200


def test_rejects_wrong_key(monkeypatch):
    client = _build_client(monkeypatch, "s3cret")
    assert client.get("/api/secret", headers={"X-API-Key": "nope"}).status_code == 401


def test_health_exempt_even_when_enabled(monkeypatch):
    client = _build_client(monkeypatch, "s3cret")
    assert client.get("/healthz").status_code == 200
