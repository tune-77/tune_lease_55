"""DemoReadonlyMiddleware の検証。

starlette が無い環境では自動 skip する。ミドルウェアは api/demo_guard.py に
独立しているため、重い api.main を読み込まずにテストできる。
"""
import pytest

pytest.importorskip("starlette")
from starlette.applications import Starlette  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from api.demo_guard import DemoReadonlyMiddleware  # noqa: E402


def _build_client(monkeypatch, readonly: str | None) -> TestClient:
    if readonly is None:
        monkeypatch.delenv("DEMO_READONLY", raising=False)
    else:
        monkeypatch.setenv("DEMO_READONLY", readonly)

    async def item(request):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[
        Route("/api/cases/{cid}", item, methods=["GET", "DELETE"]),
    ])
    app.add_middleware(DemoReadonlyMiddleware)
    return TestClient(app)


def test_delete_allowed_when_disabled(monkeypatch):
    client = _build_client(monkeypatch, None)
    assert client.delete("/api/cases/1").status_code == 200


def test_delete_blocked_when_readonly(monkeypatch):
    client = _build_client(monkeypatch, "1")
    assert client.delete("/api/cases/1").status_code == 403


def test_non_delete_allowed_when_readonly(monkeypatch):
    client = _build_client(monkeypatch, "1")
    # 閲覧・試用（GET/POST等）は readonly でも通す
    assert client.get("/api/cases/1").status_code == 200


def test_truthy_variants(monkeypatch):
    for val in ("true", "yes", "on", "1"):
        client = _build_client(monkeypatch, val)
        assert client.delete("/api/cases/1").status_code == 403
    for val in ("0", "false", ""):
        client = _build_client(monkeypatch, val)
        assert client.delete("/api/cases/1").status_code == 200
