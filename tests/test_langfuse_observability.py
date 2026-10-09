from __future__ import annotations

import sys
from types import ModuleType

import api.langfuse_observability as observability


def _reset(monkeypatch):
    monkeypatch.setattr(observability, "_initialized", False)
    monkeypatch.setattr(observability, "_client", None)


def test_langfuse_is_disabled_by_default(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.delenv("SHION_LANGFUSE_ENABLED", raising=False)

    assert observability.setup_langfuse_observability() is False


def test_langfuse_requires_both_project_keys(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("SHION_LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-test-key")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert observability.setup_langfuse_observability() is False


def test_langfuse_instrumentation_always_redacts_content(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("SHION_LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-test-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-test-key")

    captured: dict = {}

    class FakeClient:
        def flush(self):
            captured["flushed"] = True

    class FakeTraceConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    class FakeInstrumentor:
        def instrument(self, **kwargs):
            captured["instrument"] = kwargs

    langfuse = ModuleType("langfuse")
    langfuse.get_client = lambda: FakeClient()
    openinference = ModuleType("openinference")
    instrumentation = ModuleType("openinference.instrumentation")
    instrumentation.TraceConfig = FakeTraceConfig
    google_adk = ModuleType("openinference.instrumentation.google_adk")
    google_adk.GoogleADKInstrumentor = FakeInstrumentor

    monkeypatch.setitem(sys.modules, "langfuse", langfuse)
    monkeypatch.setitem(sys.modules, "openinference", openinference)
    monkeypatch.setitem(sys.modules, "openinference.instrumentation", instrumentation)
    monkeypatch.setitem(sys.modules, "openinference.instrumentation.google_adk", google_adk)

    assert observability.setup_langfuse_observability() is True
    assert observability.setup_langfuse_observability() is True
    assert all(captured["config"].values())
    assert captured["instrument"]["config"].__class__ is FakeTraceConfig
    assert observability.flush_langfuse_observability() is True
    assert captured["flushed"] is True


def test_trace_annotation_uses_only_safe_operational_metadata(monkeypatch):
    _reset(monkeypatch)
    captured: dict = {}

    class FakeClient:
        def update_current_trace(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(observability, "_initialized", True)
    monkeypatch.setattr(observability, "_client", FakeClient())

    assert observability.annotate_shion_trace(
        session_id="session-123",
        params={
            "company_name": "秘密の会社",
            "asset_name": "秘密の設備",
            "score": 82.5,
            "asset_warnings": ["warning"],
        },
    ) is True
    assert captured["name"] == "shion-screening"
    assert captured["session_id"] == "session-123"
    assert captured["tags"] == ["shion", "screening", "google-adk"]
    assert captured["metadata"]["has_asset_warnings"] == "true"
    assert "秘密" not in repr(captured)


def test_langfuse_dependency_failure_is_non_fatal(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("SHION_LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-test-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-test-key")
    monkeypatch.setitem(sys.modules, "langfuse", None)

    assert observability.setup_langfuse_observability() is False
    assert observability.flush_langfuse_observability() is False
