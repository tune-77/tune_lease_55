"""Privacy-safe common clients and local usage metrics for AI providers.

The wrappers in this module deliberately do not record prompts, responses,
API keys, URLs, or exception messages.  Each completed call appends one JSON
object containing only operational metadata to ``data/ai_usage.jsonl``.

Provider SDK objects are proxied instead of reimplemented, so existing call
sites keep using the native SDK response types and configuration objects.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

_T = TypeVar("_T")
_WRITE_LOCK = threading.Lock()
_DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "data" / "ai_usage.jsonl"


def _enabled() -> bool:
    return os.environ.get("AI_USAGE_LOG_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def usage_log_path() -> Path:
    configured = os.environ.get("AI_USAGE_LOG_PATH", "").strip()
    return Path(configured).expanduser() if configured else _DEFAULT_LOG_PATH


def _integer(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value.get(name)
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def extract_token_usage(response: Any) -> dict[str, int | None]:
    """Extract token counts from Gemini, Anthropic, or OpenAI responses."""
    usage = _field(response, "usage_metadata", "usage")
    if usage is None and isinstance(response, dict):
        usage = response.get("usageMetadata")

    input_tokens = _integer(
        _field(
            usage,
            "prompt_token_count",
            "promptTokenCount",
            "input_tokens",
            "inputTokens",
            "prompt_tokens",
        )
    )
    output_tokens = _integer(
        _field(
            usage,
            "candidates_token_count",
            "candidatesTokenCount",
            "output_tokens",
            "outputTokens",
            "completion_tokens",
        )
    )
    total_tokens = _integer(
        _field(usage, "total_token_count", "totalTokenCount", "total_tokens")
    )
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _append_usage(entry: dict[str, Any]) -> None:
    if not _enabled():
        return
    try:
        path = usage_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        with _WRITE_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                try:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                except (ImportError, OSError):
                    pass
                handle.write(line)
                handle.flush()
                try:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass
    except Exception:
        # Metrics must never make an AI request fail.
        return


def tracked_ai_call(
    call: Callable[[], _T],
    *,
    provider: str,
    model: str,
    feature: str,
    operation: str = "generate_content",
    token_extractor: Callable[[Any], dict[str, int | None]] = extract_token_usage,
) -> _T:
    """Execute one provider call and record privacy-safe operational metadata."""
    started = time.perf_counter()
    base: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": str(provider or "unknown"),
        "feature": str(feature or "unknown"),
        "operation": str(operation or "unknown"),
        "model": str(model or "unknown"),
    }
    try:
        response = call()
    except Exception as exc:
        _append_usage(
            {
                **base,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "ok": False,
                "error_type": type(exc).__name__,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }
        )
        raise

    try:
        tokens = token_extractor(response)
    except Exception:
        tokens = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    _append_usage(
        {
            **base,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "ok": True,
            "error_type": None,
            **tokens,
        }
    )
    return response


def _http_token_usage(response: Any) -> dict[str, int | None]:
    try:
        payload = response.json()
    except Exception:
        payload = None
    return extract_token_usage(payload)


def tracked_ai_http_call(
    call: Callable[[], _T],
    *,
    provider: str,
    model: str,
    feature: str,
    operation: str = "generate_content",
) -> _T:
    """Track an HTTP AI call, including non-success status codes as failures."""
    def _execute() -> _T:
        response = call()
        raise_for_status = getattr(response, "raise_for_status", None)
        if callable(raise_for_status):
            raise_for_status()
        return response

    return tracked_ai_call(
        _execute,
        provider=provider,
        model=model,
        feature=feature,
        operation=operation,
        token_extractor=_http_token_usage,
    )


class _GoogleModelsProxy:
    def __init__(self, target: Any, feature: str):
        self._target = target
        self._feature = feature

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model") or (args[0] if args else "unknown")
        return tracked_ai_call(
            lambda: self._target.generate_content(*args, **kwargs),
            provider="google",
            model=str(model),
            feature=self._feature,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _GoogleClientProxy:
    def __init__(self, target: Any, feature: str):
        self._target = target
        self.models = _GoogleModelsProxy(target.models, feature)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def google_genai_client(
    *,
    feature: str,
    client_factory: Callable[..., Any] | None = None,
    **client_kwargs: Any,
) -> Any:
    """Create an instrumented ``google.genai.Client`` compatible proxy."""
    if client_factory is None:
        from google import genai

        client_factory = genai.Client
    return _GoogleClientProxy(client_factory(**client_kwargs), feature)


class _LegacyGeminiModelProxy:
    def __init__(self, target: Any, feature: str, model: str):
        self._target = target
        self._feature = feature
        self._model = model

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        return tracked_ai_call(
            lambda: self._target.generate_content(*args, **kwargs),
            provider="google",
            model=self._model,
            feature=self._feature,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def instrument_legacy_gemini_model(target: Any, *, feature: str, model: str) -> Any:
    """Wrap a ``google.generativeai.GenerativeModel`` without changing its API."""
    return _LegacyGeminiModelProxy(target, feature, str(model or "unknown"))


class _MessagesProxy:
    def __init__(self, target: Any, feature: str, provider: str):
        self._target = target
        self._feature = feature
        self._provider = provider

    def create(self, *args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model") or "unknown"
        return tracked_ai_call(
            lambda: self._target.create(*args, **kwargs),
            provider=self._provider,
            model=str(model),
            feature=self._feature,
            operation="messages.create",
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _AnthropicClientProxy:
    def __init__(self, target: Any, feature: str):
        self._target = target
        self.messages = _MessagesProxy(target.messages, feature, "anthropic")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def anthropic_client(
    *,
    feature: str,
    client_factory: Callable[..., Any] | None = None,
    **client_kwargs: Any,
) -> Any:
    """Create an instrumented ``anthropic.Anthropic`` compatible proxy."""
    if client_factory is None:
        import anthropic

        client_factory = anthropic.Anthropic
    return _AnthropicClientProxy(client_factory(**client_kwargs), feature)
