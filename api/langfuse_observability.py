"""Opt-in, privacy-preserving Langfuse tracing for Shion's Google ADK agent.

Tracing is disabled unless ``SHION_LANGFUSE_ENABLED`` is explicitly true and
both Langfuse project keys are present.  User/model content is always redacted;
the integration records operational metadata such as spans, tool names,
latency, errors, model names, and token counts.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_initialized = False
_client: Any | None = None


def _enabled() -> bool:
    return os.environ.get("SHION_LANGFUSE_ENABLED", "").strip().lower() in _TRUE_VALUES


def setup_langfuse_observability() -> bool:
    """Initialize Langfuse's Google ADK instrumentation once.

    Returns ``True`` only when instrumentation is active.  Configuration and
    dependency errors never prevent Shion from starting.
    """
    global _client, _initialized

    if _initialized:
        return True
    if not _enabled():
        return False

    missing = [
        name
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
        if not os.environ.get(name, "").strip()
    ]
    if missing:
        logger.warning(
            "Langfuse tracing is enabled but credentials are missing; tracing remains off (%s)",
            ", ".join(missing),
        )
        return False

    try:
        # The CLI/docs commonly call this setting BASE_URL, while the Python
        # SDK reads LANGFUSE_HOST. Accept both so local and deployed setups
        # behave consistently.
        if not os.environ.get("LANGFUSE_HOST", "").strip():
            base_url = os.environ.get("LANGFUSE_BASE_URL", "").strip()
            if base_url:
                os.environ["LANGFUSE_HOST"] = base_url
        from langfuse import get_client
        from openinference.instrumentation import TraceConfig
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor

        # Financial and company information must never leave the application in
        # trace payloads.  Keep this policy in code instead of relying on mutable
        # deployment environment defaults.
        config = TraceConfig(
            hide_inputs=True,
            hide_outputs=True,
            hide_input_messages=True,
            hide_output_messages=True,
            hide_input_text=True,
            hide_output_text=True,
            hide_prompts=True,
            hide_choices=True,
            hide_llm_invocation_parameters=True,
            hide_embeddings_text=True,
            hide_embeddings_vectors=True,
            hide_retrieval_documents=True,
        )
        _client = get_client()
        GoogleADKInstrumentor().instrument(config=config)
    except Exception as exc:  # noqa: BLE001 - observability must not break screening
        _client = None
        logger.warning("Langfuse tracing initialization failed; tracing remains off (%s)", type(exc).__name__)
        return False

    _initialized = True
    logger.info("Langfuse tracing enabled with input/output content redacted")
    return True


def flush_langfuse_observability() -> bool:
    """Flush pending spans when tracing is active; remain non-fatal on failure."""
    if not _initialized or _client is None:
        return False
    try:
        _client.flush()
    except Exception as exc:  # noqa: BLE001 - shutdown must continue
        logger.warning("Langfuse trace flush failed (%s)", type(exc).__name__)
        return False
    return True


def annotate_shion_trace(*, session_id: str, params: dict[str, Any]) -> bool:
    """Attach safe, low-cardinality request context to the current ADK trace.

    ADK/OpenInference creates the generation and tool observations.  This
    helper only adds operational context; it deliberately never forwards the
    case payload, company/asset names, prompts, replies, or numeric financial
    values.
    """
    if not _initialized or _client is None:
        return False
    try:
        _client.update_current_trace(
            name="shion-screening",
            session_id=session_id,
            metadata={
                "feature": "lease-screening",
                "route": "gunshi-stream",
                "input_redacted": "true",
                "has_asset_warnings": str(bool(params.get("asset_warnings"))).lower(),
                "has_default_warnings": str(bool(params.get("default_warnings"))).lower(),
            },
            tags=["shion", "screening", "google-adk"],
        )
    except Exception as exc:  # noqa: BLE001 - tracing must not break screening
        logger.warning("Langfuse trace annotation failed (%s)", type(exc).__name__)
        return False
    return True
