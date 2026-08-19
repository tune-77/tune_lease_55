"""Shared Vertex AI plain text generation for Obsidian curator scripts.

Reuses the existing Vertex auth/config/SDK plumbing from
``api/vertex_agent_search.py`` (project id, location, ``_access_token()``)
instead of introducing a separate Gemini API key. This module intentionally
has no ``tools=[...]`` (no Search grounding) — for buzz/web checks, call
``api.vertex_agent_search.google_search_grounding`` directly.

Every failure mode (missing credentials, network, SDK not installed) degrades
to a skipped/error payload; it never raises into a caller script.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "gemini-2.5-flash"


def _compact_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def generate_text(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.4,
    max_output_tokens: int = 1024,
) -> dict[str, Any]:
    """Generate plain text via Vertex AI. Never raises."""
    prompt = str(prompt or "").strip()
    if not prompt:
        return {"used": False, "status": "empty_prompt", "text": "", "error": ""}

    try:
        from api.vertex_agent_search import get_config, _access_token  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - import failure is environmental
        return {"used": False, "status": "unavailable", "text": "", "error": _compact_text(exc, 240)}

    config = get_config()
    if not config.enabled or not config.project_id:
        return {"used": False, "status": "not_configured", "text": "", "error": ""}

    model_id = model or os.environ.get("VERTEX_TEXT_GEN_MODEL") or DEFAULT_MODEL
    location = os.environ.get("VERTEX_AI_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"

    try:
        from google import genai  # type: ignore[import-not-found]
        from google.genai.types import GenerateContentConfig, HttpOptions  # type: ignore[import-not-found]
        from google.oauth2.credentials import Credentials as _OAuthCredentials  # type: ignore[import-not-found]

        token = _access_token()
        credentials = _OAuthCredentials(token=token) if token else None
        client = genai.Client(
            vertexai=True,
            project=config.project_id,
            location=location,
            credentials=credentials,
            http_options=HttpOptions(api_version="v1"),
        )
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        text = _compact_text(getattr(response, "text", "") or "", 6000)
        return {"used": bool(text), "status": "ok" if text else "empty_response", "text": text, "error": ""}
    except Exception as exc:
        return {"used": False, "status": "error", "text": "", "error": _compact_text(exc, 240)}
