"""Small helpers for brittle LLM JSON responses.

The goal is to keep prompts short and structured, then recover complete fields
when the model wraps JSON in fences or stops mid-object.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any


class LLMJsonError(ValueError):
    def __init__(self, message: str, raw: str = "", finish_reason: str = "") -> None:
        super().__init__(message)
        self.raw = raw
        self.finish_reason = finish_reason


def extract_candidate_text(response_json: dict[str, Any]) -> tuple[str, str]:
    candidate = (response_json.get("candidates") or [{}])[0]
    finish_reason = str(candidate.get("finishReason") or "")
    parts = ((candidate.get("content") or {}).get("parts") or [])
    text = ""
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text += part["text"]
    return text.strip(), finish_reason


def strip_json_fences(raw: str) -> str:
    text = (raw or "").strip()
    if "```" not in text:
        return text
    parts = text.split("```")
    for part in parts[1::2]:
        cleaned = re.sub(r"^\s*json\s*", "", part, flags=re.IGNORECASE).strip()
        if cleaned:
            return cleaned
    return text


def parse_json_object(raw: str) -> dict[str, Any]:
    text = strip_json_fences(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start : end + 1])
        else:
            raise
    if not isinstance(parsed, dict):
        raise LLMJsonError("LLM response is not a JSON object", raw)
    return parsed


def _field_fragment(raw: str, field: str) -> str:
    text = strip_json_fences(raw)
    match = re.search(rf'"{re.escape(field)}"\s*:\s*', text)
    if not match:
        return ""
    return text[match.end() :]


def _recover_string(raw: str, field: str) -> str | None:
    fragment = _field_fragment(raw, field).lstrip()
    match = re.match(r'"((?:\\.|[^"\\])*)"', fragment, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return None


def _recover_array(raw: str, field: str) -> list[str] | None:
    fragment = _field_fragment(raw, field).lstrip()
    if not fragment.startswith("["):
        return None
    closed = fragment.find("]")
    array_text = fragment[: closed + 1] if closed >= 0 else fragment
    values: list[str] = []
    for match in re.finditer(r'"((?:\\.|[^"\\])*)"', array_text, flags=re.DOTALL):
        try:
            values.append(json.loads(f'"{match.group(1)}"'))
        except json.JSONDecodeError:
            continue
    return values


def _recover_number(raw: str, field: str) -> int | float | None:
    fragment = _field_fragment(raw, field).lstrip()
    if fragment.startswith("null"):
        return None
    match = re.match(r"-?\d+(?:\.\d+)?", fragment)
    if not match:
        return None
    value = match.group(0)
    return float(value) if "." in value else int(value)


def _recover_bool(raw: str, field: str) -> bool | None:
    fragment = _field_fragment(raw, field).lstrip()
    if fragment.startswith("true"):
        return True
    if fragment.startswith("false"):
        return False
    return None


def parse_or_recover_json(
    raw: str,
    *,
    defaults: dict[str, Any],
    string_fields: set[str] | None = None,
    array_fields: set[str] | None = None,
    number_fields: set[str] | None = None,
    bool_fields: set[str] | None = None,
) -> tuple[dict[str, Any], bool]:
    try:
        parsed = parse_json_object(raw)
        return {**copy.deepcopy(defaults), **parsed}, False
    except (json.JSONDecodeError, LLMJsonError):
        recovered = copy.deepcopy(defaults)
        for field in string_fields or set():
            value = _recover_string(raw, field)
            if value is not None:
                recovered[field] = value
        for field in array_fields or set():
            value = _recover_array(raw, field)
            if value is not None:
                recovered[field] = value
        for field in number_fields or set():
            if f'"{field}"' in raw:
                recovered[field] = _recover_number(raw, field)
        for field in bool_fields or set():
            value = _recover_bool(raw, field)
            if value is not None:
                recovered[field] = value
        recovered["_json_recovered"] = True
        return recovered, True


def with_retry_tokens(payload: dict[str, Any], max_output_tokens: int) -> dict[str, Any]:
    retried = copy.deepcopy(payload)
    generation_config = retried.setdefault("generationConfig", {})
    generation_config["maxOutputTokens"] = max_output_tokens
    generation_config["temperature"] = min(float(generation_config.get("temperature", 0.2)), 0.2)
    generation_config["responseMimeType"] = "application/json"
    return retried
