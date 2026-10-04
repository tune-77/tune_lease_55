from __future__ import annotations

import json

import pytest

from ai_runtime_client import (
    anthropic_client,
    extract_token_usage,
    google_genai_client,
    tracked_ai_call,
    tracked_ai_http_call,
)
from scripts.report_ai_usage import summarize


def _read_entries(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_google_client_records_metadata_without_content(tmp_path, monkeypatch):
    log_path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("AI_USAGE_LOG_PATH", str(log_path))

    class Usage:
        prompt_token_count = 12
        candidates_token_count = 5
        total_token_count = 17

    class Models:
        def generate_content(self, **_kwargs):
            return type("Response", (), {"text": "private answer", "usage_metadata": Usage()})()

    raw = type("Client", (), {"models": Models()})()
    client = google_genai_client(feature="screening_chat", client_factory=lambda **_: raw)
    response = client.models.generate_content(model="gemini-test", contents="private prompt")

    assert response.text == "private answer"
    [entry] = _read_entries(log_path)
    assert entry["provider"] == "google"
    assert entry["feature"] == "screening_chat"
    assert entry["model"] == "gemini-test"
    assert entry["input_tokens"] == 12
    assert entry["output_tokens"] == 5
    assert entry["total_tokens"] == 17
    serialized = json.dumps(entry, ensure_ascii=False)
    assert "private prompt" not in serialized
    assert "private answer" not in serialized


def test_failure_records_only_exception_type(tmp_path, monkeypatch):
    log_path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("AI_USAGE_LOG_PATH", str(log_path))

    def fail():
        raise RuntimeError("secret customer text")

    with pytest.raises(RuntimeError, match="secret customer text"):
        tracked_ai_call(
            fail,
            provider="google",
            model="gemini-test",
            feature="test",
        )

    [entry] = _read_entries(log_path)
    assert entry["ok"] is False
    assert entry["error_type"] == "RuntimeError"
    assert "secret customer text" not in json.dumps(entry, ensure_ascii=False)


def test_anthropic_usage_is_normalized(tmp_path, monkeypatch):
    log_path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("AI_USAGE_LOG_PATH", str(log_path))

    usage = type("Usage", (), {"input_tokens": 8, "output_tokens": 3})()
    response = type("Response", (), {"usage": usage})()
    messages = type("Messages", (), {"create": lambda self, **_: response})()
    raw = type("Client", (), {"messages": messages})()
    client = anthropic_client(feature="mebuki", client_factory=lambda **_: raw)

    assert client.messages.create(model="claude-test") is response
    [entry] = _read_entries(log_path)
    assert entry["provider"] == "anthropic"
    assert entry["input_tokens"] == 8
    assert entry["output_tokens"] == 3
    assert entry["total_tokens"] == 11


def test_extract_token_usage_accepts_rest_dict():
    assert extract_token_usage(
        {"usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 6, "totalTokenCount": 10}}
    ) == {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10}


def test_http_call_records_usage_and_preserves_response(tmp_path, monkeypatch):
    log_path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("AI_USAGE_LOG_PATH", str(log_path))

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 7}}

    response = Response()
    actual = tracked_ai_http_call(
        lambda: response,
        provider="google",
        model="gemini-rest",
        feature="rest_test",
    )

    assert actual is response
    [entry] = _read_entries(log_path)
    assert entry["total_tokens"] == 9


def test_usage_log_rotates_at_configured_size(tmp_path, monkeypatch):
    log_path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("AI_USAGE_LOG_PATH", str(log_path))
    monkeypatch.setenv("AI_USAGE_LOG_MAX_BYTES", "1")

    for _ in range(2):
        tracked_ai_call(
            lambda: {"usageMetadata": {"totalTokenCount": 1}},
            provider="google",
            model="gemini-test",
            feature="rotation_test",
        )

    rotated = log_path.with_suffix(".jsonl.1")
    assert log_path.exists()
    assert rotated.exists()
    assert len(_read_entries(log_path)) == 1
    assert len(_read_entries(rotated)) == 1


def test_usage_report_includes_rotated_generation(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    rotated = log_path.with_suffix(".jsonl.1")
    rotated.write_text(
        json.dumps(
            {
                "provider": "google",
                "feature": "screening_chat",
                "model": "gemini-test",
                "ok": True,
                "duration_ms": 10,
                "input_tokens": 3,
                "output_tokens": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log_path.write_text(
        json.dumps(
            {
                "provider": "google",
                "feature": "screening_chat",
                "model": "gemini-test",
                "ok": False,
                "duration_ms": 30,
                "input_tokens": 7,
                "output_tokens": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    [row] = summarize(log_path)

    assert row["calls"] == 2
    assert row["errors"] == 1
    assert row["avg_duration_ms"] == 20.0
    assert row["input_tokens"] == 10
    assert row["output_tokens"] == 3
