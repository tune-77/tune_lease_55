import requests

import api.usage_loop_engineering as usage_loop


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "generated") -> None:
        self.status_code = status_code
        self._text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self) -> dict:
        return {"candidates": [{"content": {"parts": [{"text": self._text}]}}]}


def test_call_gemini_retries_ssl_error(tmp_path, monkeypatch):
    calls = 0

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise requests.exceptions.SSLError("temporary SSL error")
        return _FakeResponse(200, " recovered ")

    monkeypatch.setattr(usage_loop, "_gemini_api_key", lambda: "x")
    monkeypatch.setattr(usage_loop, "_GEMINI_RETRY_DELAYS_S", (0, 0))
    monkeypatch.setattr(requests, "post", fake_post)

    assert usage_loop._call_gemini("prompt") == "recovered"
    assert calls == 3


def test_call_gemini_does_not_retry_400(monkeypatch):
    calls = 0

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _FakeResponse(400)

    monkeypatch.setattr(usage_loop, "_gemini_api_key", lambda: "x")
    monkeypatch.setattr(usage_loop, "_GEMINI_RETRY_DELAYS_S", (0, 0))
    monkeypatch.setattr(requests, "post", fake_post)

    try:
        usage_loop._call_gemini("prompt")
    except requests.exceptions.HTTPError:
        pass
    else:
        raise AssertionError("HTTPError が送出されませんでした")
    assert calls == 1
