"""experiments/clef_vs_jev/clef_client.py: Cloudflare の封筒を剥がして Jev と同じ形で返す。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "clef_vs_jev"))

import clef_client  # noqa: E402


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


def test_request_unwraps_result_and_hides_token(monkeypatch):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, body=json)
        return _Resp(200, {"success": True, "result": {"answers": {"q": {"type": "noul", "noul": 0.9}},
                                                       "usage": {"input_tokens": 10}}})

    monkeypatch.setattr(clef_client.httpx, "post", fake_post)
    client = clef_client.ClefClient("secret-token", "acc")
    body = client.request({"state": {}, "model": "jev-latest", "questions": {}}, "clef-flash")

    assert seen["url"].endswith("/accounts/acc/ai/run/@cf/cloudflare/clef-flash")
    assert seen["body"]["model"] == "clef-flash"
    assert body["answers"]["q"]["noul"] == 0.9 and body["model"] == "clef-flash"
    assert "secret-token" not in repr(client)


def test_request_raises_on_error_envelope(monkeypatch):
    monkeypatch.setattr(clef_client.httpx, "post",
                        lambda *a, **k: _Resp(401, {"success": False, "errors": [{"message": "Authentication error"}]}))
    with pytest.raises(clef_client.ClefError, match="401"):
        clef_client.ClefClient("t", "acc").request({"questions": {}}, "clef")
