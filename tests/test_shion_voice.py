"""REV-421: 紫苑音声通話トークン発行APIのテスト。"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.routers.shion_voice as sv


class _FakeClient:
    def __init__(self, model_ok=True):
        self.created = []
        self.models = SimpleNamespace(get=self._get)
        self.auth_tokens = SimpleNamespace(create=self._create)
        self._model_ok = model_ok

    def _get(self, model):
        if not self._model_ok:
            raise RuntimeError("not found")

    def _create(self, config):
        self.created.append(config)
        return SimpleNamespace(name="auth_tokens/abc")


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("SHION_VOICE_ENABLED", "1")
    monkeypatch.setenv("SHION_VOICE_MIN_INTERVAL_SECONDS", "0")
    monkeypatch.setattr(sv, "_issued", {})
    monkeypatch.setattr(sv, "_last_issued_at", 0.0)
    monkeypatch.setattr(sv, "_model_cache", {})
    monkeypatch.setattr(sv, "_build_system_instruction", lambda user_id: "紫苑です")
    fake = _FakeClient()
    monkeypatch.setattr(sv, "_client", lambda: fake)
    app = FastAPI()
    app.include_router(sv.router)
    return TestClient(app), fake


def test_disabled_returns_404(setup, monkeypatch):
    client, _ = setup
    monkeypatch.setenv("SHION_VOICE_ENABLED", "0")
    assert client.post("/api/shion/voice/session", json={}).status_code == 404


def test_session_token_is_locked_single_use(setup):
    client, fake = setup
    res = client.post("/api/shion/voice/session", json={"user_id": "u"})
    assert res.status_code == 200
    body = res.json()
    assert body["token"] == "auth_tokens/abc"
    assert body["model"] == "gemini-3.8-live"
    assert body["max_seconds"] == 600
    cfg = fake.created[0]
    assert cfg.uses == 1
    assert (cfg.expire_time - cfg.new_session_expire_time).total_seconds() == pytest.approx(600, abs=2)
    live = cfg.live_connect_constraints.config
    assert live.system_instruction == "紫苑です"
    assert live.input_audio_transcription is not None
    assert live.output_audio_transcription is not None
    assert cfg.lock_additional_fields == []


def test_fallback_model_when_primary_missing(setup, monkeypatch):
    client, _ = setup
    fake = _FakeClient(model_ok=False)
    monkeypatch.setattr(sv, "_client", lambda: fake)
    assert client.post("/api/shion/voice/session", json={}).json()["model"] == sv.FALLBACK_MODEL


def test_daily_limit(setup, monkeypatch):
    client, _ = setup
    monkeypatch.setenv("SHION_VOICE_DAILY_LIMIT", "2")
    assert client.post("/api/shion/voice/session", json={}).json()["remaining_today"] == 1
    assert client.post("/api/shion/voice/session", json={}).status_code == 200
    assert client.post("/api/shion/voice/session", json={}).status_code == 429


def test_min_interval(setup, monkeypatch):
    client, _ = setup
    monkeypatch.setenv("SHION_VOICE_MIN_INTERVAL_SECONDS", "60")
    assert client.post("/api/shion/voice/session", json={}).status_code == 200
    assert client.post("/api/shion/voice/session", json={}).status_code == 429


def test_mint_failure_does_not_count(setup, monkeypatch):
    client, _ = setup

    def boom():
        raise RuntimeError("secret-ish detail")

    monkeypatch.setattr(sv, "_client", boom)
    res = client.post("/api/shion/voice/session", json={})
    assert res.status_code == 502
    assert "secret" not in res.text
    assert sv._issued == {}


def test_transcript_maps_roles_and_truncates(setup, monkeypatch):
    client, _ = setup
    saved = []
    monkeypatch.setattr("api.chat_memory.save_message", lambda u, r, t: saved.append((u, r, t)))
    res = client.post(
        "/api/shion/voice/transcript",
        json={"user_id": "u", "turns": [
            {"role": "user", "text": " こんにちは "},
            {"role": "model", "text": "あ" * 3000},
            {"role": "user", "text": "   "},
        ]},
    )
    assert res.json() == {"saved": 2}
    assert saved[0] == ("u", "user", "こんにちは")
    assert saved[1][1] == "assistant" and len(saved[1][2]) == 2000


def test_recall_empty_fallback(setup, monkeypatch):
    client, _ = setup
    monkeypatch.setattr("api.shion_memory_recall.build_recall_prompt_block", lambda q, limit: ("", {}))
    assert client.post("/api/shion/voice/recall", json={"query": "前回"}).json() == {"result": "該当する記憶はありません"}
