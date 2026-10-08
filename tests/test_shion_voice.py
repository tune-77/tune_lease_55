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
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr("api.user_affect_memory._state_path", lambda: tmp_path / "user_affect_state.json")
    monkeypatch.setattr("api.shion_relationship._STATE_PATH", tmp_path / "relationship.json")
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


def test_transcript_records_voice_affect(setup, monkeypatch):
    client, _ = setup
    monkeypatch.setattr("api.chat_memory.save_message", lambda u, r, t: None)
    client.post(
        "/api/shion/voice/transcript",
        json={"user_id": "u", "turns": [
            {"role": "user", "text": "今日はもう疲れた…"},
            {"role": "model", "text": "やった！嬉しい！"},  # 紫苑側の発言は推定に使わない
        ]},
    )
    from api.user_affect_memory import recall_user_affect

    assert recall_user_affect("u").label == "疲れ"
    from api.shion_relationship import get_relationship_state

    assert get_relationship_state()["total_interactions"] == 1  # REV-467: 通話も関係性に記録


def test_session_enables_affective_only_for_supported_models(setup, monkeypatch):
    client, fake = setup
    client.post("/api/shion/voice/session", json={"user_id": "u"})
    assert fake.created[-1].live_connect_constraints.config.enable_affective_dialog is None  # gemini-3.8-live

    assert sv._affective_enabled("gemini-2.5-flash-native-audio-latest") is True
    assert sv._affective_enabled("gemini-3.8-live") is False
    monkeypatch.setenv("SHION_VOICE_AFFECTIVE", "1")
    assert sv._affective_enabled("gemini-3.8-live") is True
    monkeypatch.setenv("SHION_VOICE_AFFECTIVE", "0")
    assert sv._affective_enabled("gemini-2.5-flash-native-audio-latest") is False


def test_system_instruction_includes_voice_tone_and_affect_memory(monkeypatch, tmp_path):
    from datetime import datetime, timedelta

    from api.user_affect_memory import record_user_affect

    path = tmp_path / "s.json"
    monkeypatch.setattr("api.user_affect_memory._state_path", lambda: path)
    record_user_affect("u", "不安", 0.9, now=datetime.now() - timedelta(hours=5), path=path)
    monkeypatch.setattr("api.chat_memory.get_recent_messages", lambda user_id, limit=10: [])
    monkeypatch.setattr("api.prompt_generator.load_mind", lambda: {})
    monkeypatch.setattr("api.prompt_generator.build_shion_system_prompt", lambda mind, now: "紫苑")
    prompt = sv._build_system_instruction("u")
    assert "声の調子に合わせる" in prompt
    assert "相手の最近の様子" in prompt and "不安" in prompt


def test_recall_empty_fallback(setup, monkeypatch):
    client, _ = setup
    monkeypatch.setattr("api.shion_memory_recall.build_recall_prompt_block", lambda q, limit: ("", {}))
    assert client.post("/api/shion/voice/recall", json={"query": "前回"}).json() == {"result": "該当する記憶はありません"}


# ── REV-460: 歌唱 ──

_SCORE = {"title": "朝の歌", "bpm": 120, "notes": [
    {"lyric": "あ", "key": 60, "beats": 1},
    {"lyric": "さ", "key": 62, "beats": 0.5},
    {"lyric": "", "key": None, "beats": 1},
    {"lyric": "きょ", "key": 64, "beats": 2},
    {"lyric": "う", "key": 65, "beats": 1},
]}


@pytest.fixture
def sing_setup(monkeypatch):
    monkeypatch.setenv("SHION_SING_ENABLED", "1")
    monkeypatch.setenv("SHION_SING_MIN_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("SHION_SING_ACCOMPANIMENT", "0")
    monkeypatch.setattr(sv, "_sung", {})
    monkeypatch.setattr(sv, "_last_sung_at", 0.0)
    monkeypatch.setattr(sv, "_generate_score", lambda theme: _SCORE)
    calls = []

    def fake_synth(score, teacher_id, voice_id):
        calls.append((score, teacher_id, voice_id))
        return b"RIFFwav"

    monkeypatch.setattr(sv, "_synthesize", fake_synth)
    monkeypatch.setattr(sv, "_credit", lambda t, v: "VOICEVOX:テスト")
    app = FastAPI()
    app.include_router(sv.router)
    return TestClient(app), calls


def test_sing_disabled_returns_404(sing_setup, monkeypatch):
    client, _ = sing_setup
    monkeypatch.setenv("SHION_SING_ENABLED", "0")
    assert client.post("/api/shion/voice/sing", json={"theme": "朝"}).status_code == 404


def test_sing_returns_wav_and_credit(sing_setup, monkeypatch):
    client, calls = sing_setup
    monkeypatch.setenv("SHION_SING_VOICE_ID", "3002")
    body = client.post("/api/shion/voice/sing", json={"theme": "朝"}).json()
    assert body["audio_base64"] == "UklGRndhdg=="
    assert body["credit"] == "VOICEVOX:テスト"
    assert body["lyrics"] == "あさきょう"
    score, teacher_id, voice_id = calls[0]
    assert (teacher_id, voice_id) == (6000, 3002)
    notes = score["notes"]
    assert notes[0] == {"key": None, "frame_length": 15, "lyric": ""}
    assert notes[1] == {"key": 62, "frame_length": round(0.5 * 93.75), "lyric": "あ"}  # 1拍@120bpm、既定で+2半音
    assert notes[3]["key"] is None and notes[-1]["key"] is None


def test_sing_key_shift_env(sing_setup, monkeypatch):
    client, calls = sing_setup
    monkeypatch.setenv("SHION_SING_KEY_SHIFT", "0")
    client.post("/api/shion/voice/sing", json={"theme": "朝"})
    monkeypatch.setenv("SHION_SING_KEY_SHIFT", "99")  # 上げすぎは +5 で頭打ち
    client.post("/api/shion/voice/sing", json={"theme": "朝"})
    assert [score["notes"][1]["key"] for score, _, _ in calls] == [60, 65]


def test_sing_mixes_accompaniment(sing_setup, monkeypatch):
    import api.shion_sing_accompaniment as acc

    client, calls = sing_setup
    monkeypatch.setenv("SHION_SING_ACCOMPANIMENT", "1")
    seen = []
    monkeypatch.setattr(acc, "add_accompaniment",
                        lambda wav, vv, chords, bpm, key_shift: seen.append((wav, chords, bpm, key_shift)) or b"MIXED")
    body = client.post("/api/shion/voice/sing", json={"theme": "朝"}).json()
    assert body["audio_base64"] == "TUlYRUQ=" and body["accompaniment"] is True
    assert seen == [(b"RIFFwav", None, 120, 2)]  # 伴奏も歌と同じだけ移調
    monkeypatch.setattr(acc, "add_accompaniment", lambda *a, **kw: None)  # 伴奏失敗は歌声だけ
    body = client.post("/api/shion/voice/sing", json={"theme": "朝"}).json()
    assert body["audio_base64"] == "UklGRndhdg==" and body["accompaniment"] is False


def test_to_voicevox_score_clamps_bad_llm_output():
    score = {"bpm": 999, "notes": [{"lyric": "ら", "key": 90, "beats": 10}] * 3
             + [{"lyric": "ー", "key": 60, "beats": 1}, {"lyric": "っ", "key": 60, "beats": 1}]
             + [{"lyric": "ら", "key": 30, "beats": 0.01}] * 100}
    vv, lyrics = sv._to_voicevox_score(score)
    sung = [n for n in vv["notes"] if n["key"] is not None]
    assert all(sv._KEY_MIN <= n["key"] <= sv._KEY_MAX for n in sung)
    assert len(vv["notes"]) == sv._MAX_NOTES + 2  # 先頭・末尾の休符
    assert vv["notes"][1]["frame_length"] == round(4 * 60 / 180 * 93.75)
    assert vv["notes"][4]["key"] is None and vv["notes"][5]["key"] is None  # 長音・促音は休符化
    assert lyrics == "ら" * (sv._MAX_NOTES - 2)


def test_to_voicevox_score_rejects_too_few_notes():
    with pytest.raises(ValueError):
        sv._to_voicevox_score({"notes": [{"lyric": "ら", "key": 60, "beats": 1}]})


def test_sing_daily_limit(sing_setup, monkeypatch):
    client, _ = sing_setup
    monkeypatch.setenv("SHION_SING_DAILY_LIMIT", "1")
    assert client.post("/api/shion/voice/sing", json={"theme": "朝"}).json()["remaining_today"] == 0
    assert client.post("/api/shion/voice/sing", json={"theme": "朝"}).status_code == 429


def test_sing_busy_returns_429(sing_setup):
    client, _ = sing_setup
    assert sv._sing_lock.acquire(blocking=False)
    try:
        assert client.post("/api/shion/voice/sing", json={"theme": "朝"}).status_code == 429
    finally:
        sv._sing_lock.release()


def test_sing_voicevox_failure_does_not_count(sing_setup, monkeypatch):
    client, _ = sing_setup

    def boom(*args):
        raise ConnectionError("refused")

    monkeypatch.setattr(sv, "_synthesize", boom)
    res = client.post("/api/shion/voice/sing", json={"theme": "朝"})
    assert res.status_code == 503
    assert sv._sung == {}
    assert not sv._sing_lock.locked()


# ── REV-462: 対話の発言から歌う ──

@pytest.mark.parametrize("message,theme", [
    ("歌って", ""),
    ("何か歌って！", ""),
    ("紫苑、秋の朝の歌を歌って", "秋の朝"),
    ("雨について歌ってほしい", "雨"),
    ("猫をテーマに一曲歌ってよ", "猫"),
    ("リース審査の歌を聞かせて", "リース審査"),
    ("夏の海で一曲お願い", "夏の海"),
])
def test_extract_theme(message, theme):
    assert sv._extract_theme(message) == theme


def test_sing_from_dialogue_message(sing_setup, monkeypatch):
    client, _ = sing_setup
    themes = []
    monkeypatch.setattr(sv, "_generate_score", lambda theme: themes.append(theme) or _SCORE)
    body = client.post("/api/shion/voice/sing", json={"message": "秋の朝の歌を歌って"}).json()
    assert themes == ["秋の朝"] and body["theme"] == "秋の朝"
    body = client.post("/api/shion/voice/sing", json={"message": "歌って"}).json()
    assert themes[-1] == "" and body["title"] == "朝の歌"


def test_generate_score_lets_shion_choose_theme(monkeypatch):
    import api.loop_engineering_common as lec
    import api.prompt_generator as pg

    prompts = []
    monkeypatch.setattr(pg, "load_mind", lambda: {})
    monkeypatch.setattr(pg, "build_shion_system_prompt", lambda mind, now: "紫苑")
    monkeypatch.setattr(lec, "call_gemini_json", lambda prompt, **kw: prompts.append(prompt) or {})
    sv._generate_score("")
    sv._generate_score("雨")
    assert "自由に決め" in prompts[0] and "テーマ「雨」" in prompts[1]


# ── REV-463: 読み上げ ──

@pytest.fixture
def tts_client(monkeypatch):
    monkeypatch.setenv("SHION_TTS_ENABLED", "1")
    calls = []
    monkeypatch.setattr(sv, "_tts_synthesize", lambda text, speaker: calls.append((text, speaker)) or b"RIFFtts")
    monkeypatch.setattr(sv, "_tts_credit", lambda speaker: "VOICEVOX:冥鳴ひまり")
    app = FastAPI()
    app.include_router(sv.router)
    return TestClient(app), calls


def test_tts_returns_wav_with_speaker_from_env(tts_client, monkeypatch):
    client, calls = tts_client
    monkeypatch.setenv("SHION_TTS_SPEAKER_ID", "9")
    res = client.post("/api/shion/voice/tts", json={"text": " こんにちは。 "})
    assert res.status_code == 200 and res.content == b"RIFFtts"
    assert res.headers["content-type"] == "audio/wav"
    assert calls == [("こんにちは。", 9)]
    from urllib.parse import unquote
    assert unquote(res.headers["x-voice-credit"]) == "VOICEVOX:冥鳴ひまり"


def test_tts_default_speaker_and_disabled(tts_client, monkeypatch):
    client, calls = tts_client
    client.post("/api/shion/voice/tts", json={"text": "はい"})
    assert calls[-1][1] == 14
    monkeypatch.setenv("SHION_TTS_ENABLED", "0")
    assert client.post("/api/shion/voice/tts", json={"text": "はい"}).status_code == 404


def test_tts_engine_down_returns_503(tts_client, monkeypatch):
    client, _ = tts_client

    def boom(*args):
        raise ConnectionError("refused")

    monkeypatch.setattr(sv, "_tts_synthesize", boom)
    assert client.post("/api/shion/voice/tts", json={"text": "はい"}).status_code == 503
    assert client.post("/api/shion/voice/tts", json={"text": "あ" * 301}).status_code == 422


@pytest.mark.parametrize(
    ("engine", "tts", "expected"),
    [("", "1", ("kore", True)), ("himari", "1", ("himari", True)), ("himari", "0", ("kore", False)), ("bogus", "1", ("kore", True))],
)
def test_session_reports_voice_engine_default(setup, monkeypatch, engine, tts, expected):
    """REV-530: 既定は Kore。ひまり（VOICEVOX）は /tts が有効な時だけ既定にできる。"""
    client, _ = setup
    monkeypatch.setenv("SHION_VOICE_ENGINE", engine)
    monkeypatch.setenv("SHION_TTS_ENABLED", tts)
    body = client.post("/api/shion/voice/session", json={}).json()
    assert (body["default_voice"], body["himari_available"]) == expected
