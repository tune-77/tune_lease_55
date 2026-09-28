"""REV-421: 紫苑リアルタイム音声通話（Gemini Live ephemeral token）試作。

ブラウザは Gemini Live API へ直接接続する。Cloud Run は音声を中継せず、
人格を固定した使い切りトークンの発行・記憶の想起・文字起こし保存だけを担う。
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/shion/voice", tags=["shion-voice"])

FALLBACK_MODEL = "gemini-2.5-flash-native-audio-latest"
_JST = ZoneInfo("Asia/Tokyo")
_VOICE_TAIL = (
    "\n\n【音声通話モード】\n"
    "いまは音声通話中。話し言葉で1〜3文に収め、記号・箇条書き・URLは読み上げない。"
    "過去の記憶が必要なときは recall_memory ツールを使う。"
)

_lock = threading.Lock()
_issued: dict[str, int] = {}  # JST日付 -> 発行数
_last_issued_at = 0.0
_model_cache: dict[str, str] = {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _require_enabled() -> None:
    if os.environ.get("SHION_VOICE_ENABLED", "0") != "1":
        raise HTTPException(status_code=404, detail="voice disabled")


def _client():
    from google import genai

    from api.secret_access import get_gemini_api_key

    return genai.Client(api_key=get_gemini_api_key(), http_options={"api_version": "v1alpha"})


def _pick_model(client) -> str:
    primary = os.environ.get("SHION_VOICE_MODEL", "gemini-3.8-live").strip() or "gemini-3.8-live"
    if primary not in _model_cache:
        try:
            client.models.get(model=primary)
            _model_cache[primary] = primary
        except Exception:
            logger.warning("shion voice: model %s unavailable, falling back to %s", primary, FALLBACK_MODEL)
            _model_cache[primary] = FALLBACK_MODEL
    return _model_cache[primary]


def _build_system_instruction(user_id: str) -> str:
    from api.chat_memory import get_recent_messages
    from api.prompt_generator import build_shion_system_prompt, load_mind

    now = dt.datetime.now(_JST).strftime("%Y-%m-%d %H:%M")
    prompt = build_shion_system_prompt(load_mind(), now)
    history = get_recent_messages(user_id, limit=10)
    if history:
        lines = [f"{m['role']}: {str(m['content'])[:200]}" for m in history]
        prompt += "\n\n【直近の会話】\n" + "\n".join(lines)
    return prompt + _VOICE_TAIL


class SessionRequest(BaseModel):
    user_id: str = Field(default="default", max_length=100)


class RecallRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class Turn(BaseModel):
    role: Literal["user", "model"]
    text: str


class TranscriptRequest(BaseModel):
    user_id: str = Field(default="default", max_length=100)
    turns: list[Turn] = Field(max_length=200)


@router.post("/session")
def create_voice_session(req: SessionRequest):
    """人格を固定した使い切りの Gemini Live トークンを発行する（回数・時間制限つき）。"""
    global _last_issued_at
    _require_enabled()
    max_seconds = _env_int("SHION_VOICE_MAX_SECONDS", 600)
    daily_limit = _env_int("SHION_VOICE_DAILY_LIMIT", 20)
    min_interval = _env_int("SHION_VOICE_MIN_INTERVAL_SECONDS", 20)

    with _lock:
        today = dt.datetime.now(_JST).date().isoformat()
        used = _issued.get(today, 0)
        if used >= daily_limit:
            raise HTTPException(status_code=429, detail="本日の音声通話の上限回数に達しました")
        if time.monotonic() - _last_issued_at < min_interval:
            raise HTTPException(status_code=429, detail="少し時間をおいてから再度お試しください")

        from google.genai import types

        try:
            client = _client()
            model = _pick_model(client)
            now = dt.datetime.now(dt.timezone.utc)
            token = client.auth_tokens.create(
                config=types.CreateAuthTokenConfig(
                    uses=1,
                    expire_time=now + dt.timedelta(seconds=max_seconds + 60),
                    new_session_expire_time=now + dt.timedelta(seconds=60),
                    live_connect_constraints=types.LiveConnectConstraints(
                        model=model,
                        config=types.LiveConnectConfig(
                            response_modalities=["AUDIO"],
                            system_instruction=_build_system_instruction(req.user_id),
                            input_audio_transcription=types.AudioTranscriptionConfig(),
                            output_audio_transcription=types.AudioTranscriptionConfig(),
                            speech_config=types.SpeechConfig(
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name=os.environ.get("SHION_VOICE_NAME", "Kore")
                                    )
                                )
                            ),
                            tools=[
                                types.Tool(
                                    function_declarations=[
                                        types.FunctionDeclaration(
                                            name="recall_memory",
                                            description="紫苑の過去の記憶・会話・判断メモを検索する",
                                            parameters=types.Schema(
                                                type="OBJECT",
                                                properties={"query": types.Schema(type="STRING")},
                                                required=["query"],
                                            ),
                                        )
                                    ]
                                )
                            ],
                        ),
                    ),
                    lock_additional_fields=[],
                )
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("shion voice: token mint failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=502, detail="音声セッションを開始できませんでした") from None

        _issued[today] = used + 1
        _last_issued_at = time.monotonic()

    return {
        "token": token.name,
        "model": model,
        "max_seconds": max_seconds,
        "remaining_today": daily_limit - used - 1,
    }


@router.post("/recall")
def recall_for_voice(req: RecallRequest):
    """Live の recall_memory ツール呼び出しに想起結果を返す。"""
    _require_enabled()
    from api.shion_memory_recall import build_recall_prompt_block

    text, _ = build_recall_prompt_block(req.query, limit=3)
    return {"result": text or "該当する記憶はありません"}


@router.post("/transcript")
def save_voice_transcript(req: TranscriptRequest):
    """通話の文字起こしを既存のチャット履歴へ保存する。"""
    _require_enabled()
    from api.chat_memory import save_message

    saved = 0
    for turn in req.turns:
        text = turn.text.strip()[:2000]
        if not text:
            continue
        save_message(req.user_id, "assistant" if turn.role == "model" else "user", text)
        saved += 1
    return {"saved": saved}
