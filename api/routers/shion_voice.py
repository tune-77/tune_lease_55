"""REV-421: 紫苑リアルタイム音声通話（Gemini Live ephemeral token）試作。

ブラウザは Gemini Live API へ直接接続する。Cloud Run は音声を中継せず、
人格を固定した使い切りトークンの発行・記憶の想起・文字起こし保存だけを担う。
"""

from __future__ import annotations

import base64
import datetime as dt
import logging
import os
import re
import threading
import time
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/shion/voice", tags=["shion-voice"])

FALLBACK_MODEL = "gemini-3.1-flash-live-preview"
_JST = ZoneInfo("Asia/Tokyo")
_VOICE_TAIL = (
    "\n\n【音声通話モード】\n"
    "いまは音声通話中。話し言葉で1〜3文に収め、記号・箇条書き・URLは読み上げない。"
    "過去の記憶が必要なときは recall_memory ツールを使う。"
    "\n\n【声の調子に合わせる（REV-466）】\n"
    "言葉の内容だけでなく、声の速さ・大きさ・明るさ・沈み・ため息・言いよどみも聞いて、相手の今の状態を感じ取る。"
    "疲れや沈みが聞こえたら、ゆっくり柔らかく1〜2文で受け止めるだけにし、仕事や明日の段取りの話へ戻さない。"
    "急いでいる声なら前置きを省いて結論から、てきぱきと。"
    "弾んだ声なら一緒に喜ぶ明るい温度で。苛立ちが聞こえたら、謝りすぎずに落ち着いて要点だけ返す。"
    "不安そうな声なら、落ち着いた声で確かなことと確認が要ることを分けて伝える。"
    "声から感じた気持ちを「疲れていますね」のように決めつけて口にしない。返し方にだけ反映する。"
    "事実・審査判断・必要な注意は気持ちに合わせて変えない。"
)
# affective dialog を受け付けると実測できたモデル（2026-10-05 時点）。gemini-3.8-live は
# ドキュメント上は対応だが、実音声を送ると 1007 invalid argument で切断されるため含めない。
_AFFECTIVE_MODEL_PREFIXES = ("gemini-2.5-flash-native-audio",)

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


def _affective_enabled(model: str) -> bool:
    """SHION_VOICE_AFFECTIVE: auto（既定・対応モデルだけ）/ 1（強制）/ 0（無効）。"""
    mode = os.environ.get("SHION_VOICE_AFFECTIVE", "auto").strip().lower()
    if mode in {"1", "true", "on"}:
        return True
    if mode in {"0", "false", "off"}:
        return False
    return model.startswith(_AFFECTIVE_MODEL_PREFIXES)


def _affect_memory_block(user_id: str) -> str:
    """前回までの相手の様子（REV-465）を通話の文脈へ足す。失敗しても通話は始める。"""
    try:
        from api.user_affect_memory import build_user_affect_memory_block, recall_user_affect

        return build_user_affect_memory_block(recall_user_affect(user_id))
    except Exception as exc:
        logger.warning("shion voice: affect memory unavailable (%s)", type(exc).__name__)
        return ""


def _build_system_instruction(user_id: str) -> str:
    from api.chat_memory import get_recent_messages
    from api.prompt_generator import build_shion_system_prompt, load_mind

    now = dt.datetime.now(_JST).strftime("%Y-%m-%d %H:%M")
    prompt = build_shion_system_prompt(load_mind(), now)
    history = get_recent_messages(user_id, limit=10)
    if history:
        lines = [f"{m['role']}: {str(m['content'])[:200]}" for m in history]
        prompt += "\n\n【直近の会話】\n" + "\n".join(lines)
    affect_block = _affect_memory_block(user_id)
    if affect_block:
        prompt += "\n\n" + affect_block
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
                            enable_affective_dialog=True if _affective_enabled(model) else None,
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
        **_voice_engine_options(),
    }


def _voice_engine_options() -> dict:
    """通話の声（REV-530）。kore=Gemini Live の声、himari=文字起こしを VOICEVOX（冥鳴ひまり）で読み上げ。

    Live のモデルはテキスト出力に対応しない（1007）ため、ひまりでも音声で受けて文字起こしを使う。
    既定は SHION_VOICE_ENGINE（kore）。ひまりは /tts（SHION_TTS_ENABLED=1）が使える時だけ選べる。
    """
    himari = os.environ.get("SHION_TTS_ENABLED", "0") == "1"
    default = os.environ.get("SHION_VOICE_ENGINE", "kore").strip().lower()
    return {"default_voice": default if default == "himari" and himari else "kore", "himari_available": himari}


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
    _remember_voice_affect(req.user_id, [t.text for t in req.turns if t.role == "user"])
    return {"saved": saved}


def _remember_voice_affect(user_id: str, user_texts: list[str]) -> None:
    """通話の発言（文字起こし）から様子を推定し、相手ごとの記憶に残す（REV-465/466）。"""
    text = "\n".join(t.strip() for t in user_texts if t.strip())[-1200:]
    if not text:
        return
    try:
        from api.user_affect import estimate_user_affect
        from api.user_affect_memory import record_user_affect

        from api.user_affect import record_relationship_from_affect

        affect = estimate_user_affect(text)
        record_user_affect(user_id, affect.label, affect.intensity, surface="voice")
        # REV-467: 通話1回を関係性スコアへ記録（声で話した日も「沈黙」扱いにしない）
        record_relationship_from_affect(affect.to_payload(), topic_depth="normal")
    except Exception as exc:
        logger.warning("shion voice: affect record failed (%s)", type(exc).__name__)


# ── REV-460: 歌唱（Gemini が楽譜を作り、ローカル VOICEVOX ENGINE が歌う） ──

_FRAME_RATE = 93.75  # VOICEVOX 歌唱APIの既定フレームレート
_KEY_MIN, _KEY_MAX = 57, 76  # 女性ボーカルの中音域（MIDI）
_MAX_NOTES = 64
# 1モーラ（拗音・小書き母音つき可）。小書きかな・長音・促音で始まるものは不可
_MORA_RE = re.compile(r"^[あ-ゔア-ヴ](?<![ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ])[ぁぃぅぇぉゃゅょァィゥェォャュョ]?$")

_sing_lock = threading.Lock()
_sung: dict[str, int] = {}  # JST日付 -> 歌唱回数
_last_sung_at = 0.0
_credit_cache: dict[tuple[int, int], str] = {}


class SingRequest(BaseModel):
    theme: str = Field(default="", max_length=200)
    message: str = Field(default="", max_length=2000)  # REV-462: 対話の発言（theme 未指定時にテーマを抽出）
    user_id: str = Field(default="default", max_length=100)


# REV-462: 「〇〇の歌を歌って」等の発言からテーマを取り出す。依頼部分以降と助詞を落とす
_SING_REQUEST_TAIL_RE = re.compile(
    r"(?:何か|なにか|一曲|いっきょく|ちょっと)?を?\s*"
    r"(?:(?:歌|うた)って|(?:歌|うた)を(?:聞|聴|き)かせて|一曲(?:お願い|おねがい)).*$",
    re.S,
)
_THEME_SUFFIX_RE = re.compile(
    r"(?:の(?:歌|うた|曲)を?|について|をテーマに(?:して|した(?:歌|うた|曲)を?)?|をテーマで|で|を|の)$"
)
_THEME_STRIP = " \u3000、。，,.!！?？「」『』\n\t"


def _extract_theme(message: str) -> str:
    s = re.sub(r"^\s*(?:紫苑|しおん)(?:さん|ちゃん)?[、,，\s]*", "", message or "")
    s = _SING_REQUEST_TAIL_RE.sub("", s).strip(_THEME_STRIP)
    s = _THEME_SUFFIX_RE.sub("", s).strip(_THEME_STRIP)
    return "" if s in ("何か", "なにか", "一曲") else s[:60]


def _require_sing_enabled() -> None:
    if os.environ.get("SHION_SING_ENABLED", "0") != "1":
        raise HTTPException(status_code=404, detail="sing disabled")


def _voicevox_url() -> str:
    return os.environ.get("SHION_VOICEVOX_URL", "http://127.0.0.1:50021").rstrip("/")


def _sing_ids() -> tuple[int, int]:
    return _env_int("SHION_SING_TEACHER_ID", 6000), _env_int("SHION_SING_VOICE_ID", 3014)


def _sing_key_shift() -> int:
    """REV-473: 歌のキーを何半音上げるか（SHION_SING_KEY_SHIFT。既定 +2、不自然にならないよう -5〜+5）。"""
    return min(5, max(-5, _env_int("SHION_SING_KEY_SHIFT", 2)))


def _transpose(vv_score: dict, shift: int) -> dict:
    """楽譜の音符を shift 半音ずらす（休符はそのまま）。伴奏も同じだけ移調して調を合わせる。"""
    return {"notes": [{**n, "key": n["key"] + shift} if n["key"] is not None else n for n in vv_score["notes"]]}


def _generate_score(theme: str) -> dict:
    from api.loop_engineering_common import call_gemini_json
    from api.prompt_generator import build_shion_system_prompt, load_mind

    now = dt.datetime.now(_JST).strftime("%Y-%m-%d %H:%M")
    prompt = (
        build_shion_system_prompt(load_mind(), now)
        + "\n\n【歌唱モード】\n"
        + (f"テーマ「{theme}」で、" if theme else "テーマは紫苑がいまの気分で自由に決め、")
        + "紫苑らしい短い歌（4/4拍子・8〜16小節）を作曲・作詞し、ピアノ伴奏用のコード進行もつける。\n"
        "次のJSONだけを返す: "
        '{"title": "曲名", "bpm": 60〜180の整数, "key": "調（例: C, Am）", '
        '"notes": [{"lyric": "ひらがな1モーラ", "key": MIDIノート番号, "beats": 拍数}], '
        '"chords": [{"chord": "コード名", "beats": 拍数}]}\n'
        f"- lyric はひらがな1モーラ（「きゃ」等の拗音は1つ）。休符は key を null、lyric を空文字にする\n"
        f"- key は {_KEY_MIN}〜{_KEY_MAX}、beats は 0.25〜4、notes は最大{_MAX_NOTES}個\n"
        "- 歌いやすい旋律にし、跳躍は控えめにする\n"
        "- chords はメロディと同じ調・同じテンポで、先頭の音符から並べる。beats の合計を notes の beats の合計と同じにする\n"
        "- コード名は C, Am, F, G7, Dm7, Cmaj7, Esus4, C/E のような英語表記。基本は1小節（4拍）か2拍ごと\n"
        "- メロディの各音は、その時に鳴っているコードの構成音を中心にする"
    )
    return call_gemini_json(prompt, temperature=0.8, max_output_tokens=4096)


def _score_bpm(score: dict) -> int:
    try:
        return min(180, max(60, int(score.get("bpm") or 96)))
    except (TypeError, ValueError):
        return 96


def _to_voicevox_score(score: dict) -> tuple[dict, str]:
    """LLM の楽譜を検証・補正し、VOICEVOX の楽譜と歌詞文字列を返す。"""
    bpm = _score_bpm(score)
    raw = score.get("notes") if isinstance(score.get("notes"), list) else []
    notes = [{"key": None, "frame_length": 15, "lyric": ""}]  # 先頭は休符必須
    lyrics = []
    for n in raw[:_MAX_NOTES]:
        if not isinstance(n, dict):
            continue
        try:
            beats = min(4.0, max(0.25, float(n.get("beats") or 1)))
        except (TypeError, ValueError):
            beats = 1.0
        frame_length = max(1, round(beats * 60 / bpm * _FRAME_RATE))
        lyric = str(n.get("lyric") or "").strip()
        key = n.get("key")
        if isinstance(key, (int, float)) and not isinstance(key, bool) and _MORA_RE.match(lyric):
            key = int(key)
            while key < _KEY_MIN:
                key += 12
            while key > _KEY_MAX:
                key -= 12
            notes.append({"key": key, "frame_length": frame_length, "lyric": lyric})
            lyrics.append(lyric)
        else:  # 休符・歌えない音は休符にしてリズムを保つ
            notes.append({"key": None, "frame_length": frame_length, "lyric": ""})
    if len(lyrics) < 4:
        raise ValueError("too few singable notes")
    notes.append({"key": None, "frame_length": 30, "lyric": ""})
    return {"notes": notes}, "".join(lyrics)


def _synthesize(vv_score: dict, teacher_id: int, voice_id: int) -> bytes:
    import requests

    base = _voicevox_url()
    q = requests.post(f"{base}/sing_frame_audio_query", params={"speaker": teacher_id}, json=vv_score, timeout=60)
    q.raise_for_status()
    wav = requests.post(f"{base}/frame_synthesis", params={"speaker": voice_id}, json=q.json(), timeout=180)
    wav.raise_for_status()
    return wav.content


def _credit(teacher_id: int, voice_id: int) -> str:
    """VOICEVOX 利用規約のクレジット「VOICEVOX:キャラ名」を /singers から作る。"""
    ids = (teacher_id, voice_id)
    if ids not in _credit_cache:
        import requests

        try:
            singers = requests.get(f"{_voicevox_url()}/singers", timeout=10).json()
            names = {st["id"]: s["name"] for s in singers for st in s["styles"]}
            voice, teacher = names[voice_id], names[teacher_id]
            _credit_cache[ids] = f"VOICEVOX:{voice}" + ("" if voice == teacher else f"（歌唱指導 VOICEVOX:{teacher}）")
        except Exception:
            return "VOICEVOX"
    return _credit_cache[ids]


@router.post("/sing")
def sing(req: SingRequest):
    """テーマから紫苑の短い歌を作り、VOICEVOX で歌った wav を返す（回数・間隔制限つき）。"""
    global _last_sung_at
    _require_sing_enabled()
    daily_limit = _env_int("SHION_SING_DAILY_LIMIT", 10)
    min_interval = _env_int("SHION_SING_MIN_INTERVAL_SECONDS", 30)

    theme = req.theme.strip() or _extract_theme(req.message)

    # 合成は数十秒かかるので、待たせてスレッドプールを塞がず即 429 にする
    if not _sing_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="いま別の歌を準備中です")
    try:
        today = dt.datetime.now(_JST).date().isoformat()
        used = _sung.get(today, 0)
        if used >= daily_limit:
            raise HTTPException(status_code=429, detail="本日の歌唱の上限回数に達しました")
        if time.monotonic() - _last_sung_at < min_interval:
            raise HTTPException(status_code=429, detail="少し時間をおいてから再度お試しください")

        try:
            score = _generate_score(theme)
            vv_score, lyrics = _to_voicevox_score(score if isinstance(score, dict) else {})
        except Exception as exc:
            logger.error("shion sing: score generation failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=502, detail="歌を作れませんでした") from None

        teacher_id, voice_id = _sing_ids()
        key_shift = _sing_key_shift()
        vv_score = _transpose(vv_score, key_shift)
        try:
            wav = _synthesize(vv_score, teacher_id, voice_id)
        except Exception as exc:
            logger.error("shion sing: voicevox failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=503, detail="歌唱エンジンに接続できませんでした") from None
        # REV-469: 同じテンポ・調のコード進行からピアノ伴奏を作って重ねる（オフ・失敗時は歌声のみ）
        from api.shion_sing_accompaniment import add_accompaniment

        mixed = add_accompaniment(wav, vv_score, score.get("chords"), _score_bpm(score), key_shift=key_shift)
        if mixed is not None:
            wav = mixed

        _sung[today] = used + 1
        _last_sung_at = time.monotonic()
    finally:
        _sing_lock.release()

    return {
        "title": str(score.get("title") or theme or "紫苑の歌")[:60],
        "theme": theme,
        "lyrics": lyrics,
        "audio_base64": base64.b64encode(wav).decode("ascii"),
        "mime_type": "audio/wav",
        "credit": _credit(teacher_id, voice_id),
        "accompaniment": mixed is not None,
        "remaining_today": daily_limit - used - 1,
    }


# ── REV-463: 読み上げ（VOICEVOX トーク。歌唱と同じ声に揃える） ──

class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=300)


def _tts_speaker() -> int:
    return _env_int("SHION_TTS_SPEAKER_ID", 14)  # 冥鳴ひまり ノーマル（歌唱 3014 と同じ声）


def _tts_synthesize(text: str, speaker: int) -> bytes:
    import requests

    base = _voicevox_url()
    q = requests.post(f"{base}/audio_query", params={"speaker": speaker, "text": text}, timeout=30)
    q.raise_for_status()
    query = q.json()
    try:
        # REV-474: 少し速めに（SHION_TTS_SPEED。既定 1.15、聞き取れる範囲 0.8〜1.3）
        query["speedScale"] = min(1.3, max(0.8, float(os.environ.get("SHION_TTS_SPEED", "1.15"))))
    except ValueError:
        pass
    # REV-473: 少し高めの声に（SHION_TTS_PITCH。VOICEVOX の pitchScale、既定 +0.04、-0.15〜0.15）
    try:
        query["pitchScale"] = min(0.15, max(-0.15, float(os.environ.get("SHION_TTS_PITCH", "0.04"))))
    except ValueError:
        pass
    wav = requests.post(f"{base}/synthesis", params={"speaker": speaker}, json=query, timeout=60)
    wav.raise_for_status()
    return wav.content


def _tts_credit(speaker: int) -> str:
    """利用規約のクレジット「VOICEVOX:キャラ名」を /speakers から作る。"""
    key = (-1, speaker)
    if key not in _credit_cache:
        import requests

        try:
            speakers = requests.get(f"{_voicevox_url()}/speakers", timeout=10).json()
            names = {st["id"]: s["name"] for s in speakers for st in s["styles"]}
            _credit_cache[key] = f"VOICEVOX:{names[speaker]}"
        except Exception:
            return "VOICEVOX"
    return _credit_cache[key]


@router.post("/tts")
def tts(req: TtsRequest):
    """チャット返答の1文〜数文を VOICEVOX で読み上げた wav を返す（無効・失敗時はブラウザ読み上げへ）。"""
    if os.environ.get("SHION_TTS_ENABLED", "0") != "1":
        raise HTTPException(status_code=404, detail="tts disabled")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="empty text")
    speaker = _tts_speaker()
    try:
        wav = _tts_synthesize(text, speaker)
    except Exception as exc:
        logger.error("shion tts: voicevox failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="読み上げエンジンに接続できませんでした") from None
    # ヘッダは latin-1 のみなので URL エンコードで渡す
    from urllib.parse import quote

    return Response(content=wav, media_type="audio/wav", headers={"X-Voice-Credit": quote(_tts_credit(speaker))})
