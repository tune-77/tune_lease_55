"""
紫苑（SHION）自己分析モジュール

data/mind.json の world_view を Gemini で分析し、
討論ページのペルソナ注入に使う楽観/懐疑/統合の傾向を返す。
24時間キャッシュ（data/shion_self_analysis_cache.json）。
"""
from __future__ import annotations

import json
import os
import re
import requests
from datetime import datetime, timezone, timedelta

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_MIND_PATH = os.path.join(_DATA_DIR, "mind.json")
_CACHE_PATH = os.path.join(_DATA_DIR, "shion_self_analysis_cache.json")
_CACHE_TTL_HOURS = 24


def _gemini_url() -> str:
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _get_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    _here = os.path.dirname(os.path.abspath(__file__))
    cur = os.path.dirname(_here)
    for _ in range(5):
        sec_path = os.path.join(cur, ".streamlit", "secrets.toml")
        if os.path.exists(sec_path):
            try:
                with open(sec_path, encoding="utf-8") as f:
                    for line in f:
                        m = re.match(r'^GEMINI_API_KEY\s*=\s*["\'](.+)["\']', line.strip())
                        if m:
                            return m.group(1)
            except Exception:
                pass
        cur = os.path.dirname(cur)
    return ""


def _load_mind() -> dict:
    with open(_MIND_PATH, encoding="utf-8") as f:
        return json.load(f)


def _cache_valid() -> bool:
    if not os.path.exists(_CACHE_PATH):
        return False
    try:
        with open(_CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        generated_at = datetime.fromisoformat(cache.get("generated_at", "2000-01-01"))
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        return datetime.now(tz=timezone.utc) - generated_at < timedelta(hours=_CACHE_TTL_HOURS)
    except Exception:
        return False


def _load_cache() -> dict:
    with open(_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_cache(data: dict) -> None:
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_analysis_prompt(mind: dict) -> str:
    wv = mind.get("world_view", {})
    summary = wv.get("summary", "")
    key_signals = wv.get("key_signals", [])
    feed_count = wv.get("feed_count", 0)

    signals_text = "\n".join(f"- {s}" for s in key_signals)
    return f"""以下は審査AIエージェント「紫苑（SHION）」の世界観データです。
これを分析して、紫苑が討論エージェントとして持つ傾向を抽出してください。

## 世界観サマリー
{summary}

## 注目シグナル（{feed_count}件のフィードから抽出）
{signals_text}

以下のJSON形式のみで回答してください（説明文不要）:
{{
  "optimist_traits": ["楽観的傾向・重視ポイントを3〜5件、具体的な文で"],
  "skeptic_traits": ["懐疑的傾向・チェックポイントを3〜5件、具体的な文で"],
  "arbiter_style": "統合派としての裁定スタイルを1文で",
  "keypoints_used": {feed_count}
}}"""


_SYSTEM_PROMPT = (
    "あなたはリース審査AIエージェント「紫苑」の自己分析を行うアナリストです。"
    "与えられたデータから紫苑の審査判断傾向を客観的に抽出してください。"
    "必ず有効なJSONのみで回答してください。"
)


def _call_gemini(prompt: str) -> dict:
    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が設定されていません")
    payload = {
        "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(
        _gemini_url(),
        json=payload,
        headers={"x-goog-api-key": api_key},
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts[1::2]:
            cleaned = part.lstrip("json\n").strip()
            if cleaned:
                raw = cleaned
                break
    return json.loads(raw)


def get_shion_self_analysis(force_refresh: bool = False) -> dict:
    """紫苑の自己分析を取得する。キャッシュが有効な場合はキャッシュを返す。"""
    if not force_refresh and _cache_valid():
        return _load_cache()

    mind = _load_mind()
    prompt = _build_analysis_prompt(mind)
    result = _call_gemini(prompt)

    now = datetime.now(tz=timezone.utc).isoformat()
    cache = {
        "optimist_traits": result.get("optimist_traits", []),
        "skeptic_traits": result.get("skeptic_traits", []),
        "arbiter_style": result.get("arbiter_style", "双方の論点を整理して説明可能な判断を下す"),
        "generated_at": now,
        "keypoints_used": result.get("keypoints_used", 0),
    }
    _save_cache(cache)
    return cache
