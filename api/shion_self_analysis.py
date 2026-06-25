"""
紫苑（SHION）自己分析モジュール

以下のデータを Gemini で分析し、討論ページのペルソナ注入に使う
楽観/懐疑/統合の傾向を返す。24時間キャッシュ（data/shion_self_analysis_cache.json）。

分析材料:
  - data/mind.json の world_view（summary / key_signals）
  - Obsidian vault mind.json の conversation_keypoints（審査チャット重要事実、最大50件）
"""
from __future__ import annotations

import json
import os
import re
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

from api.llm_json_guard import extract_candidate_text, parse_or_recover_json, with_retry_tokens

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_MIND_PATH = os.path.join(_DATA_DIR, "mind.json")
_CACHE_PATH = os.path.join(_DATA_DIR, "shion_self_analysis_cache.json")
_CACHE_TTL_HOURS = 24
_KEYPOINTS_LIMIT = 50


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


def _load_local_mind() -> dict:
    """data/mind.json（プロジェクトローカル）を読む。"""
    with open(_MIND_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_vault_keypoints() -> list[str]:
    """Obsidian vault の mind.json から conversation_keypoints を読む。

    vault が見つからない・読めない場合は空リストを返す。
    """
    try:
        from lease_news_digest import find_vault
        vault = find_vault()
        if not vault:
            return []
        vault_mind = (
            Path(vault)
            / "Projects"
            / "tune_lease_55"
            / "Lease Intelligence"
            / "mind.json"
        )
        if not vault_mind.exists():
            return []
        with vault_mind.open(encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("conversation_keypoints") or []
        contents = []
        for entry in raw:
            if isinstance(entry, dict):
                content = str(
                    entry.get("content")
                    or entry.get("fact")
                    or entry.get("text")
                    or ""
                ).strip()
            else:
                content = str(entry).strip()
            if content:
                contents.append(content)
        return contents[-_KEYPOINTS_LIMIT:]
    except Exception:
        return []


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


def _build_analysis_prompt(local_mind: dict, keypoints: list[str]) -> str:
    wv = local_mind.get("world_view", {})
    summary = wv.get("summary", "")
    key_signals = wv.get("key_signals", [])
    feed_count = wv.get("feed_count", 0)

    signals_text = "\n".join(f"- {s}" for s in key_signals)

    # セントラル共有認識 (REV-155)
    central_section = ""
    try:
        from lease_intelligence_central import get_central_commentary
        from lease_news_digest import find_vault
        _vault = find_vault()
        if _vault:
            _commentary = get_central_commentary(str(_vault))
            _confirmed = _commentary.get("confirmed_beliefs") or []
            _tradeoffs = _commentary.get("known_tradeoffs") or []
            if _confirmed or _tradeoffs:
                _lines = ["\n## セントラルからの共有認識（討論を通じた蓄積）"]
                if _confirmed:
                    _lines.append("確信に昇格した論点:")
                    for _b in _confirmed:
                        _lines.append(f"- {_b}")
                if _tradeoffs:
                    _lines.append("既知のトレードオフ:")
                    for _t in _tradeoffs:
                        _theme = _t.get("theme") if isinstance(_t, dict) else str(_t)
                        _lines.append(f"- {_theme}")
                central_section = "\n".join(_lines) + "\n"
    except Exception:
        pass

    keypoints_section = ""
    if keypoints:
        kp_lines = "\n".join(f"- {kp}" for kp in keypoints)
        keypoints_section = f"""
## 審査チャットから蓄積された重要事実（{len(keypoints)}件）
{kp_lines}
"""

    return f"""以下は審査AIエージェント「紫苑（SHION）」の世界観データと審査経験です。
これを分析して、紫苑が討論エージェントとして持つ傾向を抽出してください。

## 世界観サマリー
{summary}

## 注目シグナル（{feed_count}件のフィードから抽出）
{signals_text}
{central_section}{keypoints_section}
以下のJSON形式のみで回答してください（説明文不要）:
{{
  "optimist_traits": ["楽観的傾向・重視ポイントを3〜5件、具体的な文で"],
  "skeptic_traits": ["懐疑的傾向・チェックポイントを3〜5件、具体的な文で"],
  "arbiter_style": "統合派としての裁定スタイルを1文で",
  "keypoints_used": {len(keypoints)}
}}"""


_SYSTEM_PROMPT = (
    "あなたはリース審査AIエージェント「紫苑」の自己分析を行うアナリストです。"
    "与えられたデータから紫苑の審査判断傾向を客観的に抽出してください。"
    "必ず有効なJSONのみで回答してください。"
)


_ANALYSIS_DEFAULTS = {
    "optimist_traits": [
        "案件の成長余地と営業上の機会を重視する",
        "条件設定により前向きに検討できる余地を探す",
    ],
    "skeptic_traits": [
        "資金繰り、財務耐久力、物件保全を慎重に確認する",
        "説明可能性が不足する案件では追加確認を優先する",
    ],
    "arbiter_style": "双方の論点を整理して説明可能な判断を下す",
    "keypoints_used": 0,
}


def _call_gemini(prompt: str) -> dict:
    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が設定されていません")
    payload = {
        "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.25,
            "maxOutputTokens": 768,
            "responseMimeType": "application/json",
        },
    }
    raw = ""
    finish_reason = ""
    for current_payload in (payload, with_retry_tokens(payload, 1536)):
        resp = requests.post(
            _gemini_url(),
            json=current_payload,
            headers={"x-goog-api-key": api_key},
            timeout=60,
        )
        resp.raise_for_status()
        raw, finish_reason = extract_candidate_text(resp.json())
        result, recovered = parse_or_recover_json(
            raw,
            defaults=_ANALYSIS_DEFAULTS,
            string_fields={"arbiter_style"},
            array_fields={"optimist_traits", "skeptic_traits"},
            number_fields={"keypoints_used"},
        )
        if not recovered and finish_reason != "MAX_TOKENS":
            return result
        if not recovered:
            return result
    result, _ = parse_or_recover_json(
        raw,
        defaults=_ANALYSIS_DEFAULTS,
        string_fields={"arbiter_style"},
        array_fields={"optimist_traits", "skeptic_traits"},
        number_fields={"keypoints_used"},
    )
    result["_finish_reason"] = finish_reason
    return result


def get_shion_self_analysis(force_refresh: bool = False) -> dict:
    """紫苑の自己分析を取得する。キャッシュが有効な場合はキャッシュを返す。"""
    if not force_refresh and _cache_valid():
        return _load_cache()

    local_mind = _load_local_mind()
    keypoints = _load_vault_keypoints()
    prompt = _build_analysis_prompt(local_mind, keypoints)
    result = _call_gemini(prompt)

    now = datetime.now(tz=timezone.utc).isoformat()
    cache = {
        "optimist_traits": result.get("optimist_traits", []),
        "skeptic_traits": result.get("skeptic_traits", []),
        "arbiter_style": result.get("arbiter_style", "双方の論点を整理して説明可能な判断を下す"),
        "generated_at": now,
        "keypoints_used": result.get("keypoints_used", len(keypoints)),
    }
    _save_cache(cache)
    return cache
