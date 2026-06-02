"""Gemini APIを使って改善案を整理・統合するモジュール."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
_GEMINI_REST_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{_GEMINI_MODEL}:generateContent"
)

_CONSOLIDATION_PROMPT = """\
以下はリースシステムの改善案リストです。類似・重複するものをグループ化して、重要度順に10〜15件にまとめてください。

各項目について以下の形式でJSONを返してください：
{{
  "items": [
    {{
      "title": "整理後のタイトル（簡潔に）",
      "description": "何をするか1〜2文で説明",
      "priority": "high|medium|low",
      "category": "small_ui|rag_chat|data|large",
      "original_count": N,
      "original_titles": ["元のタイトル1", ...]
    }}
  ]
}}

改善案リスト:
{improvements_text}
"""


def _get_api_key() -> str:
    return (
        os.environ.get("GOOGLE_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )


def _build_improvements_text(improvements: list[dict]) -> str:
    lines = []
    for i, imp in enumerate(improvements, 1):
        lines.append(f"{i}. {imp['title']}")
        if imp.get("reason"):
            lines.append(f"   理由: {imp['reason']}")
    return "\n".join(lines)


def _call_gemini_sdk(prompt: str, api_key: str) -> str | None:
    """google.generativeai SDK経由でGemini APIを呼び出す."""
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=_GEMINI_MODEL,
            generation_config={"max_output_tokens": 4096, "temperature": 0.3},
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"情報: google.generativeai SDK失敗 ({type(e).__name__}): {e}", file=sys.stderr)
    return None


def _call_gemini_rest(prompt: str, api_key: str) -> str | None:
    """HTTP REST経由でGemini APIを呼び出す（SDKのフォールバック）."""
    try:
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.3},
        }).encode("utf-8")
        url = f"{_GEMINI_REST_URL}?key={api_key}"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"情報: Gemini REST API失敗 ({type(e).__name__}): {e}", file=sys.stderr)
    return None


def _extract_json_items(text: str) -> list[dict] | None:
    """レスポンステキストからitemsリストをパースする."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        data = json.loads(text)
        items = data.get("items", [])
        if isinstance(items, list) and items:
            return items
    except json.JSONDecodeError:
        pass
    # JSON部分だけ抽出して再試行
    m = re.search(r'\{[^{}]*"items"\s*:.*\}', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            items = data.get("items", [])
            if isinstance(items, list) and items:
                return items
        except Exception:
            pass
    return None


def consolidate_with_ai(improvements: list[dict]) -> list[dict]:
    """
    Gemini APIを使って改善案を整理・統合する。
    - 類似テーマをグループ化して代表タイトルに統合
    - 優先度（high/medium/low）を付与
    - 10〜15件に圧縮
    - 各グループに統合された元件数を記録

    フォールバック: APIキーなし・エラー時は入力リストをそのまま返す。
    """
    if not improvements:
        return improvements

    api_key = _get_api_key()
    if not api_key:
        print(
            "情報: GOOGLE_API_KEY/GEMINI_API_KEY 未設定のため AI統合をスキップします",
            file=sys.stderr,
        )
        return improvements

    improvements_text = _build_improvements_text(improvements)
    prompt = _CONSOLIDATION_PROMPT.format(improvements_text=improvements_text)

    print(
        f"Gemini APIで改善案を統合中... ({len(improvements)}件 → 10〜15件)",
        file=sys.stderr,
    )

    raw_text = _call_gemini_sdk(prompt, api_key) or _call_gemini_rest(prompt, api_key)
    if not raw_text:
        print("警告: Gemini API応答なし。重複排除済みリストを使用します。", file=sys.stderr)
        return improvements

    items = _extract_json_items(raw_text)
    if not items:
        print("警告: JSON解析失敗。重複排除済みリストを使用します。", file=sys.stderr)
        return improvements

    consolidated: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        original_count = int(item.get("original_count", 1))
        consolidated.append({
            "tag": "改善",
            "title": title,
            "reason": str(item.get("description", "")),
            "priority": str(item.get("priority", "medium")),
            "category": str(item.get("category", "")),
            "original_count": original_count,
            "original_titles": list(item.get("original_titles", [])),
            "duplicate_count": original_count,
        })

    if not consolidated:
        print("警告: 統合結果が空。重複排除済みリストを使用します。", file=sys.stderr)
        return improvements

    print(f"AI統合完了: {len(consolidated)}件にまとめました", file=sys.stderr)
    return consolidated
