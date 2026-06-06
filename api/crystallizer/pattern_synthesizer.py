"""
抽出した外れ値案件群を Gemini に渡して「暗黙知パターン」を言語化する。
"""
from __future__ import annotations

import json
import os
import re
import requests

from api.crystallizer.anomaly_extractor import AnomalyCase

def _gemini_url() -> str:
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_SYSTEM = """あなたはリース審査のナレッジマネジメント専門家です。
与えられた審査案件群から、今後の審査に役立つ新しいパターンや教訓を簡潔に言語化してください。
200字以内の日本語で、具体的な業種・スコア傾向・リスクの観点から記述してください。"""


def _get_api_key() -> str:
    # main.py の _load_secrets_to_env() が起動時に環境変数へ注入済みのため、
    # 環境変数のみを参照する。
    return os.environ.get("GEMINI_API_KEY", "").strip()


def synthesize_pattern(cases: list[AnomalyCase]) -> str:
    """
    案件リストから学べるパターンを Gemini で言語化して返す。
    API 呼び出し失敗時はルールベースのフォールバック文を返す。
    """
    if not cases:
        return "対象案件が0件のため、パターン合成をスキップしました。"

    api_key = _get_api_key()
    if not api_key:
        return _fallback_pattern(cases)

    summaries = "\n".join(f"- {c.to_summary()}" for c in cases[:10])
    prompt = f"""以下の審査案件群を分析し、今後の審査に役立つ新しいパターンを200字以内で言語化してください。

## 対象案件
{summaries}

## 出力形式
パターン名: （10字以内）
内容: （200字以内）"""

    payload = {
        "system_instruction": {"parts": [{"text": _SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 512},
    }
    try:
        resp = requests.post(
            _gemini_url(),
            json=payload,
            headers={"x-goog-api-key": api_key},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return _fallback_pattern(cases, error=str(e))


def _fallback_pattern(cases: list[AnomalyCase], error: str = "") -> str:
    """Gemini API が使えない場合のルールベースパターン。"""
    industries = list({c.industry for c in cases if c.industry != "不明"})
    avg_score = sum(c.score for c in cases) / len(cases) if cases else 0
    note = f"（API呼び出し失敗: {error}）" if error else ""
    return (
        f"パターン名: 外れ値案件群{note}\n"
        f"内容: 対象業種 {', '.join(industries[:3]) or '不明'} の{len(cases)}件について"
        f"平均スコア{avg_score:.1f}点の案件群。意見割れまたはスコア乖離が発生した案件の傾向を精査すること。"
    )
