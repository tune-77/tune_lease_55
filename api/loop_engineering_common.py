"""
Loop Engineering 系機能（Usage / Judgment Divergence / Feedback Pattern /
Outcome Drift / Knowledge Gap）で共通して使う薄いユーティリティ。

各ループは「Observe（既存ログを読む）→ Aggregate（集計する）→
Propose（Geminiに提案させる）→ Persist（JSONLへ追記する）」という
同じ形をしている。ここでは Gemini 呼び出しとJSONL入出力だけを共通化し、
集計ロジック・プロンプト内容は各ループ側に残す（提案の質はドメイン
知識に依存するため、無理に共通化しない）。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data")))


def gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    here = Path(__file__).parent
    for _ in range(5):
        sec = here / ".streamlit" / "secrets.toml"
        if sec.exists():
            for line in sec.read_text(encoding="utf-8").splitlines():
                m = re.match(r'^GEMINI_API_KEY\s*=\s*["\'](.+)["\']', line.strip())
                if m:
                    return m.group(1)
        here = here.parent
    return ""


def call_gemini_json(prompt: str, *, temperature: float = 0.4, max_output_tokens: int = 8192) -> Any:
    """Gemini に JSON 出力を要求し、パース済みの値（list/dict）を返す。失敗時は例外を投げる。"""
    import requests

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    api_key = gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が見つかりません")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(url, json=payload, headers={"x-goog-api-key": api_key}, timeout=60)
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    return json.loads(text)


def append_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_jsonl(path: Path, limit: int | None = None, newest_first: bool = True) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if newest_first:
        entries.reverse()
    return entries[:limit] if limit else entries
