"""world_feed.jsonl の直近N件を読み、Gemini API で紫苑視点の世界読みを生成する。

生成結果は data/mind.json の world_view フィールドに書き込む。
Usage: python update_world_view.py [--limit N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
from pathlib import Path

_FEED_PATH = Path(__file__).parent / "data" / "world_feed.jsonl"
_MIND_PATH = Path(__file__).parent / "data" / "mind.json"
_DEFAULT_LIMIT = 20
_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _gemini_api_key() -> str:
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


def _read_recent_feed(limit: int) -> list[dict]:
    if not _FEED_PATH.exists():
        return []
    lines = _FEED_PATH.read_text(encoding="utf-8").splitlines()
    entries: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries[-limit:]


def _build_prompt(entries: list[dict]) -> str:
    if not entries:
        feed_text = "（取得できた外部情報はありません）"
    else:
        lines = []
        for e in entries:
            src = e.get("source", "不明")
            title = e.get("title", "")
            desc = e.get("description", "")
            pub = (e.get("published") or e.get("fetched_at") or "")[:16]
            line = f"[{src}] {title}"
            if desc:
                line += f" — {desc[:100]}"
            if pub:
                line += f" ({pub})"
            lines.append(line)
        feed_text = "\n".join(lines)

    return f"""あなたは「紫苑（リース知性体）」として以下の外部情報フィードを読んでいます。
リース審査・与信・金融・経済の視点から、この情報をどう解釈するか、
あなた自身の言葉でまとめてください。

【外部情報フィード（直近取得分）】
{feed_text}

以下のJSON形式のみで返答してください（コードブロック不要）：
{{
  "summary": "現在の世界認識サマリー（200字以内、紫苑の解釈込みで）",
  "key_signals": [
    "シグナル1（リース審査に関係する形で1〜2文）",
    "シグナル2",
    "シグナル3"
  ]
}}

注意:
- 情報が少ない・不明な場合は「情報不足のため現時点では判断保留」と正直に記載する
- 推測は「〜とみられる」「〜の可能性がある」と明示する
- リース業（設備投資・残価・中古市況・金利動向・借手与信）に引きつけて読む
- 政治的意見は避け、審査判断に関係する経済事実のみを扱う
"""


def _call_gemini(prompt: str) -> dict:
    """Gemini API を呼んで world_view JSON を取得する。失敗時は空辞書を返す。"""
    try:
        import requests
        api_key = _gemini_api_key()
        if not api_key:
            print("[update_world_view] GEMINI_API_KEY が見つかりません", file=sys.stderr)
            return {}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{_GEMINI_MODEL}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
                "thinkingConfig": {"thinkingBudget": 512},
            },
        }
        resp = requests.post(url, json=payload, headers={"x-goog-api-key": api_key}, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            return json.loads(m.group(0))
    except Exception as exc:
        print(f"[update_world_view] Gemini API 呼び出し失敗: {exc}", file=sys.stderr)
    return {}


def _write_world_view(world_view: dict) -> None:
    """data/mind.json の world_view フィールドを原子的に更新する。"""
    current: dict = {}
    if _MIND_PATH.exists():
        try:
            current = json.loads(_MIND_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            current = {}
    current["world_view"] = world_view

    fd, tmp = tempfile.mkstemp(prefix=".mind-", suffix=".json", dir=_MIND_PATH.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, _MIND_PATH)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main(limit: int = _DEFAULT_LIMIT) -> None:
    entries = _read_recent_feed(limit)
    print(f"[update_world_view] フィード {len(entries)}件を読み込みました。")

    prompt = _build_prompt(entries)
    result = _call_gemini(prompt)

    if not result.get("summary"):
        print("[update_world_view] 世界観の生成に失敗しました（graceful degradation）", file=sys.stderr)
        return

    world_view: dict = {
        "summary": str(result.get("summary", ""))[:300],
        "key_signals": [str(s)[:200] for s in (result.get("key_signals") or [])[:5]],
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "feed_count": len(entries),
    }
    _write_world_view(world_view)
    print(f"[update_world_view] world_view を {_MIND_PATH} に書き込みました。")
    print(f"  summary: {world_view['summary'][:80]}...")
    for sig in world_view["key_signals"]:
        print(f"  signal: {sig[:60]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="紫苑の世界観（world_view）を更新する")
    parser.add_argument("--limit", type=int, default=_DEFAULT_LIMIT,
                        help=f"フィードの読み込み件数（デフォルト: {_DEFAULT_LIMIT}）")
    args = parser.parse_args()
    main(args.limit)
