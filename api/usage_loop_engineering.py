"""
画面利用ループエンジニアリング（Usage Loop Engineering）。

Observe   : フロントエンドから届く画面訪問イベントを記録する
Aggregate : 直近N日の訪問回数・最終訪問日を画面ごとに集計する
Propose   : 集計結果をGeminiに渡し、UI/UX改善案を生成する
Persist   : 生成した改善案を追記保存し、次回以降の一覧取得に使う
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any

_DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent / "data")))
_EVENTS_PATH = _DATA_DIR / "usage_loop_events.jsonl"
_PROPOSALS_PATH = _DATA_DIR / "usage_loop_proposals.jsonl"

_LOOKBACK_DAYS = 30


def record_visit(path: str, user_id: str = "default") -> None:
    """画面訪問イベントを1件追記する。失敗しても呼び出し元を止めない。"""
    if not path:
        return
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "path": path,
        "user_id": user_id or "default",
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
    }
    with _EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_events(days: int) -> list[dict[str, Any]]:
    if not _EVENTS_PATH.exists():
        return []
    cutoff = dt.datetime.now() - dt.timedelta(days=days)
    events: list[dict[str, Any]] = []
    for line in _EVENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            recorded = dt.datetime.fromisoformat(str(entry.get("ts") or ""))
        except ValueError:
            continue
        if recorded >= cutoff:
            events.append(entry)
    return events


def aggregate_usage(days: int = _LOOKBACK_DAYS) -> dict[str, Any]:
    """直近days日分のイベントを画面ごとに集計する。"""
    events = _load_events(days)
    stats: dict[str, dict[str, Any]] = {}
    for entry in events:
        path = str(entry.get("path") or "")
        if not path:
            continue
        bucket = stats.setdefault(path, {"path": path, "visit_count": 0, "last_visited": ""})
        bucket["visit_count"] += 1
        ts = str(entry.get("ts") or "")
        if ts > bucket["last_visited"]:
            bucket["last_visited"] = ts

    ranked_desc = sorted(stats.values(), key=lambda s: s["visit_count"], reverse=True)
    ranked_asc = sorted(stats.values(), key=lambda s: s["visit_count"])
    return {
        "window_days": days,
        "total_events": len(events),
        "pages": ranked_desc,
        "most_used": ranked_desc[:5],
        "least_used": ranked_asc[:5] if ranked_desc else [],
    }


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


def _call_gemini(prompt: str) -> str:
    import requests

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    api_key = _gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が見つかりません")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 2000,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(url, json=payload, headers={"x-goog-api-key": api_key}, timeout=60)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _build_prompt(usage: dict[str, Any]) -> str:
    most_used_lines = "\n".join(
        f"- {p['path']}: {p['visit_count']}回, 最終訪問 {p['last_visited']}"
        for p in usage["most_used"]
    ) or "（データなし）"
    least_used_lines = "\n".join(
        f"- {p['path']}: {p['visit_count']}回, 最終訪問 {p['last_visited']}"
        for p in usage["least_used"]
    ) or "（データなし）"
    return f"""あなたはリース審査AIシステム「紫苑」です。ユーザーの画面利用状況を観察し、
UI/UXや機能の改善案を考えるのがあなたの役目の一つです。

【直近{usage['window_days']}日間の画面利用状況】
総アクセス数: {usage['total_events']}

よく使われている画面:
{most_used_lines}

あまり使われていない画面:
{least_used_lines}

この利用状況を踏まえて、実務上価値のある改善案を3〜5件、以下のJSON配列形式のみで返してください
（前後の説明テキストは不要）:

[
  {{
    "title": "改善案のタイトル（30字以内）",
    "target_page": "対象画面のパス（例: /screening）",
    "reason": "なぜこの改善が必要か。利用状況のどの部分から着想したかを含めて説明する（100字程度）",
    "priority": "high|medium|low"
  }}
]

よく使われる画面はさらに使いやすくする改善案を、あまり使われない画面は理由を推測して
統合・削除・導線改善などを提案してください。一般論ではなく、与えられた利用状況の数字を
根拠にした具体的な提案にしてください。"""


def generate_proposals(days: int = _LOOKBACK_DAYS) -> dict[str, Any]:
    """利用状況を集計し、Geminiで改善案を生成して保存する。"""
    usage = aggregate_usage(days)
    if usage["total_events"] == 0:
        return {"generated": False, "reason": "利用データがまだありません", "proposals": []}

    prompt = _build_prompt(usage)
    try:
        raw = _call_gemini(prompt)
        proposals = json.loads(raw)
        if not isinstance(proposals, list):
            raise ValueError("Gemini応答がリストではありません")
    except Exception as exc:
        return {"generated": False, "reason": f"Gemini生成に失敗: {exc}", "proposals": []}

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = dt.datetime.now().isoformat(timespec="seconds")
    saved: list[dict[str, Any]] = []
    with _PROPOSALS_PATH.open("a", encoding="utf-8") as f:
        for item in proposals:
            if not isinstance(item, dict) or not str(item.get("title") or "").strip():
                continue
            entry = {
                "title": str(item.get("title") or "").strip(),
                "target_page": str(item.get("target_page") or "").strip(),
                "reason": str(item.get("reason") or "").strip(),
                "priority": str(item.get("priority") or "medium").strip(),
                "generated_at": generated_at,
                "status": "proposed",
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            saved.append(entry)

    return {"generated": True, "usage_summary": usage, "proposals": saved}


def load_proposals(limit: int = 20) -> list[dict[str, Any]]:
    """保存済みの改善案を新しい順に返す。"""
    if not _PROPOSALS_PATH.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in _PROPOSALS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    entries.reverse()
    return entries[:limit]
