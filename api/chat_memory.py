"""
汎用チャットの永続記憶モジュール。
chat_messages テーブルを lease_data.db 内に作成し、
ユーザーとアシスタントのメッセージを保存・取得する。
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from typing import Optional
import requests
import re
from runtime_paths import get_data_path

DB_PATH = get_data_path("lease_data.db")

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _gemini_model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


def _gemini_url() -> str:
    return f"{_GEMINI_API_BASE}/{_gemini_model()}:generateContent"


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_chat_messages_table() -> None:
    with closing(_open_db()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_messages(user_id, created_at)"
        )
        conn.commit()


def get_recent_messages(user_id: str = "default", limit: int = 20) -> list[dict]:
    """直近 limit 件のメッセージを古い順で返す。"""
    init_chat_messages_table()
    with closing(_open_db()) as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, role, content, created_at
            FROM chat_messages
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def save_message(user_id: str, role: str, content: str) -> None:
    init_chat_messages_table()
    with closing(_open_db()) as conn:
        conn.execute(
            "INSERT INTO chat_messages (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        conn.commit()


def normalize_chat_text(content: str) -> str:
    """Normalize AI chat text for display/storage without changing substance."""
    text = str(content or "").strip()
    if not text:
        return ""
    text = (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "  ")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    text = re.sub(r"(?m)^\s*[\*\u2022]\s+", "- ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chat_max_tokens() -> int:
    raw = os.environ.get("MEBUKI_CHAT_MAX_TOKENS", "1200")
    try:
        value = int(raw)
    except ValueError:
        value = 1200
    return max(400, min(2400, value))


def _candidate_finish_reason(data: dict) -> str:
    try:
        candidate = (data.get("candidates") or [{}])[0]
        return str(candidate.get("finishReason") or candidate.get("finish_reason") or "")
    except Exception:
        return ""


def _candidate_text(data: dict) -> str:
    try:
        return str(data["candidates"][0]["content"]["parts"][0]["text"])
    except Exception:
        return ""


def get_message_count(user_id: str = "default") -> int:
    init_chat_messages_table()
    with closing(_open_db()) as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM chat_messages WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row["cnt"] if row else 0


def delete_history(user_id: str = "default") -> int:
    init_chat_messages_table()
    with closing(_open_db()) as conn:
        cur = conn.execute(
            "DELETE FROM chat_messages WHERE user_id = ?", (user_id,)
        )
        conn.commit()
        return cur.rowcount


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
                with open(sec_path, "r", encoding="utf-8") as f:
                    for line in f:
                        m = re.match(r'^GEMINI_API_KEY\s*=\s*["\'](.+)["\']', line.strip())
                        if m:
                            return m.group(1)
            except Exception:
                pass
        cur = os.path.dirname(cur)
    return ""


def get_summary(user_id: str = "default") -> str:
    """直近50件をGeminiで要約して長期記憶の圧縮テキストを返す。"""
    with closing(_open_db()) as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM chat_messages
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT 50
            """,
            (user_id,),
        ).fetchall()
    if not rows:
        return ""
    lines = "\n".join(
        f"{'ユーザー' if r['role'] == 'user' else 'AI'}: {r['content']}"
        for r in reversed(rows)
    )
    api_key = _get_gemini_api_key()
    if not api_key:
        return ""
    try:
        payload = {
            "system_instruction": {"parts": [{"text": "以下の会話を100文字以内で要約してください。"}]},
            "contents": [{"role": "user", "parts": [{"text": lines}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 256},
        }
        resp = requests.post(
            _gemini_url(),
            json=payload,
            headers={"x-goog-api-key": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return ""


def call_gemini_with_tools(
    system_prompt: str,
    history: list[dict],
    user_message: str,
    tool_declarations: list[dict],
    tool_executor: "Callable[[str, dict], Any]",
    max_tool_rounds: int = 3,
) -> str:
    """
    Gemini API を function calling 対応の multi-turn 形式で呼び出す。
    tool_declarations: Gemini function_declarations スキーマのリスト
    tool_executor: (tool_name, args_dict) -> result_any を実行する関数
    """
    from typing import Callable, Any as _Any  # noqa: F401

    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が設定されていません")

    contents: list[dict] = []
    for msg in history:
        gemini_role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": gemini_role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    base_payload: dict = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"function_declarations": tool_declarations}],
        "tool_config": {"function_calling_config": {"mode": "AUTO"}},
        "generationConfig": {
            "temperature": 0.45,
            "maxOutputTokens": _chat_max_tokens(),
        },
    }

    text = ""
    for _round in range(max_tool_rounds + 1):
        payload = {**base_payload, "contents": contents}
        resp = requests.post(
            _gemini_url(),
            json=payload,
            headers={"x-goog-api-key": api_key},
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()

        candidate = (data.get("candidates") or [{}])[0]
        parts = (candidate.get("content") or {}).get("parts") or []

        # Collect any function calls in this response
        func_calls = [p["functionCall"] for p in parts if "functionCall" in p]
        text_parts = [p.get("text", "") for p in parts if "text" in p]
        if text_parts:
            text = normalize_chat_text("".join(text_parts))

        if not func_calls or _round == max_tool_rounds:
            break

        # Append model turn (function calls)
        contents.append({"role": "model", "parts": parts})

        # Execute each tool and build function response parts
        response_parts = []
        for fc in func_calls:
            result = tool_executor(fc["name"], fc.get("args") or {})
            response_parts.append({
                "functionResponse": {
                    "name": fc["name"],
                    "response": {"result": result},
                }
            })
        contents.append({"role": "user", "parts": response_parts})

    return text


def call_gemini_chat(
    system_prompt: str,
    history: list[dict],
    user_message: str,
) -> str:
    """
    Gemini API を multi-turn 形式で呼び出す。
    history: [{"role": "user"|"assistant", "content": str}, ...]
    """
    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が設定されていません")

    contents = []
    for msg in history:
        gemini_role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": gemini_role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.45,
            "maxOutputTokens": _chat_max_tokens(),
        },
    }
    resp = requests.post(
        _gemini_url(),
        json=payload,
        headers={"x-goog-api-key": api_key},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text = _candidate_text(data)

    if _candidate_finish_reason(data).upper() == "MAX_TOKENS" and text.strip():
        continuation_payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents + [
                {"role": "model", "parts": [{"text": text}]},
                {
                    "role": "user",
                    "parts": [{
                        "text": (
                            "前の回答が途中で切れています。"
                            "重複せず、直前の続きから必要な結論まで簡潔に続けてください。"
                        )
                    }],
                },
            ],
            "generationConfig": {
                "temperature": 0.35,
                "maxOutputTokens": _chat_max_tokens(),
            },
        }
        try:
            cont = requests.post(
                _gemini_url(),
                json=continuation_payload,
                headers={"x-goog-api-key": api_key},
                timeout=60,
            )
            cont.raise_for_status()
            continuation = _candidate_text(cont.json()).strip()
            if continuation:
                text = f"{text.rstrip()}\n\n{continuation}"
        except Exception:
            text = f"{text.rstrip()}\n\n（回答が長く、続きの取得に失敗しました。必要なら「続き」と送ってください。）"

    return normalize_chat_text(text)
