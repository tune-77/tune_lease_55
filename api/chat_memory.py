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

_DATA_DIR = "/Users/kobayashiisaoryou/clawd/tune_lease_55/data"
DB_PATH = os.path.join(_DATA_DIR, "lease_data.db")

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


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
        resp = requests.post(f"{_GEMINI_URL}?key={api_key}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return ""


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
            "temperature": 0.7,
            "maxOutputTokens": 1024,
        },
    }
    resp = requests.post(
        f"{_GEMINI_URL}?key={api_key}",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
