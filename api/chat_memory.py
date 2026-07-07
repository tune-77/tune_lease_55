"""
汎用チャットの永続記憶モジュール。
chat_messages テーブルを lease_data.db 内に作成し、
ユーザーとアシスタントのメッセージを保存・取得する。
"""
from __future__ import annotations

import os
from typing import Optional
import requests
import re

from api.db_connection import current_backend, get_connection, placeholder, ensure_schema

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _gemini_model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


def _gemini_url() -> str:
    return f"{_GEMINI_API_BASE}/{_gemini_model()}:generateContent"


def init_chat_messages_table() -> None:
    ensure_schema()


def get_recent_messages(user_id: str = "default", limit: int = 20, since: str | None = None) -> list[dict]:
    """直近 limit 件のメッセージを古い順で返す。"""
    try:
        init_chat_messages_table()
        ph = placeholder()
        with get_connection() as conn:
            cur = conn.cursor()
            if since:
                if current_backend() == "postgresql":
                    cur.execute(
                        f"""
                        SELECT id, user_id, role, content, created_at
                        FROM chat_messages
                        WHERE user_id = {ph}
                          AND created_at >= {ph}
                        ORDER BY created_at DESC
                        LIMIT {ph}
                        """,
                        (user_id, since, limit),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT id, user_id, role, content, created_at
                        FROM chat_messages
                        WHERE user_id = {ph}
                          AND datetime(created_at) >= datetime({ph})
                        ORDER BY created_at DESC
                        LIMIT {ph}
                        """,
                        (user_id, since, limit),
                    )
            else:
                cur.execute(
                    f"""
                    SELECT id, user_id, role, content, created_at
                    FROM chat_messages
                    WHERE user_id = {ph}
                    ORDER BY created_at DESC
                    LIMIT {ph}
                    """,
                    (user_id, limit),
                )
            rows = cur.fetchall()
        return [dict(r) for r in reversed(rows)]
    except Exception as exc:
        print(f"[chat_memory] get_recent_messages skipped: {exc}")
        return []


def save_message(user_id: str, role: str, content: str) -> None:
    try:
        init_chat_messages_table()
        ph = placeholder()
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"INSERT INTO chat_messages (user_id, role, content) VALUES ({ph}, {ph}, {ph})",
                (user_id, role, content),
            )
    except Exception as exc:
        print(f"[chat_memory] save_message skipped: {exc}")


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
    text = re.sub(r"(?m)^\s*[\*•]\s+", "- ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chat_max_tokens() -> int:
    raw = os.environ.get("MEBUKI_CHAT_MAX_TOKENS", "2400")
    try:
        value = int(raw)
    except ValueError:
        value = 2400
    return max(800, min(8192, value))


def _chat_continuation_rounds() -> int:
    raw = os.environ.get("MEBUKI_CHAT_CONTINUATION_ROUNDS", "3")
    try:
        value = int(raw)
    except ValueError:
        value = 3
    return max(1, min(5, value))


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


def _continue_truncated_response(
    *,
    api_key: str,
    system_prompt: str,
    contents: list[dict],
    initial_data: dict,
    initial_text: str,
    timeout: int,
) -> str:
    """Continue MAX_TOKENS responses until Gemini reports a completed turn."""
    text = str(initial_text or "").strip()
    data = initial_data
    continuation_contents = list(contents)

    for _ in range(_chat_continuation_rounds()):
        if _candidate_finish_reason(data).upper() != "MAX_TOKENS" or not text:
            break
        previous_chunk = _candidate_text(data).strip()
        if not previous_chunk:
            break
        continuation_contents.extend(
            [
                {"role": "model", "parts": [{"text": previous_chunk}]},
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "直前の回答は出力上限で途切れました。"
                                "既出部分を繰り返さず、切れた箇所の直後から回答を続け、"
                                "文・箇条書き・結論を必ず完結させてください。"
                            )
                        }
                    ],
                },
            ]
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": continuation_contents,
            "generationConfig": {
                "temperature": 0.25,
                "maxOutputTokens": _chat_max_tokens(),
            },
        }
        response = requests.post(
            _gemini_url(),
            json=payload,
            headers={"x-goog-api-key": api_key},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        chunk = _candidate_text(data).strip()
        if not chunk:
            break
        text = f"{text.rstrip()}\n\n{chunk}"

    if _candidate_finish_reason(data).upper() == "MAX_TOKENS":
        text = (
            f"{text.rstrip()}\n\n"
            "（回答が非常に長いため、続きは要点を分けて質問してください。）"
        )
    return text


def get_message_count(user_id: str = "default") -> int:
    init_chat_messages_table()
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT COUNT(*) as cnt FROM chat_messages WHERE user_id = {ph}", (user_id,)
        )
        row = cur.fetchone()
    return row["cnt"] if row else 0


def delete_history(user_id: str = "default") -> int:
    init_chat_messages_table()
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"DELETE FROM chat_messages WHERE user_id = {ph}", (user_id,)
        )
        return cur.rowcount


def _get_gemini_api_key() -> str:
    # 共通実装へ委譲（api/secret_access.py、4重複の集約）
    from api.secret_access import get_gemini_api_key

    return get_gemini_api_key()


def get_summary(user_id: str = "default") -> str:
    """直近50件をGeminiで要約して長期記憶の圧縮テキストを返す。"""
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT role, content FROM chat_messages
            WHERE user_id = {ph}
            ORDER BY created_at DESC LIMIT 50
            """,
            (user_id,),
        )
        rows = cur.fetchall()
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
    extra_user_parts: "list[dict] | None" = None,
) -> str:
    """
    Gemini API を function calling 対応の multi-turn 形式で呼び出す。
    tool_declarations: Gemini function_declarations スキーマのリスト
    tool_executor: (tool_name, args_dict) -> result_any を実行する関数
    extra_user_parts: 画像等を渡す際に text 部分の前に追加する parts (inline_data 等)
    """
    from typing import Callable, Any as _Any  # noqa: F401

    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が設定されていません")

    contents: list[dict] = []
    for msg in history:
        gemini_role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": gemini_role, "parts": [{"text": msg["content"]}]})
    user_parts: list[dict] = (list(extra_user_parts) if extra_user_parts else []) + [{"text": user_message}]
    contents.append({"role": "user", "parts": user_parts})

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
            if text_parts:
                text = _continue_truncated_response(
                    api_key=api_key,
                    system_prompt=system_prompt,
                    contents=contents,
                    initial_data=data,
                    initial_text=text,
                    timeout=90,
                )
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
        try:
            text = _continue_truncated_response(
                api_key=api_key,
                system_prompt=system_prompt,
                contents=contents,
                initial_data=data,
                initial_text=text,
                timeout=60,
            )
        except Exception:
            text = (
                f"{text.rstrip()}\n\n"
                "（回答が長く、続きの取得に失敗しました。必要なら「続き」と送ってください。）"
            )

    return normalize_chat_text(text)
