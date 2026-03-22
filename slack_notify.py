#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slack_notify.py
===============
Streamlit → Slack 通知キュー。

Streamlit (score_calculation.py) が push_notification() で書き込み、
Slack bot (slack_bot.py) が pop_notifications() で読み出して送信する。

データファイル:
  data/slack_notifications.json  … 未送信通知リスト
  data/slack_last_channel.json   … 最後にDMを送ってきたチャンネルID
"""
from __future__ import annotations

import json
import time
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
_NOTIFY_FILE = _BASE_DIR / "data" / "slack_notifications.json"
_CHANNEL_FILE = _BASE_DIR / "data" / "slack_last_channel.json"


# ── 通知キュー ───────────────────────────────────────────────────────────────

def push_notification(text: str) -> None:
    """通知テキストをキューに追加（Streamlit から呼ぶ）。"""
    _NOTIFY_FILE.parent.mkdir(exist_ok=True)
    queue: list = []
    if _NOTIFY_FILE.exists():
        try:
            queue = json.loads(_NOTIFY_FILE.read_text(encoding="utf-8"))
        except Exception:
            queue = []
    queue.append({"text": text, "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
    _NOTIFY_FILE.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")


def pop_notifications() -> list[str]:
    """未送信通知を全件取り出してキューを空にする（Slack bot から呼ぶ）。"""
    if not _NOTIFY_FILE.exists():
        return []
    try:
        queue = json.loads(_NOTIFY_FILE.read_text(encoding="utf-8"))
        if not queue:
            return []
        texts = [item["text"] for item in queue if isinstance(item, dict)]
        _NOTIFY_FILE.write_text("[]", encoding="utf-8")
        return texts
    except Exception:
        return []


# ── 最終チャンネル記録 ───────────────────────────────────────────────────────

def save_last_channel(channel_id: str) -> None:
    """最後にDMを送ってきたチャンネルIDを保存（Slack bot から呼ぶ）。"""
    _CHANNEL_FILE.parent.mkdir(exist_ok=True)
    _CHANNEL_FILE.write_text(
        json.dumps({"channel_id": channel_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S")},
                   ensure_ascii=False),
        encoding="utf-8",
    )


def load_last_channel() -> str | None:
    """最後のチャンネルIDを返す（Slack bot から呼ぶ）。"""
    if not _CHANNEL_FILE.exists():
        return None
    try:
        return json.loads(_CHANNEL_FILE.read_text(encoding="utf-8")).get("channel_id")
    except Exception:
        return None
