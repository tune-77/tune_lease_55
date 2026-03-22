#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slack_remote_control.py
=======================
Slack ボット経由のリモートコントロール用共有ファイル I/O。

フロー: Slack bot → data/remote_control.json ← Streamlit (ポーリング)

Slack bot は write_remote_command() でコマンドを書き込む。
Streamlit は各リラン時に read_remote_command() で未実行コマンドを確認し、
実行後に mark_command_executed() でマークする。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
REMOTE_CONTROL_FILE = _BASE_DIR / "data" / "remote_control.json"

# Streamlit で有効なページ名の短縮エイリアス
PAGE_ALIASES: dict[str, str] = {
    "ホーム": "🏠 ホーム",
    "home": "🏠 ホーム",
    "リースくん": "💬 リースくん",
    "chat": "💬 リースくん",
    "wizard": "💬 リースくん",
    "審査": "📋 審査・分析",
    "shinsa": "📋 審査・分析",
    "レポート": "📄 審査レポート",
    "report": "📄 審査レポート",
    "エージェント": "🤖 汎用エージェントハブ",
    "agent": "🤖 汎用エージェントハブ",
    "hub": "🤖 汎用エージェントハブ",
    "バッチ": "⚡ バッチ審査",
    "batch": "⚡ バッチ審査",
    "物件": "🏭 物件ファイナンス審査",
    "finance": "🏭 物件ファイナンス審査",
    "結果登録": "📝 結果登録 (成約/失注)",
    "result": "📝 結果登録 (成約/失注)",
    "議論": "🤝 エージェントチーム議論",
    "discuss": "🤝 エージェントチーム議論",
    "team": "🤝 エージェントチーム議論",
    "係数": "🔧 係数分析・更新 (β)",
    "coeff": "🔧 係数分析・更新 (β)",
    "ダッシュボード": "📊 履歴分析・実績ダッシュボード",
    "dashboard": "📊 履歴分析・実績ダッシュボード",
    "ルール": "⚙️ 審査ルール設定",
    "rules": "⚙️ 審査ルール設定",
    "settings": "⚙️ 審査ルール設定",
    "ログ": "🪵 アプリログ",
    "applog": "🪵 アプリログ",
}


def resolve_page(name: str) -> str | None:
    """ページ名エイリアスを正式名に変換。見つからなければ None。"""
    # 完全一致（正式名）
    full_names = list(PAGE_ALIASES.values())
    if name in full_names:
        return name
    # エイリアス一致
    return PAGE_ALIASES.get(name.strip())


def write_remote_command(command: str, payload: dict | None = None) -> None:
    """Slack ボットからリモートコマンドを書き込む。"""
    data = {
        "command": command,
        "payload": payload or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "executed": False,
    }
    REMOTE_CONTROL_FILE.parent.mkdir(exist_ok=True)
    REMOTE_CONTROL_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_remote_command() -> dict | None:
    """
    Streamlit がリラン時に呼び出す。
    未実行のコマンドがあれば dict を返す。なければ None。
    """
    if not REMOTE_CONTROL_FILE.exists():
        return None
    try:
        data = json.loads(REMOTE_CONTROL_FILE.read_text(encoding="utf-8"))
        if not data.get("executed"):
            return data
    except Exception:
        pass
    return None


def mark_command_executed() -> None:
    """Streamlit がコマンド実行後に呼び出してマークする。"""
    if not REMOTE_CONTROL_FILE.exists():
        return
    try:
        data = json.loads(REMOTE_CONTROL_FILE.read_text(encoding="utf-8"))
        data["executed"] = True
        REMOTE_CONTROL_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass
