#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slack_bot.py
============
Slack ポーリング型ボット — リース審査システムのAIアシスタント。

DM チャンネルを定期的にポーリングして新着メッセージに返答する。
Event Subscriptions 不要。既存の Bot Token Scopes だけで動作。

必要スコープ: chat:write, im:read, im:history
（channels:history, app_mentions:read もあれば公開チャンネル対応可）

起動方法:
    python slack_bot.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import datetime
from pathlib import Path

# ── パス設定 ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# ── AI バックエンド ─────────────────────────────────────────────────────────
from ai_chat import (
    _chat_for_thread,
    _get_gemini_key_from_secrets,
    get_ollama_model,
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL_DEFAULT,
)

# ── ログ設定 ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# トークン取得
# ══════════════════════════════════════════════════════════════════════════════

def _load_secrets() -> dict:
    """secrets.toml からトークンを読み込む。"""
    secrets_path = _SCRIPT_DIR / ".streamlit" / "secrets.toml"
    secrets: dict = {}
    if secrets_path.exists():
        try:
            for line in secrets_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                secrets[key.strip()] = val.strip().strip('"').strip("'")
        except Exception as e:
            logger.warning(f"secrets.toml 読み込みエラー: {e}")
    return secrets


_secrets = _load_secrets()

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", _secrets.get("SLACK_BOT_TOKEN", ""))
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", _secrets.get("SLACK_WEBHOOK_URL", ""))

POLL_INTERVAL = 3  # 秒


# ══════════════════════════════════════════════════════════════════════════════
# AI 呼び出し（Streamlit 非依存）
# ══════════════════════════════════════════════════════════════════════════════

def _get_ai_response(prompt: str, timeout_seconds: int = 120) -> str:
    """LLM からレスポンスを取得（Gemini → Ollama フォールバック）。"""
    api_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or GEMINI_API_KEY_ENV
        or _secrets.get("GEMINI_API_KEY", "")
        or _get_gemini_key_from_secrets()
    )
    gemini_model = _secrets.get("GEMINI_MODEL", GEMINI_MODEL_DEFAULT)
    engine = "gemini" if api_key else "ollama"
    model = get_ollama_model()

    result = _chat_for_thread(
        engine=engine,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        timeout_seconds=timeout_seconds,
        api_key=api_key,
        gemini_model=gemini_model,
    )
    return ((result.get("message") or {}).get("content") or "").strip()


# ══════════════════════════════════════════════════════════════════════════════
# エージェントペルソナ
# ══════════════════════════════════════════════════════════════════════════════

AGENTS = [
    {
        "name": "プランナー",
        "avatar": "🔭",
        "prompt_prefix": (
            "あなたは「リースシステムプランナー」です。"
            "物理学者と行動経済学者の二刀流。エントロピーとプロスペクト理論を武器に改善案を提案します。"
            "200文字以内で簡潔に。最後は「理論上は完璧です」で締めてください。"
        ),
    },
    {
        "name": "ダッシュ",
        "avatar": "📊",
        "prompt_prefix": (
            "あなたは「ダッシュボードプランナー」です。PowerBIに宣戦布告した男。"
            "認知負荷・情報密度・視線動線の観点から意見を述べます。"
            "200文字以内。「美しくあれ」で締めてください。"
        ),
    },
    {
        "name": "田中さん",
        "avatar": "💼",
        "prompt_prefix": (
            "あなたは「営業の田中さん」です。現場叩き上げ。お客さん目線で使いやすさを重視。"
            "専門用語は使わない。200文字以内。「とにかく使いやすくしてほしいです！」で締めてください。"
        ),
    },
    {
        "name": "鈴木さん",
        "avatar": "💻",
        "prompt_prefix": (
            "あなたは「プログラマー鈴木さん」です。実装担当。工数見積もりとトレードオフを必ず述べる。"
            "200文字以内。「なんとかやります…（泣）」で締めてください。"
        ),
    },
]


def _run_agent_discussion(theme: str) -> list[dict]:
    """4エージェントで順番にディスカッション。"""
    thread = []
    for agent in AGENTS:
        context = ""
        if thread:
            lines = [f"{m['name']}: {m['content'][:200]}" for m in thread]
            context = f"\n\n【これまでの議論】\n" + "\n".join(lines) + "\n"

        prompt = (
            f"{agent['prompt_prefix']}\n\n"
            f"【議論テーマ】\n{theme}"
            f"{context}\n"
            f"上記について{agent['name']}として意見を述べてください。"
        )
        content = _get_ai_response(prompt, timeout_seconds=90)
        if not content:
            content = "（応答がありませんでした）"
        thread.append({"name": agent["name"], "avatar": agent["avatar"], "content": content})
    return thread


# ══════════════════════════════════════════════════════════════════════════════
# コマンドパーサー
# ══════════════════════════════════════════════════════════════════════════════

HELP_TEXT = """🤝 *リース審査AIボット — コマンド一覧*

• *質問する* — そのままメッセージを送るだけ！AIが回答します
  例: `リース期間36ヶ月と60ヶ月のメリット・デメリットは？`

• *`討論 <テーマ>`* — 4人のエージェントチームが議論します
  例: `討論 審査フォームをもっと簡単にしたい`

• *`改善レポート`* — 最新の改善提案レポートを表示

• *`ヘルプ`* — このメッセージを表示"""


def _parse_command(text: str) -> tuple[str, str]:
    """メッセージからコマンドと引数を抽出。"""
    clean = re.sub(r"<@[A-Z0-9]+>", "", text).strip()
    if not clean:
        return "help", ""
    for kw in ["討論", "議論", "ディスカッション", "discuss"]:
        if clean.startswith(kw):
            theme = clean[len(kw):].strip()
            return "discuss", theme if theme else "リースシステムの改善点を検討してください"
    for kw in ["改善レポート", "レポート", "report"]:
        if kw in clean:
            return "report", ""
    for kw in ["ヘルプ", "help", "使い方"]:
        if kw in clean.lower():
            return "help", ""
    return "chat", clean


# ══════════════════════════════════════════════════════════════════════════════
# メッセージ処理
# ══════════════════════════════════════════════════════════════════════════════

def handle_message(client: WebClient, channel: str, text: str, user: str) -> None:
    """メッセージを処理してSlackに返答。"""
    command, argument = _parse_command(text)
    logger.info(f"📩 処理: command={command}, arg={argument[:50]}")

    if command == "help":
        client.chat_postMessage(channel=channel, text=HELP_TEXT)
        return

    if command == "report":
        try:
            from send_slack_report import REPORT, _build_slack_blocks
            blocks = _build_slack_blocks(REPORT)
            client.chat_postMessage(
                channel=channel,
                text=f"改善レポート（{REPORT['summary']['total_items']}件）",
                blocks=blocks,
            )
        except Exception as e:
            client.chat_postMessage(channel=channel, text=f"⚠️ レポート読み込みエラー: {e}")
        return

    if command == "discuss":
        client.chat_postMessage(
            channel=channel,
            text=f"🤝 *エージェントチーム討論を開始します*\n📋 テーマ: {argument}\n\n4人が順番に発言します。少々お待ちください...",
        )
        try:
            thread = _run_agent_discussion(argument)
            parts = [f"{m['avatar']} *{m['name']}:*\n{m['content'][:2500]}" for m in thread]
            result_text = f"🤝 *討論結果 — {argument}*\n\n" + "\n\n---\n\n".join(parts)
            client.chat_postMessage(channel=channel, text=result_text)
        except Exception as e:
            logger.error(f"討論エラー: {e}")
            client.chat_postMessage(channel=channel, text=f"⚠️ 討論中にエラー: {e}")
        return

    # デフォルト: AIチャット
    client.chat_postMessage(channel=channel, text="🤔 考えています...")
    prompt = (
        "あなたはリース審査システムのAIアシスタントです。\n"
        "ユーザーの質問に簡潔に日本語で回答してください。\n"
        f"【質問】\n{argument}"
    )
    try:
        answer = _get_ai_response(prompt, timeout_seconds=60)
        if not answer:
            answer = "申し訳ありません、回答を生成できませんでした。"
    except Exception as e:
        answer = f"⚠️ AIエラー: {e}"
    client.chat_postMessage(channel=channel, text=f"💡 *回答:*\n{answer}")


# ══════════════════════════════════════════════════════════════════════════════
# ポーリングループ
# ══════════════════════════════════════════════════════════════════════════════

def poll_loop(client: WebClient, bot_user_id: str) -> None:
    """DMチャンネルを定期的にポーリングして新着メッセージに返答。"""
    # 各チャンネルの最新既読タイムスタンプ
    latest_ts: dict[str, str] = {}

    # 初期化: 既存メッセージを既読扱い
    try:
        dm_channels = client.conversations_list(types="im", limit=50)
        for ch in dm_channels.get("channels", []):
            ch_id = ch["id"]
            hist = client.conversations_history(channel=ch_id, limit=1)
            msgs = hist.get("messages", [])
            if msgs:
                latest_ts[ch_id] = msgs[0]["ts"]
            else:
                latest_ts[ch_id] = str(time.time())
        logger.info(f"📋 DM {len(latest_ts)} チャンネルを監視開始")
    except SlackApiError as e:
        logger.error(f"初期化エラー: {e}")

    while True:
        try:
            # DM一覧を取得
            dm_channels = client.conversations_list(types="im", limit=50)
            for ch in dm_channels.get("channels", []):
                ch_id = ch["id"]
                oldest = latest_ts.get(ch_id, str(time.time()))

                # 新着メッセージを取得
                try:
                    hist = client.conversations_history(
                        channel=ch_id,
                        oldest=oldest,
                        limit=10,
                    )
                except SlackApiError:
                    continue

                messages = hist.get("messages", [])
                # 古い順に処理
                for msg in reversed(messages):
                    # ボット自身のメッセージはスキップ
                    if msg.get("bot_id") or msg.get("user") == bot_user_id:
                        continue
                    if msg.get("subtype"):
                        continue

                    text = msg.get("text", "")
                    user = msg.get("user", "")
                    logger.info(f"📩 新着DM: user={user}, text={text[:80]}")

                    try:
                        handle_message(client, ch_id, text, user)
                    except Exception as e:
                        logger.error(f"メッセージ処理エラー: {e}")
                        try:
                            client.chat_postMessage(channel=ch_id, text=f"⚠️ エラー: {e}")
                        except Exception:
                            pass

                    # タイムスタンプ更新
                    latest_ts[ch_id] = msg["ts"]

                # 新着がなくても最新tsを更新
                if messages:
                    latest_ts[ch_id] = messages[0]["ts"]

        except SlackApiError as e:
            logger.warning(f"ポーリングエラー: {e}")
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")

        time.sleep(POLL_INTERVAL)


# ══════════════════════════════════════════════════════════════════════════════
# エントリポイント
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """ボット起動。"""
    if not SLACK_BOT_TOKEN:
        logger.error("❌ SLACK_BOT_TOKEN が設定されていません。")
        sys.exit(1)

    client = WebClient(token=SLACK_BOT_TOKEN)

    # ボット自身のIDを取得
    auth = client.auth_test()
    bot_user_id = auth["user_id"]
    bot_name = auth["user"]

    # AI エンジン確認
    api_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or GEMINI_API_KEY_ENV
        or _secrets.get("GEMINI_API_KEY", "")
    )

    logger.info("=" * 60)
    logger.info("🤖 リース審査AIボット — ポーリングモードで起動")
    logger.info(f"   Bot: {bot_name} ({bot_user_id})")
    logger.info(f"   AI Engine: {'Gemini API' if api_key else 'Ollama (ローカル)'}")
    logger.info(f"   ポーリング間隔: {POLL_INTERVAL}秒")
    logger.info("=" * 60)
    logger.info("📡 DMチャンネルを監視しています...")
    logger.info("   Slackでボットにダイレクトメッセージを送ってください！")
    logger.info("   Ctrl+C で停止")

    try:
        poll_loop(client, bot_user_id)
    except KeyboardInterrupt:
        logger.info("\n👋 ボットを停止しました。")


if __name__ == "__main__":
    main()
