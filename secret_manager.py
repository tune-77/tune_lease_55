"""
秘密情報読み込みの共通化モジュール
優先順位: 環境変数 > st.secrets > secrets.toml > fallback
"""

import os
import streamlit as st
from pathlib import Path
import toml
from typing import Optional

SECRETS_TOML_PATH = Path(__file__).parent / ".streamlit" / "secrets.toml"


def get_secret_value(key: str, fallback: Optional[str] = None) -> Optional[str]:
    """
    秘密情報を統一的に取得する関数
    優先順位: 環境変数 > st.secrets > secrets.toml > fallback
    """
    # 1. 環境変数から取得
    value = os.getenv(key)
    if value:
        return value

    # 2. Streamlit secretsから取得
    try:
        value = st.secrets.get(key)
        if value:
            return value
    except Exception:
        # st.secrets は secrets.toml を見つけられないとき StreamlitSecretNotFoundError
        # を投げる。これは FileNotFoundError のサブクラスで、AttributeError でも
        # KeyError でもない。以前はこの2つしか捕捉しておらず、リポジトリ外を
        # カレントディレクトリとして起動したジョブ（launchd の定期実行など）では
        # 例外がそのまま呼び出し元へ抜け、手順3に到達していなかった。
        # SECRETS_TOML_PATH は __file__ 基準の絶対パスなので、到達しさえすれば
        # cwd に関係なく読める。手順2は優先順位つきフォールバックの途中段でしか
        # ないため、失敗の種類を問わず次へ進めてよい。
        pass

    # 3. secrets.tomlから取得
    if SECRETS_TOML_PATH.exists():
        try:
            secrets = toml.load(SECRETS_TOML_PATH)
            value = secrets.get(key)
            if value:
                return value
        except Exception:
            pass

    # 4. fallback
    return fallback


def get_gemini_api_key() -> Optional[str]:
    """Gemini APIキーを取得"""
    return get_secret_value("GEMINI_API_KEY")


def get_slack_bot_token() -> Optional[str]:
    """Slack Bot Tokenを取得"""
    return get_secret_value("SLACK_BOT_TOKEN")


def get_slack_app_token() -> Optional[str]:
    """Slack App Tokenを取得"""
    return get_secret_value("SLACK_APP_TOKEN")


def get_slack_webhook_url() -> Optional[str]:
    """Slack Webhook URLを取得"""
    return get_secret_value("SLACK_WEBHOOK_URL")


def get_anything_llm_key() -> Optional[str]:
    """AnythingLLM APIキーを取得"""
    return get_secret_value("ANYTHING_LLM_API_KEY")


def get_openai_api_key() -> Optional[str]:
    """OpenAI APIキーを取得"""
    return get_secret_value("OPENAI_API_KEY")
