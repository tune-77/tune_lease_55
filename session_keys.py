"""
session_keys.py — Streamlit session_state キー定数

全ファイルで使う主要なキーをここに集約。
既存コードはまだ文字列リテラルを使っているが、
新規コードはこのモジュールの定数を参照すること。

使い方:
    from session_keys import SK
    st.session_state[SK.LAST_RESULT]
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class _SK:
    # ── AI エンジン設定 ──────────────────────────────────────────────────────
    AI_ENGINE: str             = "ai_engine"           # "gemini" / "ollama" / "anythingllm"
    GEMINI_API_KEY: str        = "gemini_api_key"
    GEMINI_MODEL: str          = "gemini_model"
    OLLAMA_MODEL: str          = "ollama_model"
    HUMOR_STYLE: str           = "humor_style"         # "standard" / "yanami"

    # ── 審査結果 ─────────────────────────────────────────────────────────────
    LAST_RESULT: str           = "last_result"
    LAST_SUBMITTED_INPUTS: str = "last_submitted_inputs"
    CURRENT_CASE_ID: str       = "current_case_id"

    # ── ウィザード（リースくん） ─────────────────────────────────────────────
    WIZ_STEP: str              = "wiz_step"
    WIZ_DATA: str              = "wiz_data"
    WIZ_HISTORY: str           = "wiz_history"
    WIZARD_FORM_RESULT: str    = "wizard_form_result"

    # ── 軍師モード ───────────────────────────────────────────────────────────
    GUNSHI_AUTO_RESULT: str    = "gunshi_auto_result"
    GUNSHI_LLM_TEXT: str       = "gunshi_llm_text"
    GUNSHI_LAST_CASE_ID: str   = "gunshi_last_case_id"
    GUNSHI_TREND_300: str      = "_gunshi_trend_300"
    GUNSHI_CACHE_SCORE: str    = "_gunshi_cache_score"
    GUNSHI_CACHE_BN_HASH: str  = "_gunshi_cache_bn_hash"

    # ── ベイズネット ─────────────────────────────────────────────────────────
    BN_S_RESULT: str           = "_bn_s_result"
    BN_S_EVIDENCE: str         = "_bn_s_evidence"

    # ── ホーム画面 ───────────────────────────────────────────────────────────
    HOME_MESSAGES: str         = "home_messages"
    HOME_FAB_OPEN: str         = "home_fab_open"

    # ── ナビゲーション ───────────────────────────────────────────────────────
    MAIN_MODE: str             = "main_mode"
    SIDEBAR_GROUP: str         = "sidebar_group"
    PENDING_MODE: str          = "_pending_mode"
    NAV_MODE_WIDGET: str       = "nav_mode_widget"

    # ── チャット・AI相談 ─────────────────────────────────────────────────────
    CHAT_LOADING: str          = "chat_loading"
    CONSULTATION_INPUT: str    = "consultation_input"
    AUTO_AI_COMMENT: str       = "auto_ai_comment"

    # ── フォーム入力値（主要財務項目） ──────────────────────────────────────
    NENSHU: str                = "nenshu"
    RIEKI: str                 = "rieki"
    NET_ASSETS: str            = "net_assets"
    TOTAL_ASSETS: str          = "total_assets"
    LEASE_TERM: str            = "lease_term"
    ACQUISITION_COST: str      = "acquisition_cost"
    SALES_DEPT: str            = "sales_dept"

    # ── 競合・案件情報 ───────────────────────────────────────────────────────
    COMPETITOR_RATE: str       = "competitor_rate"
    SELECTED_MAJOR: str        = "select_major"
    SELECTED_SUB: str          = "select_sub"

    # ── Slack 通知設定 ───────────────────────────────────────────────────────
    SLACK_ENABLED: str         = "slack_enabled"
    SLACK_WEBHOOK_URL: str     = "slack_webhook_url"
    SLACK_CHANNEL_NAME: str    = "slack_channel_name"
    SLACK_NOTIFY_THREAD: str   = "slack_notify_thread"
    SLACK_NOTIFY_VERDICT: str  = "slack_notify_verdict"

    # ── エージェントチーム (AT) ──────────────────────────────────────────────
    AT_THEME: str              = "at_theme"
    AT_HISTORY: str            = "at_history"
    AT_PENDING: str            = "at_pending"
    AT_REBUTTAL_TARGET: str    = "at_rebuttal_target"
    AT_FACILITATOR_RESULT: str = "at_facilitator_result"
    AT_CODE_RESULTS: str       = "at_code_results"
    AT_APPLIED_FILES: str      = "at_applied_files"
    AT_SHINSA_HISTORY: str     = "at_shinsa_history"
    AT_SHINSA_PENDING: str     = "at_shinsa_pending"

    # ── こいぬエージェント ───────────────────────────────────────────────────
    KOINU_CHAT_HISTORY: str    = "koinu_chat_history"
    KOINU_MEMO_LOG: str        = "koinu_memo_log"


# シングルトンインスタンス（from session_keys import SK で使う）
SK = _SK()
