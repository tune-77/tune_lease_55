# -*- coding: utf-8 -*-
"""
components/industry_search_agent.py
=====================================
業種が選択されたら自動でネット検索し、AI 要約をホームチャット欄に投稿する。

使い方（form_apply.py から呼ぶ）:
    from components.industry_search_agent import trigger_if_changed
    trigger_if_changed(selected_sub, detail_keyword)
"""
from __future__ import annotations
import streamlit as st

# セッションキー
_LAST_SUB    = "_isearch_last_sub"
_LAST_DETAIL = "_isearch_last_detail"

_SYSTEM = (
    "あなたはリース審査AIのアシスタントです。"
    "以下のネット検索結果をもとに、業界の最新状況・リスク・注目点を"
    "箇条書き3〜5項目で簡潔に日本語でまとめてください。"
    "リース審査担当者が審査判断に活かせる観点を優先してください。"
)


def _post_to_home_chat(content: str) -> None:
    """ホーム画面のチャット欄にアシスタントメッセージとして投稿する。"""
    if "home_messages" not in st.session_state:
        st.session_state["home_messages"] = []
    st.session_state["home_messages"].append({"role": "assistant", "content": content})


def _search_and_summarize(industry: str, detail: str = "") -> None:
    """
    ネット検索 → AI 要約 → ホームチャット投稿。
    エラーがあっても処理を止めない。
    """
    try:
        from web_services import search_latest_trends, search_bankruptcy_trends

        query_base = f"{industry} 業界動向 最新情報 2025"
        if detail:
            query_base = f"{industry} {detail} 最新動向 2025"

        # 2種類の検索を実施
        trend_text = search_latest_trends(query_base)
        risk_text  = search_bankruptcy_trends(industry)

        raw = (
            f"【業種】{industry}" + (f" / {detail}" if detail else "") + "\n\n"
            f"【最新動向】\n{trend_text}\n\n"
            f"【リスク情報】\n{risk_text}"
        )

        # AI で要約
        summary = _ai_summarize(raw, industry, detail)

        header = f"🔍 **{industry}" + (f" / {detail}" if detail else "") + " — 業界最新情報**\n\n"
        _post_to_home_chat(header + summary)

    except Exception as e:
        _post_to_home_chat(f"⚠️ 業界情報の取得中にエラーが発生しました: {e}")


def _ai_summarize(raw: str, industry: str, detail: str) -> str:
    """AI で検索結果を要約して返す。AI 未接続時は生テキストを整形して返す。"""
    try:
        from ai_chat import (
            _chat_for_thread, is_ai_available, get_ollama_model,
            GEMINI_API_KEY_ENV, GEMINI_MODEL_DEFAULT, _get_gemini_key_from_secrets,
        )
        if not is_ai_available():
            return raw[:1000]

        engine       = st.session_state.get("ai_engine", "ollama")
        api_key      = (
            (st.session_state.get("gemini_api_key") or "").strip()
            or GEMINI_API_KEY_ENV
            or _get_gemini_key_from_secrets()
        )
        gemini_model = st.session_state.get("gemini_model", GEMINI_MODEL_DEFAULT)

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": raw[:3000]},
        ]
        res = _chat_for_thread(
            engine, get_ollama_model(), messages,
            timeout_seconds=90,
            api_key=api_key,
            gemini_model=gemini_model,
        )
        return (res.get("message") or {}).get("content", "") or raw[:1000]

    except Exception:
        return raw[:1000]


def trigger_if_changed(selected_sub: str, detail: str = "") -> None:
    """
    中分類または詳細キーワードが前回と変わっていたら検索を実行する。
    form_apply.py の業種選択直後に呼ぶ。
    """
    sub_changed    = st.session_state.get(_LAST_SUB)    != selected_sub
    detail_changed = st.session_state.get(_LAST_DETAIL) != detail and detail.strip()

    if sub_changed:
        st.session_state[_LAST_SUB]    = selected_sub
        st.session_state[_LAST_DETAIL] = detail
        _search_and_summarize(selected_sub, detail)

    elif detail_changed:
        st.session_state[_LAST_DETAIL] = detail
        _search_and_summarize(selected_sub, detail)
