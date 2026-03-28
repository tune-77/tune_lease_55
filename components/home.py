# -*- coding: utf-8 -*-
"""
ホーム画面 — カード型ナビゲーション（6枚メイン + もっと見る）
・最近使った機能バッジ表示
・AIチャットはFABアイコンで展開
"""
import json
import os
import streamlit as st

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_RECENT_FILE = os.path.join(_DATA_DIR, "recent_modes.json")
_MAX_RECENT = 3  # 最近使用として記録する件数

# ── メインカード（常時表示 6枚）──────────────────────────────────────────────
_MAIN_CARDS = [
    {"icon": "💬", "title": "リースくん",   "desc": "チャット形式で審査入力",      "mode": "💬 リースくん",                        "color": "#1A1A2E"},
    {"icon": "📋", "title": "審査・分析",   "desc": "新規リース審査・スコアリング", "mode": "📋 審査・分析",                        "color": "#2563eb"},
    {"icon": "📄", "title": "審査レポート", "desc": "審査結果をレポート表示",       "mode": "📄 審査レポート",                      "color": "#1d4ed8"},
    {"icon": "📝", "title": "結果登録",     "desc": "成約・失注を記録",             "mode": "📝 結果登録 (成約/失注)",              "color": "#16a34a"},
    {"icon": "📊", "title": "ダッシュボード","desc": "履歴分析・実績確認",           "mode": "📊 履歴分析・実績ダッシュボード",      "color": "#d97706"},
    {"icon": "🤖", "title": "エージェントハブ","desc": "8種の自動化エージェント",   "mode": "🤖 汎用エージェントハブ",             "color": "#7c3aed"},
]

# ── サブカード（「もっと見る」で展開）────────────────────────────────────────
_SUB_CARDS = [
    {"icon": "🏭", "title": "物件ファイナンス","desc": "物件ファイナンス審査",      "mode": "🏭 物件ファイナンス審査",             "color": "#0891b2"},
    {"icon": "📈", "title": "定量要因分析", "desc": "財務指標の要因分析",           "mode": "📈 定量要因分析 (50件〜)",            "color": "#ea580c"},
    {"icon": "📉", "title": "定性要因分析", "desc": "定性評価の傾向分析",           "mode": "📉 定性要因分析 (50件〜)",            "color": "#be185d"},
    {"icon": "🤝", "title": "エージェント議論","desc": "AIチームでディスカッション","mode": "🤝 エージェントチーム議論",           "color": "#0f766e"},
    {"icon": "⚙️", "title": "審査ルール設定","desc": "審査ロジックのカスタマイズ",  "mode": "⚙️ 審査ルール設定",                  "color": "#475569"},
    {"icon": "🔧", "title": "係数分析",     "desc": "ベイズ係数の分析・更新",       "mode": "🔧 係数分析・更新 (β)",              "color": "#6b7280"},
]

# ── 最近使用記録 ─────────────────────────────────────────────────────────────

def _load_recent() -> list[str]:
    try:
        if os.path.exists(_RECENT_FILE):
            with open(_RECENT_FILE, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_recent(mode: str) -> None:
    try:
        recent = _load_recent()
        if mode in recent:
            recent.remove(mode)
        recent.insert(0, mode)
        recent = recent[:_MAX_RECENT]
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_RECENT_FILE, "w", encoding="utf-8") as f:
            json.dump(recent, f, ensure_ascii=False)
    except Exception:
        pass


# ── CSS ─────────────────────────────────────────────────────────────────────

_CSS = """
<style>
.home-header {
    text-align: center;
    padding: 1.6rem 0 1.2rem;
}
.home-header h1 {
    font-size: 2rem;
    font-weight: 800;
    color: #1e3a5f;
    margin: 0;
}
.home-header p {
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 0.35rem;
}
/* カードボタン共通 */
div[data-testid="stButton"] button {
    height: auto !important;
    min-height: 105px !important;
    white-space: pre-wrap !important;
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    border-top: 4px solid #2563eb !important;
    background: linear-gradient(160deg, #f0f4ff 0%, #e8f0fe 60%, #fdf4ff 100%) !important;
    box-shadow: 0 2px 8px rgba(30,58,95,0.08) !important;
    font-size: 0.76rem !important;
    color: #1e293b !important;
    transition: transform 0.15s, box-shadow 0.15s;
    padding: 0.65rem 0.4rem !important;
    line-height: 1.45 !important;
    overflow: hidden !important;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(30,58,95,0.16) !important;
    filter: brightness(0.96);
}
div[data-testid="stButton"] button p {
    margin: 0 !important;
    line-height: 1.45 !important;
    overflow: hidden !important;
    word-break: break-word !important;
}
/* CTAバナー */
.home-cta-banner {
    background: linear-gradient(135deg, #1A1A2E 0%, #2d2d4e 100%);
    border-radius: 16px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}
.home-cta-text h2 {
    color: #E8A838;
    font-size: 1.15rem;
    font-weight: 800;
    margin: 0 0 .25rem;
}
.home-cta-text p {
    color: rgba(255,255,255,0.75);
    font-size: .82rem;
    margin: 0;
}
/* FABチャットパネル */
.fab-chat-panel {
    background: #fff;
    border: 1.5px solid #e2e8f0;
    border-radius: 16px;
    padding: 1rem 1.2rem 0.8rem;
    margin-top: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
</style>
"""

_HOME_CHAT_SYSTEM = (
    "あなたは温水式リース審査AIのアシスタントです。"
    "リース審査・融資判断・財務指標・業界動向に関する質問に日本語で答えてください。"
    "回答は簡潔にまとめ、必要に応じて箇条書きを使ってください。"
)


def _render_fab_chat() -> None:
    """FABボタンで開閉するAIチャットパネル。"""
    if "home_fab_open" not in st.session_state:
        st.session_state["home_fab_open"] = False
    if "home_messages" not in st.session_state:
        st.session_state["home_messages"] = []

    # FABボタン
    label = "💬 AI相談を閉じる" if st.session_state["home_fab_open"] else "💬 AI審査アシスタントに相談する"
    if st.button(label, key="home_fab_toggle", width='content'):
        st.session_state["home_fab_open"] = not st.session_state["home_fab_open"]
        st.rerun()

    if not st.session_state["home_fab_open"]:
        return

    # チャットパネル
    st.markdown('<div class="fab-chat-panel">', unsafe_allow_html=True)
    st.caption("💡 リース審査・融資・財務に関する質問をどうぞ")

    chat_box = st.container(height=300)
    with chat_box:
        for m in st.session_state["home_messages"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    pending = st.session_state.pop("home_chat_pending_q", None)
    if pending:
        with chat_box:
            with st.chat_message("user"):
                st.markdown(pending)
            with st.chat_message("assistant"):
                try:
                    from ai_chat import (
                        _chat_for_thread, is_ai_available,
                        GEMINI_API_KEY_ENV, GEMINI_MODEL_DEFAULT,
                        _get_gemini_key_from_secrets, get_ollama_model,
                    )
                    _engine = st.session_state.get("ai_engine", "ollama")
                    _api_key = (
                        (st.session_state.get("gemini_api_key") or "").strip()
                        or GEMINI_API_KEY_ENV
                        or _get_gemini_key_from_secrets()
                    )
                    _gemini_model = st.session_state.get("gemini_model", GEMINI_MODEL_DEFAULT)
                    if not is_ai_available():
                        content = "⚠️ AIサーバーに接続できません。サイドバーでエンジンを確認してください。"
                    else:
                        history = [{"role": "system", "content": _HOME_CHAT_SYSTEM}]
                        for m in st.session_state["home_messages"]:
                            history.append({"role": m["role"], "content": m["content"]})
                        history.append({"role": "user", "content": pending})
                        with st.spinner("思考中..."):
                            ans = _chat_for_thread(
                                _engine, get_ollama_model(), history,
                                timeout_seconds=120, api_key=_api_key,
                                gemini_model=_gemini_model,
                            )
                        content = (ans.get("message") or {}).get("content", "") or "（応答なし）"
                    st.markdown(content)
                    st.session_state["home_messages"].append({"role": "assistant", "content": content})
                except Exception as e:
                    st.error(f"AIエラー: {e}")

    user_input = st.chat_input("質問を入力…（例: 自己資本比率の目安は？）", key="home_chat_input_widget")
    if user_input and user_input.strip():
        st.session_state["home_messages"].append({"role": "user", "content": user_input.strip()})
        st.session_state["home_chat_pending_q"] = user_input.strip()
        st.rerun()

    col_clr, _ = st.columns([1, 4])
    with col_clr:
        if st.session_state["home_messages"]:
            if st.button("🗑️ 履歴クリア", key="home_chat_clear"):
                st.session_state["home_messages"] = []
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def _render_cards(cards: list, recent: list[str], offset: int = 0) -> None:
    """カードグリッドを描画。recentに含まれるものはバッジ付き。"""
    cols = st.columns(3)
    for i, card in enumerate(cards):
        with cols[i % 3]:
            is_recent = card["mode"] in recent
            badge = "⭐ よく使う\n\n" if is_recent else ""
            label = f"{badge}{card['icon']}\n\n**{card['title']}**\n\n{card['desc']}"
            if st.button(label, key=f"home_card_{offset + i}",
                         width='stretch', help=card["desc"]):
                _save_recent(card["mode"])
                st.session_state["_pending_mode"] = card["mode"]
                st.rerun()


def render_home() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="home-header"><h1>🏢 温水式 リース審査AI</h1>'
        '<p>機能を選んでください</p></div>',
        unsafe_allow_html=True,
    )

    recent = _load_recent()

    # ── CTAバナー ─────────────────────────────────────────────────────────────
    st.markdown("""
<div class="home-cta-banner">
  <div class="home-cta-text">
    <h2>🎩 はじめての方はリースくんから</h2>
    <p>チャット形式で質問に答えるだけ。10ステップで審査データを入力できます。</p>
  </div>
</div>
""", unsafe_allow_html=True)
    if st.button("🎩 リースくんで審査開始 →", key="home_cta_wizard", type="primary"):
        _save_recent("💬 リースくん")
        st.session_state["_pending_mode"] = "💬 リースくん"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── メインカード 6枚 ─────────────────────────────────────────────────────
    _render_cards(_MAIN_CARDS, recent, offset=0)

    # ── もっと見る（サブカード 6枚）──────────────────────────────────────────
    with st.expander("▼ もっと見る（分析・設定）", expanded=False):
        _render_cards(_SUB_CARDS, recent, offset=len(_MAIN_CARDS))

    st.divider()

    # ── FAB AIチャット ───────────────────────────────────────────────────────
    _render_fab_chat()
