# -*- coding: utf-8 -*-
"""
ホーム画面 — カード型ナビゲーション + AI審査チャット
"""
import streamlit as st

CARDS = [
    {
        "icon": "💬",
        "title": "リースくん",
        "desc": "チャット形式で審査入力",
        "mode": "💬 リースくん",
        "color": "#1A1A2E",
    },
    {
        "icon": "📋",
        "title": "審査・分析",
        "desc": "新規リース審査・スコアリング",
        "mode": "📋 審査・分析",
        "color": "#2563eb",
    },
    {
        "icon": "📄",
        "title": "審査レポート",
        "desc": "審査結果をレポート表示",
        "mode": "📄 審査レポート",
        "color": "#1d4ed8",
    },
    {
        "icon": "🤖",
        "title": "エージェントハブ",
        "desc": "8種の自動化エージェント",
        "mode": "🤖 汎用エージェントハブ",
        "color": "#7c3aed",
    },
    {
        "icon": "🏭",
        "title": "物件ファイナンス",
        "desc": "物件ファイナンス審査",
        "mode": "🏭 物件ファイナンス審査",
        "color": "#0891b2",
    },
    {
        "icon": "📝",
        "title": "結果登録",
        "desc": "成約・失注を記録",
        "mode": "📝 結果登録 (成約/失注)",
        "color": "#16a34a",
    },
    {
        "icon": "📊",
        "title": "ダッシュボード",
        "desc": "履歴分析・実績確認",
        "mode": "📊 履歴分析・実績ダッシュボード",
        "color": "#d97706",
    },
    {
        "icon": "📈",
        "title": "定量要因分析",
        "desc": "財務指標の要因分析",
        "mode": "📈 定量要因分析 (50件〜)",
        "color": "#ea580c",
    },
    {
        "icon": "📉",
        "title": "定性要因分析",
        "desc": "定性評価の傾向分析",
        "mode": "📉 定性要因分析 (50件〜)",
        "color": "#be185d",
    },
    {
        "icon": "🤝",
        "title": "エージェント議論",
        "desc": "AIチームでディスカッション",
        "mode": "🤝 エージェントチーム議論",
        "color": "#0f766e",
    },
    {
        "icon": "⚙️",
        "title": "審査ルール設定",
        "desc": "審査ロジックのカスタマイズ",
        "mode": "⚙️ 審査ルール設定",
        "color": "#475569",
    },
    {
        "icon": "🔧",
        "title": "係数分析",
        "desc": "ベイズ係数の分析・更新",
        "mode": "🔧 係数分析・更新 (β)",
        "color": "#6b7280",
    },
]

_CSS = """
<style>
.home-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
}
.home-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1e3a5f;
    margin: 0;
}
.home-header p {
    color: #64748b;
    font-size: 1rem;
    margin-top: 0.4rem;
}
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 1rem;
    padding: 1rem 0 2rem;
}
.nav-card {
    border-radius: 14px;
    padding: 1.4rem 1rem 1.2rem;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    text-align: center;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
    text-decoration: none;
}
.nav-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
}
.nav-card .card-icon {
    font-size: 2.2rem;
    line-height: 1;
    margin-bottom: 0.6rem;
}
.nav-card .card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.3rem;
}
.nav-card .card-desc {
    font-size: 0.75rem;
    color: #64748b;
    line-height: 1.4;
}
.card-accent {
    width: 36px;
    height: 4px;
    border-radius: 2px;
    margin: 0.6rem auto 0;
}
</style>
"""

_HOME_CHAT_SYSTEM = (
    "あなたは温水式リース審査AIのアシスタントです。"
    "リース審査・融資判断・財務指標・業界動向に関する質問に日本語で答えてください。"
    "回答は簡潔にまとめ、必要に応じて箇条書きを使ってください。"
)


def _render_home_chat() -> None:
    """ホーム画面右カラムのAIチャット。審査タブとは独立した session state を使用。"""
    try:
        from ai_chat import (
            _chat_for_thread,
            is_ai_available,
            GEMINI_API_KEY_ENV,
            GEMINI_MODEL_DEFAULT,
            _get_gemini_key_from_secrets,
            get_ollama_model,
        )
    except Exception as e:
        st.caption(f"⚠️ AIモジュール読み込みエラー: {e}")
        return

    st.write("")
    st.write("")
    st.write("")
    st.markdown('<p style="font-size:0.95rem;font-weight:700;color:#1e3a5f;margin:0.3rem 0 0.1rem;">💬 AI審査アシスタント</p>', unsafe_allow_html=True)
    st.caption("リース審査・融資・財務に関する質問をどうぞ")

    # session state 初期化
    if "home_messages" not in st.session_state:
        st.session_state["home_messages"] = []

    # チャット履歴表示
    chat_box = st.container(height=380)
    with chat_box:
        for m in st.session_state["home_messages"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    # 送信待ちクエリを処理
    pending = st.session_state.pop("home_chat_pending_q", None)
    if pending:
        with chat_box:
            with st.chat_message("user"):
                st.markdown(pending)
            with st.chat_message("assistant"):
                _engine = st.session_state.get("ai_engine", "ollama")
                _api_key = (
                    (st.session_state.get("gemini_api_key") or "").strip()
                    or GEMINI_API_KEY_ENV
                    or _get_gemini_key_from_secrets()
                )
                _gemini_model = st.session_state.get("gemini_model", GEMINI_MODEL_DEFAULT)
                if not is_ai_available():
                    _engine_msgs = {
                        "anythingllm": "AnythingLLM に接続できません。http://127.0.0.1:3001 が起動しているか、APIキーを確認してください。",
                        "gemini": "Gemini APIキーを設定してください（サイドバー「AIモデル設定」）。",
                    }
                    msg = _engine_msgs.get(_engine, "AIサーバー（Ollama）が起動していません。`ollama serve` を実行するか、サイドバーで別のエンジンに切り替えてください。")
                    st.error(msg)
                    st.session_state["home_messages"].append({"role": "assistant", "content": msg})
                else:
                    history = [{"role": "system", "content": _HOME_CHAT_SYSTEM}]
                    for m in st.session_state["home_messages"]:
                        history.append({"role": m["role"], "content": m["content"]})
                    history.append({"role": "user", "content": pending})
                    with st.spinner("思考中..."):
                        ans = _chat_for_thread(
                            _engine, get_ollama_model(), history,
                            timeout_seconds=120,
                            api_key=_api_key,
                            gemini_model=_gemini_model,
                        )
                    content = (ans.get("message") or {}).get("content", "") or "（応答がありませんでした）"
                    st.markdown(content)
                    st.session_state["home_messages"].append({"role": "assistant", "content": content})

    # 入力欄
    user_input = st.chat_input("質問を入力…（例: 自己資本比率の目安は？）", key="home_chat_input_widget")
    if user_input and user_input.strip():
        st.session_state["home_messages"].append({"role": "user", "content": user_input.strip()})
        st.session_state["home_chat_pending_q"] = user_input.strip()
        st.rerun()

    # 履歴クリアボタン
    if st.session_state["home_messages"]:
        if st.button("🗑️ 履歴をクリア", key="home_chat_clear", help="チャット履歴を消去します"):
            st.session_state["home_messages"] = []
            st.rerun()


def render_home() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    col_nav, col_chat = st.columns([2, 1])

    with col_nav:
        st.markdown(
            '<div class="home-header"><h1>🏢 温水式 リース審査AI</h1>'
            '<p>機能を選んでください</p></div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(3)
        for i, card in enumerate(CARDS):
            with cols[i % 3]:
                if st.button(
                    f"{card['icon']}\n\n**{card['title']}**\n\n{card['desc']}",
                    key=f"home_card_{i}",
                    use_container_width=True,  # type: ignore
                    help=card["desc"],
                ):
                    st.session_state["_pending_mode"] = card["mode"]
                    st.rerun()

        # ボタンをカード風にスタイリング（各カード固有色を border-top に注入）
        _key_color_rules = "\n".join(
            f'button[kind="secondary"][data-testid="baseButton-secondary"]:has([data-key="home_card_{i}"]), '
            f'div:has(> button[key="home_card_{i}"]) button {{ border-top-color: {c["color"]} !important; }}'
            for i, c in enumerate(CARDS)
        )
        st.markdown(f"""
<style>
/* カードナビ全体スタイル */
div[data-testid="stButton"] button {{
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
    text-overflow: ellipsis !important;
}}
div[data-testid="stButton"] button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(30,58,95,0.16) !important;
    filter: brightness(0.96);
}}
div[data-testid="stButton"] button p {{
    margin: 0 !important;
    line-height: 1.45 !important;
    overflow: hidden !important;
    word-break: break-word !important;
}}
</style>
""", unsafe_allow_html=True)

    with col_chat:
        _render_home_chat()
