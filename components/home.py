# -*- coding: utf-8 -*-
"""
ホーム画面 — カード型ナビゲーション
"""
import streamlit as st

CARDS = [
    {
        "icon": "📋",
        "title": "審査・分析",
        "desc": "新規リース審査・スコアリング",
        "mode": "📋 審査・分析",
        "color": "#2563eb",
    },
    {
        "icon": "⚡",
        "title": "バッチ審査",
        "desc": "複数案件を一括審査",
        "mode": "⚡ バッチ審査",
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
    {
        "icon": "📐",
        "title": "係数入力",
        "desc": "事前係数の手動入力",
        "mode": "📐 係数入力（事前係数）",
        "color": "#6b7280",
    },
    {
        "icon": "🪵",
        "title": "アプリログ",
        "desc": "動作ログの確認",
        "mode": "🪵 アプリログ",
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


def render_home() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="home-header"><h1>🏢 温水式 リース審査AI</h1>'
        '<p>機能を選んでください</p></div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for i, card in enumerate(CARDS):
        with cols[i % 4]:
            if st.button(
                f"{card['icon']}\n\n**{card['title']}**\n\n{card['desc']}",
                key=f"home_card_{i}",
                use_container_width=True,  # type: ignore
                help=card["desc"],
            ):
                st.session_state["_pending_mode"] = card["mode"]
                st.rerun()

    # ボタンをカード風にスタイリング
    st.markdown("""
<style>
div[data-testid="stButton"] button {
    height: 110px;
    white-space: pre-wrap;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    background: #ffffff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    font-size: 0.82rem;
    color: #1e293b;
    transition: transform 0.15s, box-shadow 0.15s;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    border-color: #2563eb;
    color: #2563eb;
}
</style>
""", unsafe_allow_html=True)
