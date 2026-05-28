"""
RAGナレッジQ&Aビュー。
ユーザーの質問に対してナレッジベースから関連チャンクを検索し、
AIが回答を生成する Q&A インターフェース。
"""
from __future__ import annotations

import streamlit as st

_SOURCE_LABELS: dict[str, str] = {
    "faq": "FAQ",
    "manual": "審査マニュアル",
    "case": "審査事例",
    "industry": "業種別ガイド",
    "scoring": "スコアリング",
    "improvement": "スコア改善ガイド",
    "knowhow": "審査ノウハウ",
}

_SOURCE_COLORS: dict[str, str] = {
    "faq": "#2563eb",
    "manual": "#16a34a",
    "case": "#9333ea",
    "industry": "#d97706",
    "scoring": "#0891b2",
    "improvement": "#db2777",
    "knowhow": "#65a30d",
}


def _badge(source: str) -> str:
    color = _SOURCE_COLORS.get(source, "#6b7280")
    label = _SOURCE_LABELS.get(source, source)
    return (
        f'<span style="background:{color};color:white;padding:1px 7px;'
        f'border-radius:9px;font-size:0.72rem;font-weight:600;">{label}</span>'
    )


def render_rag_qa_view() -> None:
    st.title("📚 RAGナレッジQ&A")
    st.caption("審査マニュアル・FAQ・業種ガイド・事例集を横断検索し、AIが回答します。")

    from rag_knowledge import retrieve, build_rag_context, get_index_stats
    from ai_chat import chat_with_retry, get_ollama_model, is_ai_available

    # ─── インデックス統計 ────────────────────────────────────────────────
    with st.expander("🗂 ナレッジインデックス情報", expanded=False):
        stats = get_index_stats()
        cols = st.columns(3)
        cols[0].metric("総チャンク数", f"{stats['chunk_count']:,}")
        cols[1].metric("語彙サイズ", f"{stats['vocab_size']:,}")
        src_lines = [f"- {_SOURCE_LABELS.get(k, k)}: {v} チャンク" for k, v in stats["sources"].items()]
        cols[2].markdown("**ソース内訳**\n" + "\n".join(src_lines))

    st.divider()

    # ─── 質問入力 ────────────────────────────────────────────────────────
    query = st.text_input(
        "質問を入力してください",
        placeholder="例: 建設業のリース審査で重視する指標は？",
        key="rag_query_input",
    )
    top_k = st.slider("参照チャンク数", min_value=3, max_value=10, value=5, key="rag_top_k")

    col_search, col_clear = st.columns([2, 1])
    do_search = col_search.button("🔍 検索＆回答生成", type="primary", use_container_width=True)
    if col_clear.button("🗑 履歴をクリア", use_container_width=True):
        st.session_state.pop("rag_history", None)
        st.rerun()

    if do_search and query.strip():
        _run_qa(query.strip(), top_k, retrieve, build_rag_context, chat_with_retry, get_ollama_model, is_ai_available)

    # ─── Q&A 履歴表示 ───────────────────────────────────────────────────
    history: list[dict] = st.session_state.get("rag_history", [])
    if history:
        st.divider()
        st.subheader("📋 回答履歴")
        for item in reversed(history):
            _render_history_item(item)


def _run_qa(
    query: str,
    top_k: int,
    retrieve,
    build_rag_context,
    chat_with_retry,
    get_ollama_model,
    is_ai_available,
) -> None:
    """検索・回答生成を実行してセッションに保存する。"""
    with st.spinner("ナレッジを検索中..."):
        hits = retrieve(query, top_k=top_k)

    if not hits:
        st.warning("関連するナレッジが見つかりませんでした。")
        return

    # ─── 検索結果表示 ────────────────────────────────────────────────────
    st.subheader("🔎 関連ナレッジ")
    for i, hit in enumerate(hits, 1):
        badge_html = _badge(hit.chunk.source)
        score_pct = f"{hit.score * 100:.1f}%"
        with st.expander(f"[{i}] {hit.chunk.title}　{score_pct} 一致", expanded=(i == 1)):
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown(f"> {hit.chunk.text}")

    # ─── AI回答生成 ──────────────────────────────────────────────────────
    ai_answer = ""
    if is_ai_available():
        rag_ctx = build_rag_context(query, top_k=top_k)
        prompt = (
            "あなたはリース審査AIアシスタントです。\n"
            "以下のナレッジベース情報を必ず根拠として参照し、"
            "審査担当者の質問に簡潔・正確に日本語で答えてください。\n"
            "ナレッジに記載のない内容については「記載がありません」と明示してください。\n\n"
            f"{rag_ctx}\n\n"
            f"【質問】\n{query}"
        )
        with st.spinner("AI回答を生成中..."):
            try:
                resp = chat_with_retry(
                    model=get_ollama_model(),
                    messages=[{"role": "user", "content": prompt}],
                    retries=1,
                    timeout_seconds=90,
                )
                ai_answer = ((resp.get("message") or {}).get("content") or "").strip()
            except Exception as e:
                st.error(f"AI回答生成エラー: {e}")

        if ai_answer:
            st.subheader("🤖 AI回答")
            st.markdown(ai_answer)
        else:
            st.info("AI回答を取得できませんでした。上の検索結果をご参照ください。")
    else:
        st.info("AIが利用できません（APIキー未設定 or Ollama未起動）。サイドバーで設定してください。")

    # ─── 履歴に保存 ──────────────────────────────────────────────────────
    if "rag_history" not in st.session_state:
        st.session_state["rag_history"] = []
    st.session_state["rag_history"].append({
        "query": query,
        "hits": [
            {"title": h.chunk.title, "source": h.chunk.source, "text": h.chunk.text, "score": h.score}
            for h in hits
        ],
        "answer": ai_answer,
    })


def _render_history_item(item: dict) -> None:
    query = item.get("query", "")
    answer = item.get("answer", "")
    hits = item.get("hits", [])

    with st.container(border=True):
        st.markdown(f"**Q: {query}**")
        if answer:
            st.markdown(answer)
        else:
            st.caption("（AI回答なし）")
        if hits:
            with st.expander("参照チャンク", expanded=False):
                for h in hits:
                    badge_html = _badge(h.get("source", ""))
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.caption(f"{h.get('title','')} — {h.get('text','')[:120]}…")
