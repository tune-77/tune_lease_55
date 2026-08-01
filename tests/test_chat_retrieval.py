from api.chat_retrieval import build_chat_retrieval_context, extract_vertex_search_hint


def test_extract_vertex_search_hint_uses_compact_marker_block():
    message = "\n".join(
        [
            "【審査分析画面からの紫苑レビュー依頼】",
            "長いレビュー本文",
            "",
            "【Vertex補助検索ヒント】",
            "工作機械 補助金 リース料軽減",
            "公募要領 対象経費 Q_risk",
            "",
            "前提: スコア55点",
        ]
    )

    assert extract_vertex_search_hint(message) == "工作機械 補助金 リース料軽減 公募要領 対象経費 Q_risk"


def test_build_chat_retrieval_context_zero_top_k_returns_stable_shape():
    result = build_chat_retrieval_context(
        "補助金前提の工作機械リース",
        rag_top_k=0,
        question_category="lease_knowledge",
        is_general_response_mode=False,
        fallback_search=lambda _message, _top_k: [{"text": "unused"}],
    )

    assert result.rag_context == ""
    assert result.rag_refs == []
    assert result.rag_knowledge_refs == []
    assert result.vertex_agent_search["status"] == "not_attempted"
    assert result.vertex_answer_api["status"] == "not_attempted"
