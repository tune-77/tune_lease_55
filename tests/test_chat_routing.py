from api.chat_routing import (
    chat_context_budget,
    chat_context_mode,
    classify_question,
    is_lightweight_chat_observation,
)


def test_lightweight_observation_routes_to_general_without_llm():
    message = "キーエンスは検査機器の製造業だから、リースも増えそうだね"

    assert is_lightweight_chat_observation(message) is True
    assert classify_question(message) == "general"


def test_explicit_analysis_request_is_not_lightweight():
    message = "キーエンスの検査機需要とリース需要の関係を根拠付きで分析して"

    assert is_lightweight_chat_observation(message) is False


def test_news_summarize_category_uses_deterministic_shortcut():
    assert classify_question("https://example.com/news を要約して保存") == "news_summarize"


def test_context_mode_and_budget_stable_shapes():
    assert chat_context_mode("案件の審査スコアを見て", "lease_screening") == "screening"
    assert chat_context_mode("詳しく根拠も教えて", "lease_knowledge") == "deep"
    assert chat_context_mode("ありがとう", "general") == "casual"
    assert chat_context_mode("長文", "lease_knowledge", long_input=True) == "long"

    screening_budget = chat_context_budget("screening")
    assert screening_budget["rag_top_k"] == 5
    assert screening_budget["use_db"] is True
    assert chat_context_budget("unknown")["history_limit"] == 32
