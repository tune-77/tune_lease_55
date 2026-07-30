from api.main import (
    _chat_context_budget,
    _chat_context_mode,
    _classify_question,
    _is_lightweight_chat_observation,
)


def test_business_hypothesis_observation_uses_lightweight_chat_path():
    message = "キーエンスは検査機器の製造業だから、検査機が増えると言う事は生産量も増えてるから、リースも増えそうだね"

    assert _is_lightweight_chat_observation(message) is True
    assert _classify_question(message) == "general"

    mode = _chat_context_mode(message, _classify_question(message))
    budget = _chat_context_budget(mode)
    assert mode == "casual"
    assert budget["rag_top_k"] == 0


def test_explicit_analysis_request_stays_out_of_lightweight_path():
    message = "キーエンスの検査機需要とリース需要の関係を根拠付きで分析して"

    assert _is_lightweight_chat_observation(message) is False

