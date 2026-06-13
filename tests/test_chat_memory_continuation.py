from unittest.mock import Mock

import api.chat_memory as chat_memory


def _response(text: str, finish_reason: str) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "candidates": [
            {
                "finishReason": finish_reason,
                "content": {"parts": [{"text": text}]},
            }
        ]
    }
    return response


def test_chat_max_tokens_default_is_large_enough(monkeypatch):
    monkeypatch.delenv("MEBUKI_CHAT_MAX_TOKENS", raising=False)

    assert chat_memory._chat_max_tokens() == 2400


def test_chat_continues_until_finish_reason_is_not_max_tokens(monkeypatch):
    monkeypatch.setenv("MEBUKI_CHAT_CONTINUATION_ROUNDS", "3")
    post = Mock(
        side_effect=[
            _response("前半。", "MAX_TOKENS"),
            _response("中盤。", "MAX_TOKENS"),
            _response("結論。", "STOP"),
        ]
    )
    monkeypatch.setattr(chat_memory.requests, "post", post)
    monkeypatch.setattr(chat_memory, "_get_gemini_api_key", lambda: "test-key")

    result = chat_memory.call_gemini_chat("system", [], "question")

    assert result == "前半。\n\n中盤。\n\n結論。"
    assert post.call_count == 3


def test_tool_chat_continues_final_text_after_tool_round(monkeypatch):
    monkeypatch.setenv("MEBUKI_CHAT_CONTINUATION_ROUNDS", "2")
    post = Mock(
        side_effect=[
            _response("紫苑の長い回答前半。", "MAX_TOKENS"),
            _response("回答の結論。", "STOP"),
        ]
    )
    monkeypatch.setattr(chat_memory.requests, "post", post)
    monkeypatch.setattr(chat_memory, "_get_gemini_api_key", lambda: "test-key")

    result = chat_memory.call_gemini_with_tools(
        "system",
        [],
        "question",
        [],
        lambda _name, _args: {},
    )

    assert result == "紫苑の長い回答前半。\n\n回答の結論。"
    assert post.call_count == 2


def test_continuation_round_limit_adds_clear_notice(monkeypatch):
    monkeypatch.setenv("MEBUKI_CHAT_CONTINUATION_ROUNDS", "1")
    post = Mock(
        side_effect=[
            _response("前半。", "MAX_TOKENS"),
            _response("まだ途中。", "MAX_TOKENS"),
        ]
    )
    monkeypatch.setattr(chat_memory.requests, "post", post)
    monkeypatch.setattr(chat_memory, "_get_gemini_api_key", lambda: "test-key")

    result = chat_memory.call_gemini_chat("system", [], "question")

    assert "前半。" in result
    assert "まだ途中。" in result
    assert "回答が非常に長いため" in result
