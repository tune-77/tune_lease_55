from api.dialogue_history_utils import (
    clip_dialogue_history_text,
    compact_dialogue_history,
    is_long_dialogue_input,
)


def test_is_long_dialogue_input_detects_text_lines_and_files():
    assert is_long_dialogue_input("x" * 1800)
    assert is_long_dialogue_input("\n".join(["x"] * 19))
    assert is_long_dialogue_input("short", file_type="pdf")
    assert not is_long_dialogue_input("short")


def test_clip_dialogue_history_text_keeps_head_and_tail():
    clipped = clip_dialogue_history_text("abcdefghij", 6)

    assert clipped.startswith("abc")
    assert clipped.endswith("hij")
    assert "中略" in clipped


def test_compact_dialogue_history_bounds_messages_and_preserves_extra_keys():
    history = [
        {"role": "user", "content": "a" * 20, "id": "old"},
        {"role": "assistant", "content": "b" * 20, "id": "mid"},
        {"role": "user", "content": "c" * 20, "id": "new"},
    ]

    compacted = compact_dialogue_history(history, max_messages=2, max_chars_per_message=8, total_budget=120)

    assert [item["id"] for item in compacted] == ["mid", "new"]
    assert all("中略" in item["content"] for item in compacted)
    assert compacted[0]["role"] == "assistant"
