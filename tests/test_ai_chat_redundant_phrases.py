from ai_chat import _remove_redundant_phrases


def test_remove_redundant_phrase_requires_delimiter_after_hai():
    assert _remove_redundant_phrases("はいけません。これは禁止です。") == "はいけません。これは禁止です。"


def test_remove_redundant_phrase_strips_standalone_acknowledgement():
    assert _remove_redundant_phrases("はい。\n確認します。") == "確認します。"


def test_remove_redundant_phrase_strips_acknowledgement_with_delimiter():
    assert _remove_redundant_phrases("はい、確認します。") == "確認します。"

