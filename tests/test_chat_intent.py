from chat_intent import build_chat_guidance, is_out_of_scope


def test_non_domain_greeting_is_not_out_of_scope():
    assert is_out_of_scope("おはよう") is False
    assert is_out_of_scope("元気かい？") is False
    assert is_out_of_scope("今何時？") is False


def test_unrelated_short_chatter_still_out_of_scope():
    assert is_out_of_scope("ふーん") is True


def test_lease_related_short_message_still_in_scope():
    assert is_out_of_scope("案件どう？") is False


def test_greeting_guidance_has_no_out_of_scope_suffix():
    guidance = build_chat_guidance("おはよう")
    assert guidance.out_of_scope is False
    assert "審査外の質問" not in guidance.prompt_suffix
