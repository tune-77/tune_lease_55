from api.chat_external_research import (
    build_external_research_suggestion,
    external_research_permission_reply,
    external_research_topic_from_message,
)


def test_external_research_topic_strips_command_words():
    topic = external_research_topic_from_message("ネットで調べてから、補助金付き工作機械リースの注意点")

    assert topic == "から、補助金付き工作機械リースの注意点"


def test_external_research_suggestion_requires_explicit_research_request():
    assert build_external_research_suggestion(
        "補助金付き工作機械リースの注意点を教えて",
        question_category="lease_knowledge",
        knowledge_ref_count=0,
        response_mode="shion",
    ) == {"needed": False}

    suggestion = build_external_research_suggestion(
        "ネットで調べて補助金付き工作機械リースの注意点を教えて",
        question_category="lease_knowledge",
        knowledge_ref_count=0,
        response_mode="shion",
    )

    assert suggestion == {
        "needed": True,
        "topic": "補助金付き工作機械リースの注意点を教えて",
        "reason": "ユーザーが外部調査を求めています / Obsidian/RAGの参照が薄い論点です",
        "output_dir": "Projects/tune_lease_55/Research/Auto Research/",
    }


def test_external_research_suggestion_disabled_for_general_mode():
    suggestion = build_external_research_suggestion(
        "ネットで調べて補助金の期限を確認して",
        question_category="lease_knowledge",
        knowledge_ref_count=0,
        response_mode="general",
    )

    assert suggestion == {"needed": False}


def test_external_research_permission_reply_keeps_consent_wording():
    reply = external_research_permission_reply(
        {"topic": "補助金期限", "reason": "ユーザーが外部調査を求めています"}
    )

    assert "ここは手元の記憶だけで断定しない方がいいです" in reply
    assert "- 足りない情報: 補助金期限" in reply
    assert "- 理由: ユーザーが外部調査を求めています" in reply
    assert "ObsidianのResearchノートに保存してから回答に使っていいですか？" in reply
