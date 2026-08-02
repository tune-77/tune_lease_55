from api.chat_context_builder import ChatContextBuilderDeps, build_chat_context_state


def _deps(*, category: str = "general", basic_context: str = "") -> ChatContextBuilderDeps:
    return ChatContextBuilderDeps(
        classify_question=lambda _message: category,
        build_basic_lease_question_context=lambda _message: basic_context,
        chat_context_mode=lambda _message, _category, long_input=False: "screening" if _category != "general" else "casual",
        chat_context_budget=lambda mode: {
            "history_limit": 5,
            "history_messages": 5,
            "history_chars_per_message": 200,
            "history_total_budget": 1000,
            "rag_top_k": 3,
            "recall_limit": 3,
            "use_news": mode != "casual",
            "use_obsidian_daily": mode != "casual",
            "use_experience_loop": False,
        },
        chat_mode_instruction=lambda mode: f"[mode:{mode}]",
        response_mode_instruction=lambda response_mode: f"[response:{response_mode}]",
        build_identity_memory=lambda: ("identity", {"block": "identity", "refs": ["i"], "layers": {"identity": True}}),
        build_user_personal_memory=lambda: ("user-memory", {"block": "user-memory", "refs": ["u"], "line_count": 1}),
        build_continuity_hook=lambda _message: ("hook", {"used": True, "route": "lease"}),
        build_consciousness_ux=lambda: "consciousness",
        build_shion_specificity=lambda _message: "specificity",
        build_shion_light_tone=lambda _message: "light-tone",
        build_shion_non_domain=lambda _message: "non-domain",
        build_human_device_resonance=lambda _message, user_id="", now="": f"human-device:{user_id}:{now}",
        build_judgment_response_shape=lambda _message: "judgment-shape",
        build_grey_judgment=lambda _message: ("grey", {"used": True}),
    )


def test_context_builder_promotes_basic_lease_question_from_general():
    state = build_chat_context_state(
        message="ファイナンスリースとは？",
        response_mode="shion",
        user_id="u1",
        now="2026-08-02 10:00",
        long_input=False,
        news_focus_context="news",
        news_brief_context="brief",
        news_actions_context="actions",
        obsidian_daily_context="daily",
        deps=_deps(category="general", basic_context="basic lease"),
    )

    assert state.question_category == "lease_knowledge"
    assert state.context_mode == "screening"
    assert state.basic_lease_question_context == "basic lease"
    assert state.grey_judgment_payload == {"used": True}


def test_context_builder_general_response_mode_suppresses_shion_memory_blocks():
    state = build_chat_context_state(
        message="おはよう",
        response_mode="general",
        user_id="u1",
        now="2026-08-02 10:00",
        long_input=False,
        news_focus_context="news",
        news_brief_context="brief",
        news_actions_context="actions",
        obsidian_daily_context="daily",
        deps=_deps(category="general"),
    )

    assert state.is_general_response_mode is True
    assert state.identity_memory_context == ""
    assert state.identity_memory_payload["suppressed_by_response_mode"] == "general"
    assert state.user_personal_memory_context == ""
    assert state.continuity_hook_context == ""
    assert state.consciousness_ux_context == ""
    assert state.experience_loop_payload["suppressed_by_response_mode"] == "general"
    assert state.grey_judgment_payload["suppressed_by_response_mode"] == "general"


def test_context_builder_budget_can_disable_news_and_daily_context():
    state = build_chat_context_state(
        message="雑談",
        response_mode="shion",
        user_id="u1",
        now="2026-08-02 10:00",
        long_input=False,
        news_focus_context="news",
        news_brief_context="brief",
        news_actions_context="actions",
        obsidian_daily_context="daily",
        deps=_deps(category="general"),
    )

    assert state.context_mode == "casual"
    assert state.news_focus_context == ""
    assert state.news_brief_context == ""
    assert state.news_actions_context == ""
    assert state.obsidian_daily_context == ""
