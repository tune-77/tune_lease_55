import api.chat_routing as chat_routing
from api.chat_routing import (
    build_question_classification_request,
    build_chat_basic_lease_question_context,
    chat_context_budget,
    chat_context_mode,
    chat_mode_instruction,
    chat_response_mode_instruction,
    classify_question,
    is_lightweight_chat_observation,
    judge_question_category,
    should_apply_chat_pdca,
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


def test_typesafe_choice_request_contains_only_current_message():
    payload = build_question_classification_request("この案件を審査して", model="jev-test")

    assert payload["state"] == {"message": "この案件を審査して"}
    assert payload["model"] == "jev-test"
    assert set(payload["questions"]["category"]["criteria"]) == {
        "lease_screening",
        "lease_knowledge",
        "general",
        "news_summarize",
    }


def test_judge_question_category_reads_typed_choice():
    judgment = judge_question_category(
        "この案件を審査して",
        request_fn=lambda _payload: {
            "model": "jev-test",
            "answers": {
                "category": {
                    "type": "choice",
                    "choice": "lease_screening",
                    "confidence": 0.93,
                }
            },
            "usage": {"input_tokens": 12, "output_tokens": 3},
        },
    )

    assert judgment == {
        "category": "lease_screening",
        "confidence": 0.93,
        "model": "jev-test",
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }


def test_typesafe_shadow_never_changes_baseline(monkeypatch):
    monkeypatch.setenv("TYPESAFE_ROUTING_MODE", "shadow")
    monkeypatch.setattr(chat_routing, "_legacy_classify_question", lambda _message: "general")
    monkeypatch.setattr(
        chat_routing,
        "judge_question_category",
        lambda _message: {
            "category": "lease_knowledge",
            "confidence": 0.99,
            "model": "jev-test",
            "usage": {},
        },
    )

    assert chat_routing.classify_question("リースを説明して") == "general"


def test_typesafe_enforce_requires_configured_confidence(monkeypatch):
    monkeypatch.setenv("TYPESAFE_ROUTING_MODE", "enforce")
    monkeypatch.setenv("TYPESAFE_ROUTING_CONFIDENCE", "0.90")
    monkeypatch.setattr(chat_routing, "_legacy_classify_question", lambda _message: "lease_knowledge")
    monkeypatch.setattr(
        chat_routing,
        "judge_question_category",
        lambda _message: {
            "category": "lease_screening",
            "confidence": 0.91,
            "model": "jev-test",
            "usage": {},
        },
    )

    assert chat_routing.classify_question("案件を見て") == "lease_screening"


def test_typesafe_does_not_receive_screening_questions_by_default(monkeypatch):
    monkeypatch.setenv("TYPESAFE_ROUTING_MODE", "shadow")
    monkeypatch.delenv("TYPESAFE_ALLOW_SCREENING", raising=False)
    monkeypatch.setattr(chat_routing, "_legacy_classify_question", lambda _message: "lease_screening")
    monkeypatch.setattr(
        chat_routing,
        "judge_question_category",
        lambda _message: (_ for _ in ()).throw(AssertionError("must not call TypeSafe")),
    )

    assert chat_routing.classify_question("A社の案件を審査して") == "lease_screening"


def test_context_mode_and_budget_stable_shapes():
    assert chat_context_mode("案件の審査スコアを見て", "lease_screening") == "screening"
    assert chat_context_mode("詳しく根拠も教えて", "lease_knowledge") == "deep"
    assert chat_context_mode("ありがとう", "general") == "casual"
    assert chat_context_mode("長文", "lease_knowledge", long_input=True) == "long"

    screening_budget = chat_context_budget("screening")
    assert screening_budget["rag_top_k"] == 5
    assert screening_budget["use_db"] is True
    assert chat_context_budget("unknown")["history_limit"] == 32


def test_pdca_routing_excludes_general_and_disabled_budget():
    assert should_apply_chat_pdca(
        context_budget={"use_pdca": True},
        question_category="lease_knowledge",
        response_mode="shion",
    ) is True
    assert should_apply_chat_pdca(
        context_budget={"use_pdca": True},
        question_category="lease_knowledge",
        response_mode="general",
    ) is False
    assert should_apply_chat_pdca(
        context_budget={"use_pdca": False},
        question_category="lease_screening",
        response_mode="shion",
    ) is False


def test_chat_mode_instruction_keeps_known_labels_and_fallback():
    assert "審査判断/AURIONモード" in chat_mode_instruction("screening")
    assert "通常相談モード" in chat_mode_instruction("unknown")


def test_chat_response_mode_instruction_keeps_general_and_shion_modes():
    general = chat_response_mode_instruction("general")
    shion = chat_response_mode_instruction("shion")

    assert "【回答モード: 一般】" in general
    assert "中立で分かりやすい一般AI回答" in general
    assert "【回答モード: 紫苑】" in shion
    assert "言葉を最大の武器でありQリスクでもある" in shion


def test_basic_lease_question_context_uses_shared_knowledge_block():
    block = build_chat_basic_lease_question_context("ファイナンスリースとは？")

    assert "基本リースQA" in block
    assert "ファイナンス・リース" in block
