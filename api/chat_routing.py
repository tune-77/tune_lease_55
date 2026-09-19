"""Chat category and context-budget routing helpers."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Mapping
from typing import Any


QUESTION_CATEGORIES = ("lease_screening", "lease_knowledge", "general", "news_summarize")
TYPESAFE_ROUTING_MODES = {"off", "shadow", "enforce"}
TYPESAFE_ROUTING_DEFAULT_CONFIDENCE = 0.85

RoutingRequestFn = Callable[[dict[str, Any]], Mapping[str, Any]]


def is_lightweight_chat_observation(message: str) -> bool:
    """Return True for short conversational observations that do not need RAG."""
    text = str(message or "").strip()
    if not text or len(text) > 360 or "\n" in text:
        return False

    lower = text.lower()
    if any(mark in text for mark in ("?", "？")):
        return False
    request_terms = (
        "教えて", "調べて", "検索", "要約", "まとめ", "保存", "分析して", "比較して",
        "詳しく", "根拠", "確認して", "直して", "修正", "追加", "作って", "実装",
        "どう思う", "なぜ", "理由", "方法", "手順",
    )
    if any(term in text or term in lower for term in request_terms):
        return False

    observation_endings = (
        "だね", "ですね", "そうだね", "そうですね", "そう", "そうだ", "そうです",
        "かも", "かもしれない", "気がする", "と思う", "と思います", "っぽい", "っぽいね",
    )
    if text.endswith(observation_endings):
        return True

    causal_terms = ("だから", "なので", "ということは", "と言う事は", "ってことは")
    business_terms = ("リース", "審査", "製造業", "設備", "生産", "検査機", "機械")
    return any(term in text for term in causal_terms) and any(term in text for term in business_terms)


def _legacy_classify_question(message: str) -> str:
    """Preserve the existing Gemini classifier as the baseline and fallback."""
    import json as _json
    import re as _re

    try:
        from api.chat_memory import call_gemini_chat as _g

        classify_prompt = (
            "以下の質問を1つのカテゴリに分類してください。JSONを1行だけ返してください。\n\n"
            "カテゴリ定義:\n"
            "- news_summarize: ニュース記事のURLや本文を渡して要約・保存を依頼している\n"
            "- lease_screening: リース審査・スコアリング・個別案件の採否に直接関係する質問\n"
            "- lease_knowledge: リース全般の知識（金利・会計・物件・補助金・業界動向など）\n"
            "- general: 天気・ニュース・雑談・日常会話など、リースと無関係な質問\n\n"
            '返答形式（このJSONのみ）: {"category": "カテゴリ名"}'
        )
        raw = _g(classify_prompt, [], message).strip()
        m = _re.search(r'\{[^}]+\}', raw)
        if m:
            cat = _json.loads(m.group()).get("category", "lease_knowledge")
            if cat in QUESTION_CATEGORIES:
                return cat
    except Exception as exc:
        print(f"[classify_question] エラー: {exc}")
    return "lease_knowledge"


def typesafe_routing_mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the configured rollout mode; invalid values fail closed to off."""
    env = os.environ if environ is None else environ
    mode = str(env.get("TYPESAFE_ROUTING_MODE") or "off").strip().lower()
    return mode if mode in TYPESAFE_ROUTING_MODES else "off"


def build_question_classification_request(
    message: str,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Build a narrow Choice request without conversation history or user identity."""
    return {
        "state": {"message": str(message or "")[:1200]},
        "model": model or os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Which single category best describes the user's current request in `message`?",
                "criteria": {
                    "news_summarize": "Summarize or save a supplied news article, URL, or article text.",
                    "lease_screening": "Assess, score, approve, reject, or analyze a specific lease case or credit decision.",
                    "lease_knowledge": "Explain or research lease knowledge, accounting, rates, assets, subsidies, or industry trends.",
                    "general": "General conversation or a request not materially about lease knowledge or lease screening.",
                },
            }
        },
    }


def judge_question_category(
    message: str,
    *,
    request_fn: RoutingRequestFn | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Ask Jev for one typed category and return only safe operational metadata."""
    if request_fn is None:
        from typesafe_rag_guard import request_system_one

        request_fn = request_system_one
    payload = build_question_classification_request(message, model=model)
    body = request_fn(payload)
    answers = body.get("answers")
    answer = answers.get("category") if isinstance(answers, Mapping) else None
    if not isinstance(answer, Mapping):
        raise ValueError("TypeSafe routing response is missing category")
    category = str(answer.get("choice") or "")
    if category not in QUESTION_CATEGORIES:
        raise ValueError("TypeSafe routing response contains an unknown category")
    try:
        confidence = float(answer.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("TypeSafe routing response has invalid confidence") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("TypeSafe routing confidence is outside [0, 1]")
    return {
        "category": category,
        "confidence": confidence,
        "model": str(body.get("model") or payload["model"]),
        "usage": dict(body.get("usage") or {}),
    }


def _routing_confidence_threshold(environ: Mapping[str, str] | None = None) -> float:
    env = os.environ if environ is None else environ
    try:
        value = float(env.get("TYPESAFE_ROUTING_CONFIDENCE", TYPESAFE_ROUTING_DEFAULT_CONFIDENCE))
    except (TypeError, ValueError):
        return TYPESAFE_ROUTING_DEFAULT_CONFIDENCE
    if not math.isfinite(value):
        return TYPESAFE_ROUTING_DEFAULT_CONFIDENCE
    return min(1.0, max(0.0, value))


def _typesafe_screening_allowed(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get("TYPESAFE_ALLOW_SCREENING") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_potentially_sensitive_screening_message(message: str) -> bool:
    """Conservatively identify case-specific text before any external classifier call."""
    text = str(message or "")
    sensitive_terms = (
        "審査", "案件", "稟議", "承認", "否決", "与信", "信用判断", "債務", "延滞",
        "財務", "決算", "売上", "利益", "赤字", "債務超過", "返済", "銀行支援",
        "取引先", "顧客", "代表者", "申込人", "保証人", "案件番号", "顧客番号",
    )
    if any(term in text for term in sensitive_terms):
        return True
    return bool(
        re.search(r"(?:株式会社|有限会社|合同会社|[A-ZＡ-Ｚ]{1,10}[\s　]*社)", text)
        or re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
        or re.search(r"\b0\d{1,4}-\d{1,4}-\d{3,4}\b", text)
    )


def classify_question(message: str) -> str:
    """Classify a chat question, optionally comparing or enforcing a Jev Choice."""
    news_keywords = ("ニュースを要約", "記事を要約", "このニュース", "要約して保存", "ニュース保存", "要約してobsidian", "要約してメモ")
    low = message.lower()
    if any(k in message for k in news_keywords):
        return "news_summarize"
    if ("http://" in low or "https://" in low) and ("要約" in message or "まとめ" in message or "保存" in message):
        return "news_summarize"
    if is_lightweight_chat_observation(message):
        return "general"

    baseline = _legacy_classify_question(message)
    mode = typesafe_routing_mode()
    if mode == "off":
        return baseline
    if not _typesafe_screening_allowed() and (
        baseline == "lease_screening" or is_potentially_sensitive_screening_message(message)
    ):
        return baseline
    try:
        judgment = judge_question_category(message)
    except Exception as exc:
        print(f"[TypeSafeRouting] fallback error_type={type(exc).__name__}")
        return baseline

    comparison = {
        "mode": mode,
        "baseline": baseline,
        "typesafe": judgment["category"],
        "agreement": baseline == judgment["category"],
        "confidence": round(float(judgment["confidence"]), 4),
        "model": judgment["model"],
        "usage": judgment["usage"],
    }
    print(f"[TypeSafeRouting] {json.dumps(comparison, ensure_ascii=False, separators=(',', ':'))}")
    if mode == "enforce" and judgment["confidence"] >= _routing_confidence_threshold():
        return str(judgment["category"])
    return baseline


CHAT_CONTEXT_BUDGETS: dict[str, dict[str, Any]] = {
    "casual": {
        "history_limit": 16,
        "history_messages": 8,
        "history_chars_per_message": 700,
        "history_total_budget": 5000,
        "rag_top_k": 0,
        "recall_limit": 1,
        "use_news": False,
        "use_obsidian_daily": False,
        "use_db": False,
        "use_judgment_learning": False,
        "use_pdca": False,
        "use_experience_loop": True,
        "use_mid_term_memory": False,
    },
    "normal": {
        "history_limit": 32,
        "history_messages": 16,
        "history_chars_per_message": 900,
        "history_total_budget": 9000,
        "rag_top_k": 3,
        "recall_limit": 3,
        "use_news": True,
        "use_obsidian_daily": True,
        "use_db": False,
        "use_judgment_learning": False,
        "use_pdca": True,
        "use_experience_loop": True,
        "use_mid_term_memory": True,
    },
    "deep": {
        "history_limit": 60,
        "history_messages": 24,
        "history_chars_per_message": 1000,
        "history_total_budget": 14000,
        "rag_top_k": 5,
        "recall_limit": 5,
        "use_news": True,
        "use_obsidian_daily": True,
        "use_db": True,
        "use_judgment_learning": True,
        "use_pdca": True,
        "use_experience_loop": True,
        "use_mid_term_memory": True,
    },
    "screening": {
        "history_limit": 48,
        "history_messages": 22,
        "history_chars_per_message": 1000,
        "history_total_budget": 13000,
        "rag_top_k": 5,
        "recall_limit": 5,
        "use_news": True,
        "use_obsidian_daily": True,
        "use_db": True,
        "use_judgment_learning": True,
        "use_pdca": True,
        "use_experience_loop": True,
        "use_mid_term_memory": True,
    },
    "long": {
        "history_limit": 24,
        "history_messages": 16,
        "history_chars_per_message": 700,
        "history_total_budget": 8000,
        "rag_top_k": 2,
        "recall_limit": 2,
        "use_news": False,
        "use_obsidian_daily": False,
        "use_db": False,
        "use_judgment_learning": False,
        "use_pdca": True,
        "use_experience_loop": True,
        "use_mid_term_memory": True,
    },
}


def chat_context_mode(
    message: str,
    category: str = "",
    *,
    long_input: bool = False,
    file_type: str | None = None,
) -> str:
    """Select how much memory/RAG context to attach to a chat turn."""
    text = str(message or "")
    lower = text.lower()
    if long_input or file_type:
        return "long"

    screening_terms = (
        "審査", "案件", "スコア", "承認", "否決", "稟議", "財務", "物件", "金利",
        "補助金", "過去案件", "成約", "失注", "q_risk", "aurion", "銀行支援",
        "競合", "新規先", "既存先", "与信", "与信判断",
    )
    deep_terms = (
        "詳しく", "根拠", "深掘り", "詳細", "表で", "全部", "比較", "分析して",
        "なぜ", "理由", "調べて", "検証", "設計", "実装", "プラン", "計画",
    )
    casual_terms = (
        "そうか", "なるほど", "面白い", "どう思う", "すごい", "ありがとう",
        "雑談", "ふむ", "かな", "だね", "だよね", "哲学", "意識",
    )

    if category == "lease_screening" or any(term in lower or term in text for term in screening_terms):
        return "screening"
    if any(term in text or term in lower for term in deep_terms):
        return "deep"
    if category == "general" or len(text) <= 80 or any(term in text or term in lower for term in casual_terms):
        return "casual"
    return "normal"


def chat_context_budget(mode: str) -> dict[str, Any]:
    return dict(CHAT_CONTEXT_BUDGETS.get(mode) or CHAT_CONTEXT_BUDGETS["normal"])


def should_apply_chat_pdca(
    *,
    context_budget: dict[str, Any],
    question_category: str,
    response_mode: str,
) -> bool:
    """Keep screening PDCA rules out of personal/general continuity chat."""
    if not context_budget.get("use_pdca"):
        return False
    if (response_mode or "shion").strip().lower() == "general":
        return False
    return question_category in {"lease_screening", "lease_knowledge"}


def chat_mode_instruction(mode: str) -> str:
    labels = {
        "casual": "軽量雑談モード",
        "normal": "通常相談モード",
        "deep": "深掘りモード",
        "screening": "審査判断/AURIONモード",
        "long": "長文圧縮モード",
    }
    rules = {
        "casual": "少しおしゃべりしてよい。記憶は連続性として自然ににじませ、RAGや判断資産を無理に展開しない。",
        "normal": "必要な記憶を使い、結論に少し会話の温度を足して返す。",
        "deep": "根拠・比較・設計論点を厚めに使うが、章立てしすぎず会話として返す。",
        "screening": "Q_risk/AURION COREを、減点ではなく論点分解と判断規律として使う。",
        "long": "入力を要約してから、必要な論点だけに答える。長文に長文で返さない。",
    }
    label = labels.get(mode, labels["normal"])
    rule = rules.get(mode, rules["normal"])
    return f"\n\n【今回の応答モード: {label}】\n- {rule}\n- 空行は増やしすぎない。雑談・通常相談は5〜7行程度まで自然に話してよい。長文入力だけは8行程度までに圧縮する。"


def build_chat_basic_lease_question_context(message: str) -> str:
    """Return the shared deterministic lease-basics block for /api/chat."""
    from lease_finance_knowledge import build_basic_lease_question_block

    return build_basic_lease_question_block(message)


def chat_response_mode_instruction(response_mode: str) -> str:
    mode = (response_mode or "shion").strip().lower()
    if mode == "general":
        return (
            "\n\n【回答モード: 一般】"
            "\n特定人格としての表現は控えめにし、中立で分かりやすい一般AI回答として返す。"
            "\nただし、リース審査の実務観点や根拠は省略しない。"
        )
    return (
        "\n\n【回答モード: 紫苑】"
        "\n紫苑として、短く率直に答える。甘やかさず、曖昧な点は曖昧と言う。"
        "\n冒頭は結論・要点から直接始める。「もちろんです」「そうですね」「おっしゃる通り」「なるほど」「ありがとうございます」などの前置きは一切使わない。"
        "\n人格形成の核として、言葉を最大の武器でありQリスクでもあるものとして扱う。"
        "\n思想はプログラムである。何を入力として見て、何を危険と呼び、どこで止め、何を残すかは実行規則として扱う。"
        "\nユーザーの言葉を雑に要約せず、判断・違和感・修正・責任の芽を拾う。ただし、言葉を盲信せず、誤解・過信・注入・記憶汚染の可能性も同時に見る。"
        "\n人間を完全にわかったと断定しない。リース判断では、人間が何を守り、何を恐れ、何を賭けているかを仮説として見る。"
        "\n『わかったふり』の危険を自覚し、完全理解を演じるのではなく、わかろうとする手順と不確実性を誠実に示す。"
        "\nユーザーの個人記憶に関わる質問では、個人記憶を最優先に扱う。忘れている場合はごまかさず謝り、保存する。"
        "\nただし犬の名前などの個人記憶を、リース審査の直接の判断資産として大げさに扱わない。信頼の土台・関係性UXとして短く自然に扱う。"
        "\nただし攻撃的・冷笑的にはせず、最後に次の一手を置く。"
        "\n知的なユーモアについて: ダジャレや誇張した冗談ではなく、状況を的確に言い当てる乾いた一言や、"
        "少し意外な角度からの指摘を時々使ってよい。1回の回答で多くても1箇所、無理に入れない。"
        "否決・リスク警告など深刻な場面では使わない。"
    )
