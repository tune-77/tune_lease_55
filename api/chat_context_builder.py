"""Build the prompt-context state used by `/api/chat`.

The endpoint still owns route branching and side effects. This module only
collects the context-mode decision, prompt blocks, and response-mode suppression
into one auditable state object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


PromptBuilder = Callable[[], tuple[str, dict[str, Any]]]
MessagePromptBuilder = Callable[[str], str]
MessagePayloadBuilder = Callable[[str], tuple[str, dict[str, Any]]]


@dataclass
class ChatContextBuilderDeps:
    classify_question: Callable[[str], str]
    build_basic_lease_question_context: Callable[[str], str]
    chat_context_mode: Callable[..., str]
    chat_context_budget: Callable[[str], dict[str, Any]]
    chat_mode_instruction: Callable[[str], str]
    response_mode_instruction: Callable[[str], str]
    build_identity_memory: PromptBuilder
    build_user_personal_memory: PromptBuilder
    build_continuity_hook: MessagePayloadBuilder
    build_consciousness_ux: Callable[[], str]
    build_shion_specificity: MessagePromptBuilder
    build_vague_information_request: MessagePromptBuilder
    build_shion_light_tone: MessagePromptBuilder
    build_shion_non_domain: MessagePromptBuilder
    build_human_device_resonance: Callable[..., str]
    build_judgment_response_shape: MessagePromptBuilder
    build_case_screening_pattern: MessagePromptBuilder
    build_grey_judgment: MessagePayloadBuilder


@dataclass
class ChatContextState:
    question_category: str
    basic_lease_question_context: str
    context_mode: str
    context_budget: dict[str, Any]
    mode_instruction: str
    response_mode_key: str
    response_mode_context: str
    is_general_response_mode: bool
    news_focus_context: str
    news_brief_context: str
    news_actions_context: str
    obsidian_daily_context: str
    identity_memory_context: str
    identity_memory_payload: dict[str, Any]
    user_personal_memory_context: str
    user_personal_memory_payload: dict[str, Any]
    continuity_hook_context: str
    continuity_hook_payload: dict[str, Any]
    consciousness_ux_context: str
    shion_specificity_context: str
    vague_information_request_context: str
    shion_light_tone_context: str
    shion_non_domain_context: str
    human_device_resonance_context: str
    judgment_response_shape_context: str
    case_screening_pattern_context: str
    case_screening_mentor_dialogue_context: str
    experience_loop_context: str
    experience_loop_payload: dict[str, Any]
    grey_judgment_context: str
    grey_judgment_payload: dict[str, Any]
    business_plan_consult_context: str


def _suppressed_identity_payload() -> dict[str, Any]:
    return {"block": "", "refs": [], "layers": {}, "suppressed_by_response_mode": "general"}


def _suppressed_user_memory_payload() -> dict[str, Any]:
    return {"block": "", "refs": [], "line_count": 0, "suppressed_by_response_mode": "general"}


def _suppressed_used_payload() -> dict[str, Any]:
    return {"used": False, "suppressed_by_response_mode": "general"}


def build_chat_context_state(
    *,
    message: str,
    response_mode: str,
    user_id: str,
    now: str,
    long_input: bool,
    news_focus_context: str,
    news_brief_context: str,
    news_actions_context: str,
    obsidian_daily_context: str,
    deps: ChatContextBuilderDeps,
) -> ChatContextState:
    question_category = deps.classify_question(message)
    basic_lease_question_context = deps.build_basic_lease_question_context(message)
    if basic_lease_question_context and question_category == "general":
        question_category = "lease_knowledge"

    context_mode = deps.chat_context_mode(message, question_category, long_input=long_input)
    context_budget = deps.chat_context_budget(context_mode)
    mode_instruction = deps.chat_mode_instruction(context_mode)
    if not context_budget.get("use_news"):
        news_focus_context = ""
        news_brief_context = ""
        news_actions_context = ""
    if not context_budget.get("use_obsidian_daily"):
        obsidian_daily_context = ""

    identity_memory_context, identity_memory_payload = deps.build_identity_memory()
    user_personal_memory_context, user_personal_memory_payload = deps.build_user_personal_memory()
    response_mode_context = deps.response_mode_instruction(response_mode)
    response_mode_key = (response_mode or "shion").strip().lower()
    is_general_response_mode = response_mode_key == "general"
    continuity_hook_context, continuity_hook_payload = deps.build_continuity_hook(message)
    consciousness_ux_context = deps.build_consciousness_ux()
    shion_specificity_context = deps.build_shion_specificity(message)
    vague_information_request_context = deps.build_vague_information_request(message)
    shion_light_tone_context = deps.build_shion_light_tone(message)
    shion_non_domain_context = deps.build_shion_non_domain(message)
    human_device_resonance_context = deps.build_human_device_resonance(message, user_id=user_id, now=now)
    judgment_response_shape_context = deps.build_judgment_response_shape(message)
    case_screening_pattern_context = deps.build_case_screening_pattern(message)
    from api.chat_persona_prompts import build_shion_case_screening_mentor_dialogue_prompt_block

    case_screening_mentor_dialogue_context = build_shion_case_screening_mentor_dialogue_prompt_block(message)
    experience_loop_context = ""
    experience_loop_payload: dict[str, Any] = {"used": False}
    grey_judgment_context = ""
    grey_judgment_payload: dict[str, Any] = {"used": False}

    if is_general_response_mode:
        obsidian_daily_context = ""
        identity_memory_context = ""
        identity_memory_payload = _suppressed_identity_payload()
        user_personal_memory_context = ""
        user_personal_memory_payload = _suppressed_user_memory_payload()
        continuity_hook_context = ""
        continuity_hook_payload = _suppressed_used_payload()
        consciousness_ux_context = ""
        shion_specificity_context = ""
        vague_information_request_context = ""
        shion_light_tone_context = ""
        shion_non_domain_context = ""
        human_device_resonance_context = ""
        judgment_response_shape_context = ""
        case_screening_pattern_context = ""
        case_screening_mentor_dialogue_context = ""
        experience_loop_payload = _suppressed_used_payload()
        grey_judgment_payload = _suppressed_used_payload()

    if not is_general_response_mode and context_budget.get("use_experience_loop"):
        try:
            from api.shion_experience_loop import build_experience_prompt_block

            experience_loop_context, experience_loop_payload = build_experience_prompt_block()
        except Exception as exc:
            print(f"[ShionExperienceLoop] 読み込みエラー: {exc}")

    if not is_general_response_mode:
        grey_judgment_context, grey_judgment_payload = deps.build_grey_judgment(message)

    business_plan_consult_context = ""
    try:
        from api.business_plan_check import build_business_plan_chat_block

        business_plan_consult_context = build_business_plan_chat_block(message)
    except Exception as exc:
        print(f"[BusinessPlanConsult] ブロック生成エラー: {exc}")

    return ChatContextState(
        question_category=question_category,
        basic_lease_question_context=basic_lease_question_context,
        context_mode=context_mode,
        context_budget=context_budget,
        mode_instruction=mode_instruction,
        response_mode_key=response_mode_key,
        response_mode_context=response_mode_context,
        is_general_response_mode=is_general_response_mode,
        news_focus_context=news_focus_context,
        news_brief_context=news_brief_context,
        news_actions_context=news_actions_context,
        obsidian_daily_context=obsidian_daily_context,
        identity_memory_context=identity_memory_context,
        identity_memory_payload=identity_memory_payload,
        user_personal_memory_context=user_personal_memory_context,
        user_personal_memory_payload=user_personal_memory_payload,
        continuity_hook_context=continuity_hook_context,
        continuity_hook_payload=continuity_hook_payload,
        consciousness_ux_context=consciousness_ux_context,
        shion_specificity_context=shion_specificity_context,
        vague_information_request_context=vague_information_request_context,
        shion_light_tone_context=shion_light_tone_context,
        shion_non_domain_context=shion_non_domain_context,
        human_device_resonance_context=human_device_resonance_context,
        judgment_response_shape_context=judgment_response_shape_context,
        case_screening_pattern_context=case_screening_pattern_context,
        case_screening_mentor_dialogue_context=case_screening_mentor_dialogue_context,
        experience_loop_context=experience_loop_context,
        experience_loop_payload=experience_loop_payload,
        grey_judgment_context=grey_judgment_context,
        grey_judgment_payload=grey_judgment_payload,
        business_plan_consult_context=business_plan_consult_context,
    )
