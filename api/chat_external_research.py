"""External research consent and context helpers for chat."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

from api.routers.vault_hub import _research_organ_vault_path, _research_run_display
from runtime_paths import get_data_dir


NO_DATA_RESEARCH_TERMS = (
    "最新",
    "ニュース",
    "記事",
    "要約",
    "期限",
    "制度",
    "補助金",
    "金利",
    "政策",
    "動向",
    "市場",
    "統計",
    "大臣",
    "総裁",
    "日銀",
    "政府",
    "会見",
    "調べて",
    "検索",
    "リサーチ",
)


def external_research_topic_from_message(message: str) -> str:
    text = re.sub(r"\s+", " ", (message or "").strip())
    text = re.sub(
        r"(ネットで|Webで|ウェブで|外部調査で|調査器官で|調べて|調べてから|検索して|リサーチして|回答に使って)",
        "",
        text,
    ).strip(" 。、")
    if not text:
        text = (message or "").strip()
    return text[:120]


def _should_offer_research_for_no_data(message: str, *, question_category: str, knowledge_ref_count: int) -> bool:
    if knowledge_ref_count > 0:
        return False
    if question_category not in {"lease_screening", "lease_knowledge"}:
        return False
    text = re.sub(r"\s+", "", str(message or ""))
    if len(text) < 4:
        return False
    return any(term in text for term in NO_DATA_RESEARCH_TERMS)


def build_external_research_suggestion(
    message: str,
    *,
    question_category: str,
    knowledge_ref_count: int,
    response_mode: str,
) -> dict[str, Any]:
    """Return a consent request when a chat turn should use fresh web research.

    This only asks for permission. It does not call web services, write Obsidian,
    or promote anything into judgment assets.
    """
    if (response_mode or "shion").strip().lower() == "general":
        return {"needed": False}
    text = (message or "").strip()
    if not text or len(text) < 8:
        return {"needed": False}

    lower = text.lower()
    explicit_research = any(
        term in lower
        for term in (
            "ネットで調べ",
            "webで調べ",
            "ウェブで調べ",
            "検索して",
            "リサーチして",
            "外部調査",
            "調査器官",
        )
    )
    no_data_research = _should_offer_research_for_no_data(
        text,
        question_category=question_category,
        knowledge_ref_count=knowledge_ref_count,
    )
    if not explicit_research and not no_data_research:
        return {"needed": False}

    topic = external_research_topic_from_message(text)
    reason_parts: list[str] = []
    if explicit_research:
        reason_parts.append("ユーザーが外部調査を求めています")
    if no_data_research:
        reason_parts.append("手元のObsidian/RAGに該当データがありません")
    if knowledge_ref_count == 0 and question_category not in {"general", "news_summarize"}:
        reason_parts.append("Obsidian/RAGの参照が薄い論点です")
    reason = " / ".join(reason_parts[:2]) or "外部情報で補う価値があります"
    return {
        "needed": True,
        "topic": topic,
        "reason": reason,
        "output_dir": "Projects/tune_lease_55/Research/Auto Research/",
    }


def external_research_permission_reply(suggestion: dict[str, Any]) -> str:
    topic = str(suggestion.get("topic") or "").strip() or "この論点"
    reason = str(suggestion.get("reason") or "").strip() or "外部情報で補う価値があります"
    return (
        "この件は手元のデータがありません。ネットで調査してもいいですか？\n\n"
        f"- 足りない情報: {topic}\n"
        f"- 理由: {reason}\n\n"
        "許可をもらえれば、調査結果をObsidianのResearchノートに保存してから回答に使います。"
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def capture_vertex_materials_for_research(
    topic: str,
    *,
    vault_path: Path,
    modes: tuple[str, ...] = ("evidence_support", "judgment_candidates"),
    page_size: int = 5,
    workflow_runner: Callable[..., dict[str, Any]] | None = None,
    capture_runner: Callable[..., dict[str, Any]] | None = None,
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Best-effort Vertex material capture paired with external Research.

    The saved Vertex notes are research material only. They are not promoted
    into judgment assets and must not break the normal Research path.
    """
    clean_topic = (topic or "").strip()[:160]
    if not clean_topic:
        return {"used": False, "reason": "empty_topic", "captures": []}
    if not _env_bool("VERTEX_RESEARCH_MATERIAL_CAPTURE_ENABLED", True):
        return {"used": False, "reason": "disabled", "captures": []}

    if workflow_runner is None or capture_runner is None:
        from api.vertex_distillation import capture_vertex_workflow_result
        from api.vertex_knowledge_workflows import run_vertex_knowledge_workflow

        workflow_runner = workflow_runner or run_vertex_knowledge_workflow
        capture_runner = capture_runner or capture_vertex_workflow_result

    state = state_path or (get_data_dir() / "vertex_research_material_capture_state.json")
    captures: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for mode in modes:
        try:
            workflow = workflow_runner(clean_topic, mode=mode, page_size=page_size)
            capture = capture_runner(workflow, vault_path=vault_path, state_path=state)
            captures.append(
                {
                    "mode": mode,
                    "query": workflow.get("query"),
                    "refs": workflow.get("refs") or [],
                    "capture": capture,
                }
            )
        except Exception as exc:  # noqa: BLE001 - never break Research mode
            errors.append({"mode": mode, "error": str(exc)[:240]})
    saved_paths = [
        str((item.get("capture") or {}).get("note_path") or "").strip()
        for item in captures
        if (item.get("capture") or {}).get("captured")
    ]
    saved_paths = [path for path in saved_paths if path]
    return {
        "used": bool(captures or errors),
        "topic": clean_topic,
        "modes": list(modes),
        "captures": captures,
        "saved_paths": saved_paths,
        "errors": errors,
    }


def run_external_research_for_chat(topic: str) -> dict[str, Any]:
    topic = (topic or "").strip()[:160]
    if not topic:
        return {"used": False, "error": "調査テーマが空です"}
    vault = _research_organ_vault_path()
    from scripts.auto_research_lease_judgment import DEFAULT_OUTPUT_DIR, run as run_auto_research

    result = run_auto_research(vault, DEFAULT_OUTPUT_DIR, topic, False)
    vertex_materials = capture_vertex_materials_for_research(topic, vault_path=vault)
    display = _research_run_display(result)
    note_path = str(result.get("path") or "")
    rel_path = ""
    if note_path:
        try:
            rel_path = Path(note_path).resolve().relative_to(vault.resolve()).as_posix()
        except Exception:
            rel_path = note_path
    summary = [str(item).strip() for item in display.get("summary", []) if str(item).strip()]
    use_cases = [str(item).strip() for item in display.get("use_cases", []) if str(item).strip()]
    questions = [str(item).strip() for item in display.get("review_questions", []) if str(item).strip()]
    context_lines = [
        "【外部調査Researchノート】",
        f"- テーマ: {result.get('title') or topic}",
        f"- 保存先: {rel_path or note_path}",
        f"- 出典URL数: {result.get('source_count', 0)}",
        "- 注意: この調査は needs_human_review の材料。承認/否決/スコアを直接変更せず、確認質問・条件・留保として使う。",
    ]
    if summary:
        context_lines.append("結論:")
        context_lines.extend(f"- {item}" for item in summary[:3])
    if use_cases:
        context_lines.append("リース審査への適用:")
        context_lines.extend(f"- {item}" for item in use_cases[:4])
    if questions:
        context_lines.append("担当者が確認する質問:")
        context_lines.extend(f"- {item}" for item in questions[:3])
    saved_vertex_paths = [
        str(path).strip()
        for path in (vertex_materials.get("saved_paths") or [])
        if str(path).strip()
    ]
    if saved_vertex_paths:
        context_lines.append("Vertex採掘材料:")
        context_lines.extend(f"- {path}" for path in saved_vertex_paths[:4])
    return {
        "used": True,
        "topic": topic,
        "result": result,
        "display": display,
        "note_path": rel_path or note_path,
        "vertex_materials": vertex_materials,
        "prompt_context": "\n".join(context_lines),
    }
