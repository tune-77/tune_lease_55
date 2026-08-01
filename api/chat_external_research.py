"""External research consent and context helpers for chat."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from api.routers.vault_hub import _research_organ_vault_path, _research_run_display


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
    if not explicit_research:
        return {"needed": False}

    topic = external_research_topic_from_message(text)
    reason_parts: list[str] = []
    if explicit_research:
        reason_parts.append("ユーザーが外部調査を求めています")
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
        "ここは手元の記憶だけで断定しない方がいいです。\n\n"
        f"- 足りない情報: {topic}\n"
        f"- 理由: {reason}\n\n"
        "ネットで調べて、ObsidianのResearchノートに保存してから回答に使っていいですか？"
    )


def run_external_research_for_chat(topic: str) -> dict[str, Any]:
    topic = (topic or "").strip()[:160]
    if not topic:
        return {"used": False, "error": "調査テーマが空です"}
    vault = _research_organ_vault_path()
    from scripts.auto_research_lease_judgment import DEFAULT_OUTPUT_DIR, run as run_auto_research

    result = run_auto_research(vault, DEFAULT_OUTPUT_DIR, topic, False)
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
    return {
        "used": True,
        "topic": topic,
        "result": result,
        "display": display,
        "note_path": rel_path or note_path,
        "prompt_context": "\n".join(context_lines),
    }
