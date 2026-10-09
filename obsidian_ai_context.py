"""Shared Obsidian context builder for AI chat entry points."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONTEXT_TOKEN_BUDGET = 700
DEFAULT_WORKLOG_DIGEST = Path(__file__).resolve().parent / "reports" / "agent_worklog_digest_latest.json"


def _load_obsidian_bridge():
    try:
        from mobile_app.obsidian_bridge import build_obsidian_digest, collect_obsidian_context

        return collect_obsidian_context, build_obsidian_digest
    except Exception:
        try:
            from obsidian_bridge import build_obsidian_digest, collect_obsidian_context

            return collect_obsidian_context, build_obsidian_digest
        except Exception:
            return None, None


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_typesafe_rag_filter(query: str = ""):
    try:
        from api.chat_routing import is_potentially_sensitive_screening_message
        from typesafe_rag_guard import filter_hits_if_enabled, typesafe_rag_enabled

        if not _env_truthy(os.environ.get("TYPESAFE_ALLOW_SHARED_CONTEXT")):
            return None
        if is_potentially_sensitive_screening_message(query) and not _env_truthy(
            os.environ.get("TYPESAFE_ALLOW_SCREENING")
        ):
            return None
        return filter_hits_if_enabled if typesafe_rag_enabled() else None
    except Exception:
        return None


def _select_hits_with_budget(hits: list[dict[str, Any]], *, limit: int, max_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select retrieval hits before prompt assembly, using a cheap token proxy."""
    selected: list[dict[str, Any]] = []
    used_tokens = 0
    skipped_for_budget = 0
    for hit in hits[: max(limit * 2, limit)]:
        snippet = str(hit.get("snippet") or "")
        path = str(hit.get("path") or "")
        cost = _estimate_tokens(f"{path}\n{snippet[:320]}")
        if selected and used_tokens + cost > max_tokens:
            skipped_for_budget += 1
            continue
        if not selected and cost > max_tokens:
            hit = {**hit, "snippet": snippet[: max(120, max_tokens * 4)]}
            cost = _estimate_tokens(f"{path}\n{hit.get('snippet') or ''}")
        selected.append(hit)
        used_tokens += cost
        if len(selected) >= limit:
            break
    return selected, {
        "candidate_count": len(hits),
        "selected_count": len(selected),
        "token_budget": max_tokens,
        "estimated_tokens": used_tokens,
        "skipped_for_budget": skipped_for_budget,
    }


def _mark_self_answers(hits: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """REV-544: 開発用ノートを外し、対話室の会話ログ（紫苑自身の過去の回答）に写さない印を付ける。"""
    from api.answer_repeat_guard import SELF_ANSWER_START, is_dev_note_path, is_self_answer_path

    marked: list[dict[str, Any]] = []
    for hit in hits or []:
        path = str(hit.get("path") or "")
        if is_dev_note_path(path):
            continue
        if is_self_answer_path(path) and SELF_ANSWER_START not in str(hit.get("snippet") or ""):
            hit = {**hit, "snippet": f"{SELF_ANSWER_START}{hit.get('snippet') or ''}"}
        marked.append(hit)
    return marked


def collect_obsidian_ai_context(
    query: str,
    *,
    limit: int = 4,
    max_chars: int = 2400,
    max_tokens: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    heading: str = "Obsidian知識ノート",
) -> dict[str, Any]:
    """Return compact prompt context and source metadata through the shared bridge."""
    collect_obsidian_context, build_obsidian_digest = _load_obsidian_bridge()
    if collect_obsidian_context is None or build_obsidian_digest is None:
        return {"block": "", "hits": [], "source_count": 0, "retrieval_boundary": {}}
    try:
        typesafe_filter = _load_typesafe_rag_filter(query)
        candidate_limit = max(limit * 2, limit) if typesafe_filter is not None else limit
        hits: list[dict[str, Any]] = _mark_self_answers(collect_obsidian_context(query, limit=candidate_limit))
        if not hits:
            return {"block": "", "hits": [], "source_count": 0, "retrieval_boundary": {}}
        typesafe_boundary: dict[str, Any] = {"status": "unavailable"}
        if typesafe_filter is not None:
            hits, typesafe_boundary = typesafe_filter(query, hits)
        if not hits:
            return {
                "block": "",
                "hits": [],
                "source_count": 0,
                "retrieval_boundary": {"typesafe": typesafe_boundary},
            }
        selected_hits, boundary = _select_hits_with_budget(hits, limit=limit, max_tokens=max_tokens)
        boundary["typesafe"] = typesafe_boundary
        digest = build_obsidian_digest(query, selected_hits)
    except Exception:
        return {"block": "", "hits": [], "source_count": 0, "retrieval_boundary": {}}

    lines = [
        f"【{heading}】",
        "以下は iCloud 上の Obsidian Vault から検索した社内知識です。チャットログより知識ノートを優先しています。",
    ]
    digest_text = str((digest or {}).get("digest") or "").strip()
    if digest_text:
        lines.append(digest_text)
    lines.append("### 検索ヒット")
    for hit in selected_hits:
        path = str(hit.get("path") or "").strip()
        snippet = str(hit.get("snippet") or "").strip().replace("\n", " ")
        if path:
            lines.append(f"- {path}: {snippet[:320]}")
    block = "\n".join(lines).strip()[:max_chars]
    boundary["block_chars"] = len(block)
    return {
        "block": block,
        "hits": selected_hits,
        "source_count": len(selected_hits),
        "retrieval_boundary": boundary,
    }


def build_obsidian_ai_context_block(
    query: str,
    *,
    limit: int = 4,
    max_chars: int = 2400,
    max_tokens: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    heading: str = "Obsidian知識ノート",
) -> str:
    """Return a compact Obsidian context block for AI prompts."""
    return str(
        collect_obsidian_ai_context(
            query,
            limit=limit,
            max_chars=max_chars,
            max_tokens=max_tokens,
            heading=heading,
        ).get("block")
        or ""
    )


def build_recent_worklog_ai_context_block(
    *,
    limit: int = 4,
    days: int = 14,
    max_chars: int = 1800,
    report_path: Path = DEFAULT_WORKLOG_DIGEST,
    today: dt.date | None = None,
) -> str:
    """Build sanitized, recent work-log context through the shared AI context module.

    The batch digest has already reduced Vault notes to public fields.  This
    boundary re-validates source folders and dates, and never exposes raw note
    text or unknown sections to an AI prompt.
    """
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return ""

    current_day = today or dt.date.today()
    cutoff = current_day - dt.timedelta(days=max(1, days) - 1)
    allowed_markers = ("/Daily/", "/Projects/tune_lease_55/Work Logs/")
    eligible: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source = "/" + str(item.get("source_path") or "").replace("\\", "/").lstrip("/")
        if not any(marker in source for marker in allowed_markers):
            continue
        try:
            item_day = dt.date.fromisoformat(str(item.get("date") or ""))
        except ValueError:
            continue
        if item_day < cutoff or item_day > current_day:
            continue
        eligible.append(item)

    eligible.sort(
        key=lambda item: (str(item.get("date") or ""), str(item.get("time") or "")),
        reverse=True,
    )
    selected = eligible[: max(0, limit)]
    if not selected:
        return ""

    def public_text(item: dict[str, Any], field: str, char_limit: int) -> str:
        values = item.get(field)
        if not isinstance(values, list):
            return ""
        return " / ".join(str(value).strip() for value in values if str(value).strip())[:char_limit]

    lines = [
        "【Codex/Claude 作業録】",
        f"直近{max(1, days)}日以内の公開フィールドだけを使用しています。",
    ]
    for item in selected:
        title = f"{item.get('date') or ''} {item.get('time') or ''} {item.get('agent') or ''}".strip()
        summary = public_text(item, "summary", 180)
        decisions = public_text(item, "decisions", 220)
        changes = public_text(item, "changes", 180)
        line = f"- {title}"
        if summary:
            line += f" / 要約: {summary}"
        if decisions:
            line += f" / 判断: {decisions}"
        if changes:
            line += f" / 変更: {changes}"
        lines.append(line)
    return "\n".join(lines)[:max_chars]


__all__ = [
    "build_obsidian_ai_context_block",
    "build_recent_worklog_ai_context_block",
    "collect_obsidian_ai_context",
]
