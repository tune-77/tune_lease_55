"""Shared Obsidian context builder for AI chat entry points."""

from __future__ import annotations

from typing import Any


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


def build_obsidian_ai_context_block(
    query: str,
    *,
    limit: int = 4,
    max_chars: int = 2400,
    heading: str = "Obsidian知識ノート",
) -> str:
    """Return a compact Obsidian context block for AI prompts."""
    collect_obsidian_context, build_obsidian_digest = _load_obsidian_bridge()
    if collect_obsidian_context is None or build_obsidian_digest is None:
        return ""
    try:
        hits: list[dict[str, Any]] = collect_obsidian_context(query, limit=limit)
        if not hits:
            return ""
        digest = build_obsidian_digest(query, hits)
    except Exception:
        return ""

    lines = [
        f"【{heading}】",
        "以下は iCloud 上の Obsidian Vault から検索した社内知識です。チャットログより知識ノートを優先しています。",
    ]
    digest_text = str((digest or {}).get("digest") or "").strip()
    if digest_text:
        lines.append(digest_text)
    lines.append("### 検索ヒット")
    for hit in hits[:limit]:
        path = str(hit.get("path") or "").strip()
        snippet = str(hit.get("snippet") or "").strip().replace("\n", " ")
        if path:
            lines.append(f"- {path}: {snippet[:320]}")
    return "\n".join(lines).strip()[:max_chars]


__all__ = ["build_obsidian_ai_context_block"]
