"""Shared Obsidian context builder for AI chat entry points."""

from __future__ import annotations

from typing import Any

DEFAULT_CONTEXT_TOKEN_BUDGET = 700


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
        hits: list[dict[str, Any]] = collect_obsidian_context(query, limit=limit)
        if not hits:
            return {"block": "", "hits": [], "source_count": 0, "retrieval_boundary": {}}
        selected_hits, boundary = _select_hits_with_budget(hits, limit=limit, max_tokens=max_tokens)
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


__all__ = ["build_obsidian_ai_context_block", "collect_obsidian_ai_context"]
