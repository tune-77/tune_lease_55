"""Obsidian Curator tools for Shion's ADK agent.

These tools are intentionally read-only. They inspect existing sidecar reports
and the retrieval graph, then return small, reviewable curation suggestions.
They do not edit the Vault, rebuild ChromaDB, alter prompts, or promote memory.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from runtime_paths import resolve_obsidian_vault

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GRAPH_EFFECT_JSON = _REPO_ROOT / "reports" / "obsidian_graph_judgment_effect_latest.json"
_RETRIEVAL_GRAPH_JSON = _REPO_ROOT / "data" / "obsidian_retrieval_graph.json"
_ENV_MONITOR_JSON = _REPO_ROOT / "reports" / "obsidian_environment_monitor_latest.json"

_AUTO_CURATION_EXCLUDED_PARTS = (
    "Daily/",
    "AI Chat/",
    "Cloud Run Conversation Log/",
    "Improvement Log/",
    "Work Logs/",
    "作業ログ/",
    "06-日記_作業ログ/",
    "Private Reflection/",
    "Reflection/",
    "Generated/",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _clamp_limit(limit: int | None, *, default: int = 5, maximum: int = 10) -> int:
    try:
        raw = int(limit if limit is not None else default)
    except (TypeError, ValueError):
        raw = default
    return max(1, min(maximum, raw))


def _is_auto_curation_target(path: str) -> bool:
    return not any(part in path for part in _AUTO_CURATION_EXCLUDED_PARTS)


def _top_non_log_hubs(graph_effect: dict[str, Any], *, limit: int = 3) -> list[str]:
    hubs: list[str] = []
    for item in graph_effect.get("top_effective_hubs") or []:
        path = str(item.get("path") or "")
        if not path or not _is_auto_curation_target(path):
            continue
        if path.lower().endswith(".md"):
            path = path[:-3]
        hubs.append(path)
        if len(hubs) >= limit:
            break
    return hubs


def _brief_env_checks(env_report: dict[str, Any]) -> dict[str, Any]:
    checks = env_report.get("checks") if isinstance(env_report.get("checks"), dict) else {}
    if not checks and isinstance(env_report.get("checks"), list):
        checks = {
            str(item.get("name") or ""): item
            for item in env_report.get("checks") or []
            if isinstance(item, dict)
        }
    if not checks and isinstance(env_report.get("results"), list):
        checks = {
            str(item.get("name") or ""): item
            for item in env_report.get("results") or []
            if isinstance(item, dict)
        }
    out: dict[str, Any] = {}
    for name in ("vault", "rag_index", "wikilinks", "recent_note_noise", "private_reflection_meaning"):
        item = checks.get(name) if isinstance(checks, dict) else None
        if isinstance(item, dict):
            out[name] = {
                "status": item.get("status"),
                "message": item.get("message"),
            }
    return out


def review_obsidian_vault_health(limit: int = 5) -> dict[str, Any]:
    """Review Obsidian health and return the top curation opportunities.

    Args:
        limit: Maximum number of suggestions to return, clamped to 1..10.

    Returns:
        A read-only report with graph health, monitor checks, and suggested
        human-review actions. No files are modified.
    """
    max_items = _clamp_limit(limit)
    graph_effect = _read_json(_GRAPH_EFFECT_JSON)
    retrieval_graph = _read_json(_RETRIEVAL_GRAPH_JSON)
    env_report = _read_json(_ENV_MONITOR_JSON)

    nodes = retrieval_graph.get("nodes") or []
    degree0 = [n for n in nodes if isinstance(n, dict) and int(n.get("degree") or 0) == 0]
    degree0_dirs = Counter(
        str(n.get("path") or ".").split("/", 1)[0] if "/" in str(n.get("path") or "") else "."
        for n in degree0
    )
    graph_summary = graph_effect.get("summary") if isinstance(graph_effect.get("summary"), dict) else {}
    retrieval_summary = (
        retrieval_graph.get("summary") if isinstance(retrieval_graph.get("summary"), dict) else {}
    )
    hubs = _top_non_log_hubs(graph_effect, limit=3)

    proposals: list[dict[str, Any]] = []
    for item in graph_effect.get("isolated_but_used") or []:
        path = str(item.get("path") or "")
        if not path or not _is_auto_curation_target(path):
            continue
        proposals.append({
            "type": "connect_used_isolate",
            "path": path,
            "reason": "使用シグナルがあるのにリンクが薄い",
            "used_score": item.get("used_score"),
            "degree": item.get("degree"),
            "suggested_links": hubs,
            "requires_human_approval": True,
        })
        if len(proposals) >= max_items:
            break

    if len(proposals) < max_items:
        for dirname, count in degree0_dirs.most_common(max_items - len(proposals)):
            if dirname in {"Generated", "Daily", "Claude会話記録", "改善ログ"}:
                action = "原則放置。検索で問題が出たテーマだけ手動接続。"
            else:
                action = "テーマ単位で確認し、使われるノートだけ少数リンクを追加。"
            proposals.append({
                "type": "inspect_dormant_isolates",
                "directory": dirname,
                "degree0_count": count,
                "reason": "degree 0 が残っているフォルダ",
                "suggested_action": action,
                "requires_human_approval": True,
            })

    return {
        "mode": "read_only_obsidian_curator",
        "vault": str(resolve_obsidian_vault()),
        "status": "ok" if graph_summary or retrieval_summary else "missing_reports",
        "summary": {
            "graph_notes": graph_summary.get("notes"),
            "graph_edges": graph_summary.get("edges"),
            "graph_buckets": graph_summary.get("buckets"),
            "retrieval_notes": retrieval_summary.get("notes"),
            "retrieval_edges": retrieval_summary.get("edges"),
            "degree0_count": len(degree0),
            "top_degree0_dirs": dict(degree0_dirs.most_common(8)),
        },
        "monitor_checks": _brief_env_checks(env_report),
        "proposals": proposals[:max_items],
        "guardrail": "read_only_no_vault_write_no_chroma_reindex_no_prompt_change",
    }


def suggest_obsidian_curation_actions(theme: str = "", limit: int = 5) -> dict[str, Any]:
    """Suggest targeted Obsidian links/search terms for a theme.

    Args:
        theme: Topic that failed search/chat retrieval. Empty theme falls back
            to the general vault health review.
        limit: Maximum number of candidate notes to return, clamped to 1..10.

    Returns:
        Read-only suggestions. The caller should apply changes only after human
        approval and backup.
    """
    max_items = _clamp_limit(limit)
    query = str(theme or "").strip()
    if not query:
        return review_obsidian_vault_health(limit=max_items)

    retrieval_graph = _read_json(_RETRIEVAL_GRAPH_JSON)
    graph_effect = _read_json(_GRAPH_EFFECT_JSON)
    hubs = _top_non_log_hubs(graph_effect, limit=3)
    if not retrieval_graph:
        return {
            "mode": "read_only_obsidian_curator",
            "theme": query,
            "status": "missing_retrieval_graph",
            "suggestions": [],
            "next_action": "scripts/build_obsidian_retrieval_graph.py を再生成してから再確認する",
            "guardrail": "read_only_no_vault_write_no_chroma_reindex_no_prompt_change",
        }

    try:
        from scripts.build_obsidian_retrieval_graph import route_query

        candidates = route_query(query, retrieval_graph, limit=max_items)
    except Exception as exc:  # noqa: BLE001 - ADK tool should report, not crash.
        return {
            "mode": "read_only_obsidian_curator",
            "theme": query,
            "status": f"route_failed: {type(exc).__name__}",
            "suggestions": [],
            "guardrail": "read_only_no_vault_write_no_chroma_reindex_no_prompt_change",
        }

    suggestions: list[dict[str, Any]] = []
    for node in candidates[:max_items]:
        path = str(node.get("path") or "")
        existing_links = [str(link) for link in (node.get("links") or [])[:8]]
        search_terms = [str(term) for term in (node.get("matched_terms") or [])[:8]]
        if not search_terms:
            search_terms = [str(term) for term in (node.get("keywords") or [])[:5]]
        suggested_links = [hub for hub in hubs if Path(hub).stem not in existing_links]
        suggestions.append({
            "path": path,
            "title": node.get("title"),
            "route_score": node.get("route_score"),
            "query_salience": node.get("query_salience"),
            "degree": node.get("degree"),
            "traversal_edge_count": node.get("traversal_edge_count"),
            "estimated_token_saving_ratio": node.get("estimated_token_saving_ratio"),
            "existing_wikilinks": existing_links,
            "linked_context": node.get("linked_context") or [],
            "suggested_links": suggested_links[:2],
            "suggested_search_terms": search_terms[:5],
            "recommended_edit": "対象ノートの Related または検索語セクションに少数追加",
            "requires_human_approval": True,
        })

    return {
        "mode": "read_only_obsidian_curator",
        "theme": query,
        "status": "ok",
        "suggestions": suggestions,
        "guardrail": "read_only_no_vault_write_no_chroma_reindex_no_prompt_change",
    }


OBSIDIAN_CURATOR_TOOLS = [
    review_obsidian_vault_health,
    suggest_obsidian_curation_actions,
]
