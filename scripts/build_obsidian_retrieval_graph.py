#!/usr/bin/env python3
"""Build a lightweight retrieval graph index for Obsidian notes.

This is a read-only sidecar for the "router + index + nodes + edges" pattern.
It scans markdown through the shared Obsidian file lister, writes a compact
machine-readable index, and can rank candidate notes from that index without
opening note bodies. It does not edit Obsidian, prompts, RAG rankings, scoring,
or the daily improvement pipeline.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from obsidian_query import list_vault_md_files, split_query_terms  # noqa: E402
from runtime_paths import resolve_obsidian_vault  # noqa: E402
from scripts._obsidian_common import safe_rel as _safe_rel  # noqa: E402

DEFAULT_VAULT = resolve_obsidian_vault()
DEFAULT_OUTPUT_JSON = REPO_ROOT / "data" / "obsidian_retrieval_graph.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "obsidian_retrieval_graph_latest.md"

GUARDRAIL = "read_only_index_no_obsidian_write_no_rag_rank_no_prompt_no_scoring_change"
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_\-/ぁ-んァ-ン一-龥]+)")
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
EXCLUDED_PARTS = {
    ".obsidian",
    ".trash",
    ".git",
    ".claude",
    ".claudian",
    "node_modules",
}
EXCLUDED_SUFFIXES = {".canvas.md"}
PRIVATE_PATH_HINTS = ("Private Reflection", "私室")
CHAT_PATH_HINTS = ("AI Chat", "Cloud Run Conversation Log", "Dialogue")
REFLECTION_PATH_HINTS = ("Private Reflection", "Reflection", "内省")
TECH_PATH_HINTS = ("Codex", "Dev", "Logs", "技術", "実装")
JUDGMENT_PATH_HINTS = (
    "Judgment",
    "Cases",
    "Research",
    "Lease Intelligence",
    "リース",
    "審査",
    "判断",
)
VALUE_PATH_HINTS = ("Identity", "Mana", "価値", "人格", "紫苑")
ROUTE_ORDER = (
    "judgment_memory",
    "factual_memory",
    "technical_memory",
    "dialogue_memory",
    "reflection_memory",
    "value_memory",
)
GENERIC_ROUTE_TERMS = {
    "リース",
    "lease",
    "確認",
    "確認事項",
    "注意",
    "注意点",
    "ポイント",
    "方法",
    "具体的",
    "教えて",
    "会社",
    "案件",
}
HIGH_VALUE_PATH_PARTS = (
    "asset knowledge/",
    "judgment assets/",
    "lease intelligence/",
    "research/vertex distilled/",
    "lease-wiki-vault/",
    "リース審査基礎知識/",
    "リース知識/",
)
LOW_VALUE_PATH_PARTS = (
    "05-クリップ_記事/",
    "06-日記_作業ログ/",
    "improvement log/",
    "ai chat/",
    "daily/",
    "weekly review",
)
NEWS_QUERY_TERMS = ("ニュース", "記事", "報道", "最新", "今日", "昨日")
LOG_QUERY_TERMS = ("ログ", "改善", "会話", "チャット", "日報")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _strip_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    end = 0
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = idx
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip().lower()] = value.strip().strip("'\"")
    if not end:
        return meta, text
    return meta, "\n".join(lines[end + 1 :])


def _title(meta: dict[str, str], body: str, path: Path) -> str:
    if meta.get("title"):
        return meta["title"]
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def _headings(body: str, *, limit: int = 12) -> list[str]:
    headings: list[str] = []
    for line in body.splitlines():
        match = HEADING_RE.match(line)
        if not match:
            continue
        heading = match.group(2).strip(" #")
        if heading:
            headings.append(heading[:160])
        if len(headings) >= limit:
            break
    return headings


def _summary(meta: dict[str, str], body: str, title: str) -> str:
    for key in ("summary", "description", "概要"):
        if meta.get(key):
            return meta[key][:220]
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "---", "```", "<!--")):
            continue
        if line.startswith(("- [", "- TODO", "- todo")):
            continue
        return line.lstrip("- ").strip()[:220]
    return title[:220]


def _note_key(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".md"):
        text = text[:-3]
    return Path(text).stem.strip()


def _wikilinks(body: str) -> list[str]:
    seen: set[str] = set()
    links: list[str] = []
    for raw in WIKILINK_RE.findall(body):
        key = _note_key(raw)
        if key and key not in seen:
            links.append(key)
            seen.add(key)
    return links[:50]


def _tags(meta: dict[str, str], body: str) -> list[str]:
    tags: set[str] = set()
    raw = meta.get("tags", "")
    for part in re.split(r"[\s,\[\]]+", raw):
        tag = part.strip().strip("#'\"")
        if tag:
            tags.add(tag)
    for match in TAG_RE.findall(body):
        tags.add(match.strip("/"))
    return sorted(tags)[:30]


def _path_terms(rel_path: str) -> list[str]:
    parts: list[str] = []
    for part in Path(rel_path).parts:
        if part.endswith(".md"):
            part = part[:-3]
        parts.extend(split_query_terms(part.replace("_", " ").replace("-", " ")))
    return parts


def _keywords(title: str, rel_path: str, headings: list[str], tags: list[str], summary: str) -> list[str]:
    terms: list[str] = []
    terms.extend(split_query_terms(title))
    terms.extend(_path_terms(rel_path))
    terms.extend(tags)
    for heading in headings[:6]:
        terms.extend(split_query_terms(heading))
    terms.extend(split_query_terms(summary))
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        cleaned = term.strip().lower()
        if len(cleaned) < 2 or cleaned in seen:
            continue
        out.append(cleaned)
        seen.add(cleaned)
        if len(out) >= 36:
            break
    return out


def _route(rel_path: str, title: str, tags: list[str]) -> str:
    hay = " ".join([rel_path, title, " ".join(tags)]).lower()
    if any(hint.lower() in hay for hint in REFLECTION_PATH_HINTS):
        return "reflection_memory"
    if any(hint.lower() in hay for hint in CHAT_PATH_HINTS):
        return "dialogue_memory"
    if any(hint.lower() in hay for hint in TECH_PATH_HINTS):
        return "technical_memory"
    if any(hint.lower() in hay for hint in VALUE_PATH_HINTS):
        return "value_memory"
    if any(hint.lower() in hay for hint in JUDGMENT_PATH_HINTS):
        return "judgment_memory"
    return "factual_memory"


def _should_skip(path: Path, *, include_private: bool = False) -> bool:
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return True
    if not include_private and any(hint in path.parts for hint in PRIVATE_PATH_HINTS):
        return True
    return any(path.name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 3)


def build_index(vault: Path, *, include_private: bool = False) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    by_stem: dict[str, str] = {}
    duplicate_stems: Counter[str] = Counter()
    backlinks: dict[str, set[str]] = defaultdict(set)

    if vault.exists():
        for path in sorted(list_vault_md_files(vault)):
            if _should_skip(path, include_private=include_private):
                continue
            rel = _safe_rel(path, vault)
            body_raw = _read_text(path)
            meta, body = _strip_frontmatter(body_raw)
            title = _title(meta, body, path)
            headings = _headings(body)
            tags = _tags(meta, body)
            summary = _summary(meta, body, title)
            links = _wikilinks(body)
            stem = path.stem
            duplicate_stems[stem] += 1
            by_stem.setdefault(stem, rel)
            node = {
                "path": rel,
                "key": stem,
                "title": title,
                "route": _route(rel, title, tags),
                "summary": summary,
                "keywords": _keywords(title, rel, headings, tags, summary),
                "tags": tags,
                "headings": headings,
                "links": links,
                "out_degree": len(links),
                "token_estimate": _token_estimate(body_raw),
                "mtime": int(path.stat().st_mtime) if path.exists() else 0,
            }
            nodes.append(node)

    known_keys = {node["key"] for node in nodes}
    for node in nodes:
        for link in node["links"]:
            if link in known_keys:
                backlinks[link].add(str(node["key"]))
    for node in nodes:
        key = str(node["key"])
        node["backlinks"] = sorted(backlinks.get(key, set()))[:50]
        node["in_degree"] = len(backlinks.get(key, set()))
        node["degree"] = int(node["in_degree"]) + int(node["out_degree"])
        node["duplicate_stem_count"] = int(duplicate_stems[key])
        if duplicate_stems[key] > 1:
            node["warning"] = "duplicate_note_stem_makes_wikilink_resolution_ambiguous"

    route_counts = Counter(str(node["route"]) for node in nodes)
    token_total = sum(int(node["token_estimate"]) for node in nodes)
    compact_total = sum(_token_estimate(" ".join([
        str(node["path"]),
        str(node["summary"]),
        " ".join(node["keywords"]),
    ])) for node in nodes)
    router = {
        route: [
            {
                "path": node["path"],
                "title": node["title"],
                "summary": node["summary"],
                "degree": node["degree"],
            }
            for node in sorted(
                [n for n in nodes if n["route"] == route],
                key=lambda n: (int(n["degree"]), int(n["mtime"])),
                reverse=True,
            )[:8]
        ]
        for route in ROUTE_ORDER
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 2,
        "guardrail": GUARDRAIL,
        "vault": str(vault),
        "summary": {
            "vault_exists": vault.exists(),
            "include_private": include_private,
            "retrieval_mode": "graph_preroute_then_trace_relevant_edges_only",
            "notes": len(nodes),
            "routes": dict(sorted(route_counts.items())),
            "edges": sum(int(node["out_degree"]) for node in nodes),
            "estimated_raw_tokens": token_total,
            "estimated_index_tokens": compact_total,
            "estimated_token_reduction_ratio": round(
                (1.0 - (compact_total / token_total)) if token_total else 0.0,
                4,
            ),
        },
        "router": router,
        "nodes": sorted(nodes, key=lambda n: str(n["path"])),
        "path_by_stem": by_stem,
    }


def route_query(query: str, index: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    terms = split_query_terms(query)
    if not terms:
        return []
    term_set = {_normalize_route_term(term) for term in terms if _normalize_route_term(term)}
    if (
        {"q", "risk"}.issubset(term_set)
        or any("q-risk" in term or "q_risk" in term or "qrisk" in term for term in term_set)
    ):
        term_set.update({"q-risk", "q_risk", "qrisk"})
    strong_terms = {term for term in term_set if term not in GENERIC_ROUTE_TERMS}
    scoring_terms = strong_terms or term_set
    normalized_query = " ".join(term_set)
    wants_news = any(term in normalized_query for term in NEWS_QUERY_TERMS)
    wants_logs = any(term in normalized_query for term in LOG_QUERY_TERMS)
    node_by_key = {
        str(node.get("key")): node
        for node in index.get("nodes") or []
        if isinstance(node, dict) and node.get("key")
    }
    raw_token_total = int((index.get("summary") or {}).get("estimated_raw_tokens") or 0)
    scored: list[tuple[float, dict[str, Any]]] = []
    for node in index.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        keywords = {_normalize_route_term(str(term)) for term in node.get("keywords") or []}
        title = _normalize_route_term(str(node.get("title") or ""))
        path = _normalize_route_term(str(node.get("path") or ""))
        summary = _normalize_route_term(str(node.get("summary") or ""))
        score = 0.0
        matched: list[str] = []
        for term in scoring_terms:
            if term in keywords:
                score += 5.0
                matched.append(term)
            if term and term in title:
                score += 3.0
            if term and term in path:
                score += 2.0
            if term and term in summary:
                score += 1.5
        score += _route_path_quality_bonus(path, wants_news=wants_news, wants_logs=wants_logs)
        score += min(2.0, float(node.get("degree") or 0) * 0.15)
        if score <= 0:
            continue
        linked_context = _linked_context(node, node_by_key, scoring_terms, limit=4)
        traced_tokens = int(node.get("token_estimate") or 0) + sum(
            int(item.get("token_estimate") or 0) for item in linked_context
        )
        token_saving_ratio = (
            round(1.0 - (traced_tokens / raw_token_total), 4)
            if raw_token_total and traced_tokens <= raw_token_total
            else 0.0
        )
        salience_base = max(1.0, float(node.get("token_estimate") or 1) / 100.0)
        scored.append((
            score,
            {
                **node,
                "route_score": round(score, 3),
                "matched_terms": sorted(set(matched)),
                "query_salience": round(score / salience_base, 3),
                "linked_context": linked_context,
                "traversal_edge_count": len(linked_context),
                "estimated_tokens_if_full_scan": raw_token_total,
                "estimated_tokens_if_traced": traced_tokens,
                "estimated_token_saving_ratio": token_saving_ratio,
            },
        ))
    scored.sort(key=lambda item: (item[0], int(item[1].get("mtime") or 0)), reverse=True)
    return [node for _, node in scored[:limit]]


def _linked_context(
    node: dict[str, Any],
    node_by_key: dict[str, dict[str, Any]],
    scoring_terms: set[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Return adjacent notes worth tracing after the primary hit is selected."""
    neighbor_keys: list[tuple[str, str]] = []
    for key in node.get("links") or []:
        neighbor_keys.append((str(key), "out"))
    for key in node.get("backlinks") or []:
        neighbor_keys.append((str(key), "back"))

    seen: set[str] = set()
    scored: list[tuple[float, dict[str, Any]]] = []
    for key, direction in neighbor_keys:
        if key in seen:
            continue
        seen.add(key)
        neighbor = node_by_key.get(key)
        if not neighbor:
            continue
        neighbor_score = _adjacent_relevance_score(neighbor, scoring_terms)
        if neighbor_score <= 0:
            continue
        neighbor_score += min(1.0, float(neighbor.get("degree") or 0) * 0.08)
        scored.append((
            neighbor_score,
            {
                "path": neighbor.get("path"),
                "key": neighbor.get("key"),
                "title": neighbor.get("title"),
                "route": neighbor.get("route"),
                "summary": neighbor.get("summary"),
                "edge_direction": direction,
                "degree": neighbor.get("degree"),
                "token_estimate": neighbor.get("token_estimate"),
                "context_score": round(neighbor_score, 3),
            },
        ))

    scored.sort(key=lambda item: (item[0], int(item[1].get("degree") or 0)), reverse=True)
    return [item for _, item in scored[:limit]]


def _adjacent_relevance_score(node: dict[str, Any], scoring_terms: set[str]) -> float:
    keywords = {_normalize_route_term(str(term)) for term in node.get("keywords") or []}
    title = _normalize_route_term(str(node.get("title") or ""))
    path = _normalize_route_term(str(node.get("path") or ""))
    summary = _normalize_route_term(str(node.get("summary") or ""))
    score = 0.0
    for term in scoring_terms:
        if term in keywords:
            score += 2.0
        if term and term in title:
            score += 1.25
        if term and term in path:
            score += 0.75
        if term and term in summary:
            score += 0.5
    return score


def _normalize_route_term(value: str) -> str:
    return (
        value.lower()
        .replace("＿", "_")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .strip()
    )


def _route_path_quality_bonus(path: str, *, wants_news: bool, wants_logs: bool) -> float:
    normalized = path.replace("\\", "/").lower()
    bonus = 0.0
    if any(part in normalized for part in HIGH_VALUE_PATH_PARTS):
        bonus += 3.0
    if any(part in normalized for part in ("asset knowledge/", "リース審査基礎知識/", "リース知識/")):
        bonus += 2.0
    if any(part in normalized for part in ("judgment assets/", "lease-wiki-vault/")):
        bonus += 1.0
    if "05-クリップ_記事/" in normalized and not wants_news:
        bonus -= 12.0
    if any(part in normalized for part in ("improvement log/", "06-日記_作業ログ/")) and not wants_logs:
        bonus -= 8.0
    if any(part in normalized for part in ("ai chat/", "daily/", "weekly review")) and not wants_logs:
        bonus -= 5.0
    return bonus


def markdown(index: dict[str, Any], *, max_nodes: int = 80) -> str:
    summary = index.get("summary") or {}
    lines = [
        "# Obsidian Retrieval Graph",
        "",
        f"- Guardrail: `{index.get('guardrail')}`",
        f"- Vault: `{index.get('vault')}`",
        f"- Notes: {summary.get('notes', 0)}",
        f"- Edges: {summary.get('edges', 0)}",
        f"- Estimated raw tokens: {summary.get('estimated_raw_tokens', 0)}",
        f"- Estimated index tokens: {summary.get('estimated_index_tokens', 0)}",
        f"- Estimated token reduction ratio: {summary.get('estimated_token_reduction_ratio', 0)}",
        f"- Retrieval mode: `{summary.get('retrieval_mode', 'graph_preroute')}`",
        f"- Routes: {summary.get('routes', {})}",
        "",
        "## Router",
    ]
    router = index.get("router") or {}
    for route in ROUTE_ORDER:
        nodes = router.get(route) or []
        if not nodes:
            continue
        lines.append(f"### {route}")
        for node in nodes[:8]:
            lines.append(f"- `{node.get('path')}` :: {node.get('summary')}")
    lines.extend(["", "## One Line Index"])
    for node in (index.get("nodes") or [])[:max_nodes]:
        keywords = ", ".join((node.get("keywords") or [])[:8])
        lines.append(
            f"- `{node.get('path')}` | {node.get('route')} | degree={node.get('degree')} | {node.get('summary')} | {keywords}"
        )
    return "\n".join(lines) + "\n"


def _read_index(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--query", default="", help="Rank notes from an existing or freshly built index.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include private reflection notes in the sidecar index. Default keeps them out.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    index = build_index(args.vault, include_private=args.include_private)
    if args.query:
        routed = route_query(args.query, index, limit=args.limit)
        print(json.dumps({"query": args.query, "candidates": routed}, ensure_ascii=False, indent=2))
        return 0

    if args.dry_run:
        print(json.dumps(index, ensure_ascii=False, indent=2))
        return 0

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(index), encoding="utf-8")
    print(f"wrote={args.output_json}")
    print(f"wrote={args.output_md}")
    print(f"notes={index['summary']['notes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
