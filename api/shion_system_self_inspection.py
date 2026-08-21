"""Read-only self-inspection tools for Shion's system.

These tools let Shion search project implementation notes, docs, skills,
tests, and generated reports, then propose small improvement focus items.
They deliberately avoid secrets, databases, raw logs, and write operations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SEARCH_ROOTS = {
    "api": _REPO_ROOT / "api",
    "scripts": _REPO_ROOT / "scripts",
    "docs": _REPO_ROOT / "docs",
    "reports": _REPO_ROOT / "reports",
    "skills": _REPO_ROOT / ".agents" / "skills",
    "tests": _REPO_ROOT / "tests",
    "frontend": _REPO_ROOT / "frontend" / "src",
    "planning": _REPO_ROOT / "planning",
    "root_docs": _REPO_ROOT,
}

_ROOT_DOC_NAMES = {
    "README.md",
    "README.en.md",
    "AGENTS.md",
    "CLAUDE.md",
    "HEARTBEAT.md",
    "OBSIDIAN_WIKI_WORKFLOW.md",
    "CLOUD_RUN.md",
}

_ALLOWED_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".ts", ".tsx", ".sh", ".yaml", ".yml"}
_EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".streamlit",
    "logs",
    "_archive",
    "easyocr_models",
    "tune_lease_55_wt_cibundle",
}
_SECRET_MARKERS = ("secret", "secrets", "token", "apikey", "api_key", "password", ".env")
_MAX_FILE_BYTES = 320_000


def _clean_text(value: Any, *, maximum: int = 1000) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:maximum]


def _clamp_limit(limit: int | None, *, default: int = 8, maximum: int = 20) -> int:
    try:
        raw = int(limit if limit is not None else default)
    except (TypeError, ValueError):
        raw = default
    return max(1, min(maximum, raw))


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _is_allowed_path(path: Path) -> bool:
    rel_parts = set(path.relative_to(_REPO_ROOT).parts) if path.is_relative_to(_REPO_ROOT) else set(path.parts)
    lowered = str(path).lower()
    if rel_parts & _EXCLUDED_PARTS:
        return False
    if any(marker in lowered for marker in _SECRET_MARKERS):
        return False
    if path.parent == _REPO_ROOT and path.name not in _ROOT_DOC_NAMES:
        return False
    if path.suffix not in _ALLOWED_SUFFIXES:
        return False
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return False
    except OSError:
        return False
    return True


def _iter_candidate_files(area: str = "all") -> list[Path]:
    selected = []
    if area and area != "all":
        root = _SEARCH_ROOTS.get(area)
        if root:
            selected = [root]
    if not selected:
        selected = [root for key, root in _SEARCH_ROOTS.items() if key != "root_docs"]
        selected.append(_REPO_ROOT)
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in selected:
        if not root.exists():
            continue
        if root == _REPO_ROOT:
            for name in _ROOT_DOC_NAMES:
                path = _REPO_ROOT / name
                if path.exists() and _is_allowed_path(path) and path not in seen:
                    paths.append(path)
                    seen.add(path)
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path in seen:
                continue
            if _is_allowed_path(path):
                paths.append(path)
                seen.add(path)
    return paths


def _snippet(text: str, query_terms: list[str], *, width: int = 220) -> str:
    lowered = text.lower()
    positions = [lowered.find(term.lower()) for term in query_terms if term and lowered.find(term.lower()) >= 0]
    pos = min(positions) if positions else 0
    start = max(0, pos - width // 2)
    return _clean_text(text[start:start + width], maximum=width)


def search_shion_system_context(query: str, area: str = "all", limit: int = 8) -> dict[str, Any]:
    """Search Shion system docs, code, reports, skills, and tests.

    Args:
        query: Search words about Shion, memory, RAG, ADK tools, pipeline, UI,
            reports, or lease screening system behavior.
        area: all, api, scripts, docs, reports, skills, tests, frontend,
            planning, or root_docs.
        limit: Maximum matches to return, clamped to 1..20.

    Returns:
        Bounded read-only search results with path, line number, and short
        snippets. Secret-like paths and large/raw files are skipped.
    """
    clean_query = _clean_text(query, maximum=200)
    max_items = _clamp_limit(limit, default=8, maximum=20)
    terms = [term for term in re.split(r"\s+", clean_query) if len(term) >= 2]
    if not terms:
        return {
            "mode": "shion_system_context_search",
            "query": clean_query,
            "area": area,
            "count": 0,
            "results": [],
            "guardrail": "read_only_safe_paths_no_secrets_no_raw_db_no_logs",
        }

    results: list[dict[str, Any]] = []
    for path in _iter_candidate_files(area):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lowered = text.lower()
        score = sum(lowered.count(term.lower()) for term in terms)
        if score <= 0:
            continue
        first_line = 1
        for idx, line in enumerate(text.splitlines(), start=1):
            line_l = line.lower()
            if any(term.lower() in line_l for term in terms):
                first_line = idx
                break
        results.append({
            "path": _safe_rel(path),
            "line": first_line,
            "score": score,
            "snippet": _snippet(text, terms),
        })
    results.sort(key=lambda item: (item["score"], -len(item["path"])), reverse=True)
    return {
        "mode": "shion_system_context_search",
        "query": clean_query,
        "area": area if area in _SEARCH_ROOTS or area == "all" else "all",
        "count": min(len(results), max_items),
        "results": results[:max_items],
        "searched_roots": sorted(_SEARCH_ROOTS.keys()),
        "guardrail": "read_only_safe_paths_no_secrets_no_raw_db_no_logs",
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _latest_report_briefs(limit: int = 8) -> list[dict[str, str]]:
    names = [
        "recursive_self_improvement_latest.json",
        "memory_engineering_latest.json",
        "instruction_debt_latest.json",
        "obsidian_environment_monitor_latest.json",
        "shion_memory_effect_latest.json",
        "agent_action_ledger_latest.md",
        "shion_growth_brief_latest.md",
        "obsidian_curator_latest.md",
    ]
    briefs: list[dict[str, str]] = []
    for name in names:
        path = _REPO_ROOT / "reports" / name
        if not path.exists() or not _is_allowed_path(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if path.suffix == ".json":
            data = _read_json(path)
            summary = data.get("summary") or data.get("status") or data.get("generated_at") or text[:500]
            brief = _clean_text(summary, maximum=500)
        else:
            brief = _clean_text(text, maximum=500)
        briefs.append({"path": _safe_rel(path), "brief": brief})
        if len(briefs) >= limit:
            break
    return briefs


def _score_focus_proposal(proposal: dict[str, Any]) -> dict[str, str]:
    proposal_type = str(proposal.get("type") or "")
    evidence = proposal.get("evidence")
    evidence_count = len(evidence) if isinstance(evidence, list) else 1 if evidence else 0
    priority = str(proposal.get("priority") or "low")

    if proposal_type == "review_existing_reports":
        impact = "medium"
        risk = "low"
        effort = "low"
        evidence_strength = "high" if evidence_count >= 2 else "medium"
    elif proposal_type == "inspect_related_system_context":
        impact = "medium"
        risk = "low"
        effort = "medium"
        evidence_strength = "medium" if evidence_count else "low"
    elif proposal_type == "connect_agentic_skill_observations":
        impact = "medium" if priority != "low" else "low"
        risk = "low"
        effort = "low"
        evidence_strength = "high" if evidence_count else "medium"
    else:
        impact = "low"
        risk = "low"
        effort = "low"
        evidence_strength = "low"

    if risk == "low" and effort == "low" and evidence_strength in {"medium", "high"}:
        recommendation = "today"
    elif risk == "low" and evidence_strength != "low":
        recommendation = "next"
    else:
        recommendation = "observe"

    return {
        "impact": impact,
        "risk": risk,
        "effort": effort,
        "evidence": evidence_strength,
        "recommendation": recommendation,
    }


def _attach_focus_scores(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    recommendation_rank = {"today": 0, "next": 1, "observe": 2}
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    for proposal in proposals:
        item = dict(proposal)
        item["score"] = _score_focus_proposal(item)
        scored.append(item)
    scored.sort(
        key=lambda item: (
            recommendation_rank.get(str((item.get("score") or {}).get("recommendation")), 9),
            priority_rank.get(str(item.get("priority") or "low"), 9),
        )
    )
    return scored


def propose_shion_system_improvement_focus(question: str = "", limit: int = 3) -> dict[str, Any]:
    """Propose read-only improvement focus items for the Shion system.

    Args:
        question: Optional user concern or theme to search before proposing.
        limit: Maximum proposals to return, clamped to 1..3.

    Returns:
        Small human-review proposals grounded in system reports/search results.
        It does not edit files, schedule pipelines, promote memory, or change
        prompts/scoring/RAG.
    """
    max_items = _clamp_limit(limit, default=3, maximum=3)
    theme = _clean_text(question, maximum=200)
    search_query = theme or "紫苑 memory RAG 判断資産 improvement agentic skill"
    search = search_shion_system_context(search_query, area="all", limit=8)
    report_briefs = _latest_report_briefs(limit=8)
    proposals: list[dict[str, Any]] = []

    if report_briefs:
        proposals.append({
            "priority": "medium",
            "type": "review_existing_reports",
            "title": "最新レポートを根拠に改善焦点を選ぶ",
            "reason": "紫苑システムには既に複数の自己観測レポートがあり、まずここを根拠にするとノイズが少ない。",
            "evidence": report_briefs[:3],
            "human_action": "improvement-log または該当reportを見て、手動レビュー対象を1つ選ぶ",
        })

    results = search.get("results") if isinstance(search.get("results"), list) else []
    if results:
        proposals.append({
            "priority": "medium",
            "type": "inspect_related_system_context",
            "title": "関連コード/ドキュメントを横断して確認する",
            "reason": f"'{search_query}' に関連する実装・ドキュメントが見つかりました。変更前に構造を読めます。",
            "evidence": results[:5],
            "human_action": "紫苑に検索結果の要約と改善候補の切り分けをさせる",
        })

    try:
        from api.agentic_skill_usage import propose_agentic_skill_next_actions

        agentic = propose_agentic_skill_next_actions(limit=3)
        agentic_proposals = agentic.get("proposals") if isinstance(agentic.get("proposals"), list) else []
    except Exception:
        agentic_proposals = []
    if agentic_proposals:
        proposals.append({
            "priority": "low",
            "type": "connect_agentic_skill_observations",
            "title": "agentic skillレビュー箱の提案も見る",
            "reason": "紫苑が自分で作った候補・未レビュー滞留・観測導線の警告は、改善焦点の早期シグナルになる。",
            "evidence": agentic_proposals[:3],
            "human_action": "紫苑ADKレビュー箱で候補を採用・修正・保留・却下する",
        })

    if not proposals:
        proposals.append({
            "priority": "low",
            "type": "continue_observation",
            "title": "読み取り観測を続ける",
            "reason": "現時点で強い改善焦点は検出できませんでした。",
            "evidence": [],
            "human_action": "具体的な関心テーマを指定して再検索する",
        })

    scored_proposals = _attach_focus_scores(proposals)
    return {
        "mode": "shion_system_improvement_focus",
        "question": theme,
        "search": search,
        "report_briefs": report_briefs,
        "scoring_policy": {
            "impact": "効いた時の効果",
            "risk": "触る危険度",
            "effort": "確認/実装の重さ",
            "evidence": "根拠の強さ",
            "recommendation": "today | next | observe",
        },
        "proposals": scored_proposals[:max_items],
        "guardrail": "proposal_only_read_only_no_pipeline_no_prompt_change_no_memory_promotion_no_scoring_no_rag",
    }


SHION_SYSTEM_SELF_INSPECTION_TOOLS = [
    search_shion_system_context,
    propose_shion_system_improvement_focus,
]
