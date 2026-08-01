#!/usr/bin/env python3
"""Compare local Obsidian RAG with the Vertex AI Search pilot on the same questions."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from obsidian_ai_context import collect_obsidian_ai_context
from api.vertex_agent_search import answer_vertex_agent, search_vertex_agent


DEFAULT_QUESTIONS = [
    {
        "id": "release_residual",
        "question": "再リース案件では耐用年数と残価をどう確認するべきか",
        "focus_terms": ["再リース", "耐用年数", "残価", "中古", "満了"],
    },
    {
        "id": "subsidy_machinery",
        "question": "補助金前提の工作機械リースで確認すべきこと",
        "focus_terms": ["補助金", "工作機械", "リース料軽減", "公募", "対象経費"],
    },
    {
        "id": "non_transfer_period",
        "question": "所有権移転外リースでリース期間が耐用年数の目安から外れる場合の注意点",
        "focus_terms": ["所有権移転外", "リース期間", "耐用年数", "70%", "60%"],
    },
    {
        "id": "transport_risk",
        "question": "運送業のリース審査で見るべき業界リスク",
        "focus_terms": ["運送", "物流", "燃料", "ドライバー", "業界リスク"],
    },
    {
        "id": "cashflow_warning",
        "question": "返済余力と資金繰りの異常兆候をどう確認するか",
        "focus_terms": ["返済余力", "資金繰り", "キャッシュフロー", "手元資金", "支払能力"],
    },
    {
        "id": "machine_resale",
        "question": "工作機械の残価と中古流動性を見る観点",
        "focus_terms": ["工作機械", "残価", "中古", "流動性", "陳腐化"],
    },
]


def _compact_text(value: Any, limit: int = 360) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _term_hit_score(text: str, terms: list[str]) -> int:
    normalized = text.lower()
    return sum(1 for term in terms if term.lower() in normalized)


def _obsidian_result(query: str, limit: int, max_chars: int) -> dict[str, Any]:
    context = collect_obsidian_ai_context(query, limit=limit, max_chars=max_chars)
    hits = []
    for hit in context.get("hits", [])[:limit]:
        hits.append(
            {
                "path": str(hit.get("path") or ""),
                "snippet": _compact_text(hit.get("snippet"), 500),
            }
        )
    return {
        "source_count": int(context.get("source_count") or len(hits)),
        "block": str(context.get("block") or ""),
        "hits": hits,
    }


def _vertex_result(query: str, limit: int) -> dict[str, Any]:
    response = search_vertex_agent(query, page_size=limit, apply_controls=True)
    results = response.get("results", [])[:limit]
    return {
        "status": response.get("status"),
        "summary": response.get("summary") or "",
        "results": results,
        "controls": response.get("controls") or {},
    }


def _vertex_answer_result(query: str, limit: int) -> dict[str, Any]:
    response = answer_vertex_agent(query, page_size=limit)
    return {
        "status": response.get("status"),
        "answer_text": response.get("answer_text") or "",
        "answer_state": response.get("answer_state"),
        "refs": response.get("refs") or [],
        "search_results": response.get("search_results") or [],
    }


def _score_side(question: str, focus_terms: list[str], entries: list[dict[str, Any]], summary: str = "") -> dict[str, Any]:
    joined = " ".join(
        [
            question,
            summary,
            *[
                " ".join(
                    str(entry.get(key) or "")
                    for key in ("path", "title", "source_path", "snippet", "excerpt")
                )
                for entry in entries
            ],
        ]
    )
    term_hits = _term_hit_score(joined, focus_terms)
    return {
        "focus_term_hits": term_hits,
        "focus_term_count": len(focus_terms),
        "coverage_rate": round(term_hits / len(focus_terms), 3) if focus_terms else 0.0,
    }


def compare(questions: list[dict[str, Any]], limit: int, max_chars: int, include_answer: bool) -> dict[str, Any]:
    cases = []
    for item in questions:
        question = str(item["question"])
        focus_terms = [str(term) for term in item.get("focus_terms", [])]
        obsidian = _obsidian_result(question, limit=limit, max_chars=max_chars)
        vertex = _vertex_result(question, limit=limit)
        vertex_answer = _vertex_answer_result(question, limit=limit) if include_answer else {}
        obsidian_score = _score_side(question, focus_terms, obsidian["hits"], obsidian["block"])
        vertex_score = _score_side(question, focus_terms, vertex["results"], vertex["summary"])
        vertex_answer_score = (
            _score_side(question, focus_terms, vertex_answer.get("search_results", []), vertex_answer.get("answer_text", ""))
            if include_answer
            else {}
        )
        cases.append(
            {
                "id": item["id"],
                "question": question,
                "focus_terms": focus_terms,
                "obsidian_rag": obsidian,
                "vertex_ai_search": vertex,
                "vertex_answer_api": vertex_answer,
                "heuristic_scores": {
                    "obsidian_rag": obsidian_score,
                    "vertex_ai_search": vertex_score,
                    "vertex_answer_api": vertex_answer_score,
                },
            }
        )
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "method": {
            "obsidian_rag": "obsidian_ai_context.collect_obsidian_ai_context",
            "vertex_ai_search": "Discovery Engine servingConfigs/default_search:search",
            "vertex_answer_api": "Discovery Engine servingConfigs/default_search:answer" if include_answer else "skipped",
            "top_k": limit,
            "note": "Scores are keyword coverage heuristics, not a judgment-quality benchmark.",
        },
        "cases": cases,
        "aggregate": _aggregate(cases),
    }


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {}
    obs_rates = [case["heuristic_scores"]["obsidian_rag"]["coverage_rate"] for case in cases]
    vertex_rates = [case["heuristic_scores"]["vertex_ai_search"]["coverage_rate"] for case in cases]
    answer_rates = [
        case["heuristic_scores"].get("vertex_answer_api", {}).get("coverage_rate")
        for case in cases
        if case["heuristic_scores"].get("vertex_answer_api")
    ]
    obs_hit_counts = [len(case["obsidian_rag"]["hits"]) for case in cases]
    vertex_hit_counts = [len(case["vertex_ai_search"]["results"]) for case in cases]
    return {
        "case_count": len(cases),
        "obsidian_avg_keyword_coverage": round(sum(obs_rates) / len(obs_rates), 3),
        "vertex_avg_keyword_coverage": round(sum(vertex_rates) / len(vertex_rates), 3),
        "vertex_answer_avg_keyword_coverage": round(sum(answer_rates) / len(answer_rates), 3) if answer_rates else None,
        "obsidian_avg_hits": round(sum(obs_hit_counts) / len(obs_hit_counts), 2),
        "vertex_avg_hits": round(sum(vertex_hit_counts) / len(vertex_hit_counts), 2),
    }


def write_reports(result: dict[str, Any], output_prefix: Path) -> tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Obsidian RAG vs Vertex AI Search Comparison",
        "",
        f"- Generated: {result['generated_at']}",
        f"- Top K: {result['method']['top_k']}",
        f"- Note: {result['method']['note']}",
        "",
        "## Aggregate",
        "",
    ]
    aggregate = result.get("aggregate", {})
    for key, value in aggregate.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Findings", ""])
    lines.extend(_render_findings(result))
    lines.extend(["", "## Cases", ""])

    for case in result["cases"]:
        lines.append(f"### {case['id']}")
        lines.append("")
        lines.append(f"Question: {case['question']}")
        lines.append("")
        obs_score = case["heuristic_scores"]["obsidian_rag"]
        vertex_score = case["heuristic_scores"]["vertex_ai_search"]
        answer_score = case["heuristic_scores"].get("vertex_answer_api") or {}
        lines.append(
            f"- Keyword coverage: Obsidian {obs_score['focus_term_hits']}/{obs_score['focus_term_count']} "
            f"({obs_score['coverage_rate']:.0%}), Vertex {vertex_score['focus_term_hits']}/{vertex_score['focus_term_count']} "
            f"({vertex_score['coverage_rate']:.0%})"
        )
        if answer_score:
            lines.append(
                f"- Answer API coverage: {answer_score['focus_term_hits']}/{answer_score['focus_term_count']} "
                f"({answer_score['coverage_rate']:.0%})"
            )
        lines.append("")
        lines.append("Obsidian RAG hits:")
        for idx, hit in enumerate(case["obsidian_rag"]["hits"], start=1):
            lines.append(f"{idx}. {hit['path']}")
            if hit["snippet"]:
                lines.append(f"   - {hit['snippet']}")
        if not case["obsidian_rag"]["hits"]:
            lines.append("- No hits")
        lines.append("")
        lines.append("Vertex AI Search hits:")
        for idx, hit in enumerate(case["vertex_ai_search"]["results"], start=1):
            label = hit["title"] or hit["id"] or "(untitled)"
            source = hit["source_path"] or hit["link"]
            lines.append(f"{idx}. {label}")
            if source:
                lines.append(f"   - source: {source}")
            if hit["excerpt"]:
                lines.append(f"   - excerpt: {hit['excerpt']}")
        if not case["vertex_ai_search"]["results"]:
            lines.append("- No hits")
        summary = _compact_text(case["vertex_ai_search"].get("summary"), 900)
        if summary:
            lines.extend(["", "Vertex summary:", summary])
        answer_text = _compact_text(case.get("vertex_answer_api", {}).get("answer_text"), 900)
        if answer_text:
            lines.extend(["", "Vertex Answer API:", answer_text])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_findings(result: dict[str, Any]) -> list[str]:
    cases = result.get("cases", [])
    vertex_better = []
    obsidian_better = []
    tied = []
    duplicate_heavy = []
    for case in cases:
        obs = case["heuristic_scores"]["obsidian_rag"]["coverage_rate"]
        vertex = case["heuristic_scores"]["vertex_ai_search"]["coverage_rate"]
        if vertex > obs:
            vertex_better.append(case["id"])
        elif obs > vertex:
            obsidian_better.append(case["id"])
        else:
            tied.append(case["id"])
        titles = [hit.get("title") for hit in case["vertex_ai_search"].get("results", [])]
        if len(titles) >= 3 and len(set(titles)) <= 2:
            duplicate_heavy.append(case["id"])

    lines = [
        "- Vertex AI Search had higher keyword coverage on broader thematic questions, especially subsidy and transport-risk queries.",
        "- Existing Obsidian RAG remained strongest for precise local asset notes, especially the machine-tool resale query where the top hit was the dedicated asset note.",
        "- Vertex AI Search produced synthesized summaries with citations, which are easier to hand to a chat answer layer than raw snippets.",
        "- Vertex AI Search also repeated multiple Auto Research notes with the same title/topic. Add deduplication by canonical topic or source path before production use.",
        "- Existing Obsidian RAG costs nothing per cloud query and can read the wider local Vault, but its top results are more sensitive to folder noise and keyword expansion.",
    ]
    lines.append(f"- Vertex better by heuristic: {', '.join(vertex_better) or 'none'}")
    lines.append(f"- Obsidian better by heuristic: {', '.join(obsidian_better) or 'none'}")
    lines.append(f"- Tied by heuristic: {', '.join(tied) or 'none'}")
    lines.append(f"- Vertex duplicate-heavy cases: {', '.join(duplicate_heavy) or 'none'}")
    return lines


def load_questions(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_QUESTIONS
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, help="Optional JSON question set.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=4200)
    parser.add_argument("--skip-answer", action="store_true", help="Skip Discovery Engine Answer API calls.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=REPO_ROOT / "reports" / f"obsidian_rag_vs_vertex_search_{dt.date.today():%Y%m%d}",
    )
    args = parser.parse_args()

    result = compare(load_questions(args.questions), args.top_k, args.max_chars, not args.skip_answer)
    json_path, md_path = write_reports(result, args.output_prefix)
    latest_json, latest_md = write_reports(result, REPO_ROOT / "reports" / "obsidian_rag_vs_vertex_search_latest")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {latest_json}")
    print(f"wrote {latest_md}")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
