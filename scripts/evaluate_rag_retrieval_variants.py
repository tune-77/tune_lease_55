#!/usr/bin/env python3
"""Evaluate retrieval-side variants for the existing Obsidian RAG.

This is a report-only sidecar. It does not change ranking config, prompts,
ChromaDB contents, Obsidian notes, scoring, or judgment assets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.knowledge.vector_store import get_store

DEFAULT_EVAL_SET = REPO_ROOT / "api" / "knowledge" / "rag_eval_set.json"
DEFAULT_OUTPUT_PREFIX = REPO_ROOT / "reports" / "rag_retrieval_variants_latest"

DEFAULT_EXCLUDE_PATHS = (
    "Projects/tune_lease_55/AI Chat/",
    "Humor/",
    "05-クリップ_記事/業界リスクニュース/",
    "05-クリップ_記事/リースニュース/",
)

CATEGORY_PATH_HINTS: dict[str, tuple[str, ...]] = {
    "lease_basics": ("03-知識_業界/リース基礎知識/", "リース知識/"),
    "credit": ("03-知識_業界/リース審査実務/", "リース知識/"),
    "asset_medical": ("Projects/tune_lease_55/Asset Knowledge/医療機器/",),
    "asset_logistics": ("Projects/tune_lease_55/Asset Knowledge/",),
    "asset_manufacturing": ("Projects/tune_lease_55/Asset Knowledge/工作機械/",),
    "asset_construction": ("Projects/tune_lease_55/Asset Knowledge/建機/", "Asset Knowledge/建設・重機/", "Projects/tune_lease_55/Asset Knowledge/"),
    "asset_vehicle": ("Projects/tune_lease_55/Asset Knowledge/車両/",),
    "industry_risk": ("03-知識_業界/業種分析/", "リース知識/業種別"),
    "financial_analysis": ("03-知識_業界/リース審査実務/", "07-アーカイブ/Asset Knowledge/リース審査基礎知識/"),
    "subsidy_tax": ("リース知識/補助金", "03-知識_業界/補助金・融資/", "03-知識_業界/リース基礎知識/ファイナンスリース.md"),
    "pricing": ("リース知識/リース料率", "lease-wiki-vault/00_Core Definitions/リース契約：金利"),
}


def _display_path(hit: dict[str, Any]) -> str:
    raw = str(hit.get("file_path") or "").replace("\\", "/")
    if "/Obsidian Vault/" in raw:
        raw = raw.split("/Obsidian Vault/", 1)[1]
    return raw or str(hit.get("file_name") or hit.get("ref") or "")


def _matches_any(path: str, patterns: list[str] | tuple[str, ...]) -> bool:
    return any(pattern and pattern in path for pattern in patterns)


def _unique_by_path(hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        path = _display_path(hit)
        if path in seen:
            continue
        seen.add(path)
        selected.append(hit)
        if len(selected) >= top_k:
            break
    return selected


def _filter_excluded(hits: list[dict[str, Any]], exclude_paths: tuple[str, ...]) -> list[dict[str, Any]]:
    return [hit for hit in hits if not _matches_any(_display_path(hit), exclude_paths)]


def _category_boosted(hits: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    hints = CATEGORY_PATH_HINTS.get(category, ())
    if not hints or not category.startswith("asset_"):
        return list(hits)

    def hint_boost(path: str) -> float:
        for index, hint in enumerate(hints):
            if hint and hint in path:
                return max(0.1, 1.0 - index * 0.15)
        return 0.0

    def score(item: tuple[int, dict[str, Any]]) -> tuple[float, float, int]:
        index, hit = item
        path = _display_path(hit)
        boost = hint_boost(path)
        rank_score = float(hit.get("rank_score") or hit.get("score") or 0.0)
        return (boost, rank_score, -index)

    return [hit for _idx, hit in sorted(enumerate(hits), key=score, reverse=True)]


def _cases_by_filter(cases: list[dict[str, Any]], categories: list[str], max_cases: int | None) -> list[dict[str, Any]]:
    selected = cases
    if categories:
        wanted = set(categories)
        selected = [case for case in selected if str(case.get("category") or "") in wanted]
    if max_cases is not None:
        selected = selected[:max_cases]
    return selected


def _hits_for_variant(case: dict[str, Any], base_hits: list[dict[str, Any]], variant: str, top_k: int, exclude_paths: tuple[str, ...]) -> list[dict[str, Any]]:
    if variant == "baseline":
        return base_hits[:top_k]
    if variant == "path_dedupe":
        return _unique_by_path(base_hits, top_k)
    if variant == "hard_exclude":
        return _unique_by_path(_filter_excluded(base_hits, exclude_paths), top_k)
    if variant == "category_boost":
        boosted = _category_boosted(_filter_excluded(base_hits, exclude_paths), str(case.get("category") or ""))
        return _unique_by_path(boosted, top_k)
    raise ValueError(f"unknown variant: {variant}")


def _evaluate_variant(cases: list[dict[str, Any]], candidates_by_id: dict[str, list[dict[str, Any]]], variant: str, top_k: int, exclude_paths: tuple[str, ...]) -> dict[str, Any]:
    total = len(cases)
    hit_at_1 = 0
    hit_at_k = 0
    mrr_total = 0.0
    forbidden_hits = 0
    case_results: list[dict[str, Any]] = []

    for case in cases:
        hits = _hits_for_variant(case, candidates_by_id[str(case["id"])], variant, top_k, exclude_paths)
        paths = [_display_path(hit) for hit in hits]
        expected = list(case.get("expected_path_any") or [])
        forbidden = list(case.get("forbidden_path_any") or [])
        rank = next((i for i, path in enumerate(paths, start=1) if _matches_any(path, expected)), 0)
        bad_paths = [path for path in paths if _matches_any(path, forbidden)]
        hit_at_1 += int(rank == 1)
        hit_at_k += int(bool(rank))
        mrr_total += 1.0 / rank if rank else 0.0
        forbidden_hits += int(bool(bad_paths))
        case_results.append(
            {
                "id": case["id"],
                "rank": rank,
                "paths": paths,
                "forbidden_paths": bad_paths,
                "passed": bool(rank) and not bad_paths,
            }
        )

    return {
        "total": total,
        "hit_at_1": hit_at_1,
        "hit_at_k": hit_at_k,
        "hit_at_1_rate": hit_at_1 / total if total else 0.0,
        "hit_at_k_rate": hit_at_k / total if total else 0.0,
        "mrr": mrr_total / total if total else 0.0,
        "forbidden_cases": forbidden_hits,
        "passed": hit_at_k == total and forbidden_hits == 0,
        "cases": case_results,
    }


def evaluate(eval_set: Path, output_prefix: Path, top_k: int, candidate_k: int, categories: list[str], max_cases: int | None) -> dict[str, Any]:
    cases = _cases_by_filter(json.loads(eval_set.read_text(encoding="utf-8")), categories, max_cases)
    store = get_store()
    candidates_by_id = {
        str(case["id"]): store.search(str(case["query"]), top_k=max(candidate_k, top_k))
        for case in cases
    }
    variants = ("baseline", "path_dedupe", "hard_exclude", "category_boost")
    summaries = {
        variant: _evaluate_variant(cases, candidates_by_id, variant, top_k, DEFAULT_EXCLUDE_PATHS)
        for variant in variants
    }
    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "eval_set": eval_set.as_posix(),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "categories": categories,
        "max_cases": max_cases,
        "exclude_paths": list(DEFAULT_EXCLUDE_PATHS),
        "guardrail": "report_only_no_rag_config_no_prompt_no_scoring_no_obsidian_write",
        "summaries": summaries,
        "finding": _build_finding(summaries),
    }
    _write_reports(payload, output_prefix)
    return payload


def _metric(summary: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(summary.get("hit_at_k_rate") or 0.0),
        float(summary.get("mrr") or 0.0),
        -int(summary.get("forbidden_cases") or 0),
    )


def _build_finding(summaries: dict[str, Any]) -> dict[str, Any]:
    baseline = summaries["baseline"]
    best_name = max(summaries, key=lambda name: _metric(summaries[name]))
    best = summaries[best_name]
    return {
        "best_variant": best_name,
        "baseline": {
            "hit_at_1": baseline["hit_at_1"],
            "hit_at_k": baseline["hit_at_k"],
            "mrr": baseline["mrr"],
            "forbidden_cases": baseline["forbidden_cases"],
        },
        "best": {
            "hit_at_1": best["hit_at_1"],
            "hit_at_k": best["hit_at_k"],
            "mrr": best["mrr"],
            "forbidden_cases": best["forbidden_cases"],
        },
    }


def _write_reports(payload: dict[str, Any], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_prefix.with_suffix(".md").write_text(_render_markdown(payload), encoding="utf-8")


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# RAG Retrieval Variant Evaluation",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Eval set: `{payload['eval_set']}`",
        f"- Top K: {payload['top_k']}",
        f"- Candidate K: {payload['candidate_k']}",
        f"- Categories: {payload['categories'] or 'all'}",
        f"- Guardrail: {payload['guardrail']}",
        "",
        "## Summary",
        "",
        "| Variant | hit@1 | hit@k | MRR | Forbidden | Passed |",
        "|---|---:|---:|---:|---:|---|",
    ]
    total = next(iter(payload["summaries"].values()))["total"] if payload["summaries"] else 0
    for name, summary in payload["summaries"].items():
        lines.append(
            f"| {name} | {summary['hit_at_1']}/{total} | {summary['hit_at_k']}/{total} | "
            f"{summary['mrr']:.3f} | {summary['forbidden_cases']}/{total} | {summary['passed']} |"
        )
    lines.extend(["", "## Finding", ""])
    finding = payload["finding"]
    lines.append(f"- Best variant: `{finding['best_variant']}`")
    lines.append(f"- Exclude paths: `{', '.join(payload['exclude_paths'])}`")
    lines.extend(["", "## Misses / Noise", ""])
    for name, summary in payload["summaries"].items():
        bad = [case for case in summary.get("cases", []) if not case.get("passed")]
        if not bad:
            continue
        lines.append(f"### {name}")
        lines.append("")
        for case in bad:
            lines.append(f"- `{case['id']}` rank={case.get('rank') or '-'} forbidden={case.get('forbidden_paths') or []}")
            for idx, path in enumerate(case.get("paths") or [], start=1):
                lines.append(f"{idx}. {path}")
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    payload = evaluate(
        eval_set=args.eval_set.expanduser(),
        output_prefix=args.output_prefix.expanduser(),
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        categories=list(args.category),
        max_cases=args.max_cases,
    )
    finding = payload["finding"]
    print(
        f"best={finding['best_variant']} "
        f"baseline_hit@k={finding['baseline']['hit_at_k']} "
        f"best_hit@k={finding['best']['hit_at_k']} "
        f"baseline_forbidden={finding['baseline']['forbidden_cases']} "
        f"best_forbidden={finding['best']['forbidden_cases']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
