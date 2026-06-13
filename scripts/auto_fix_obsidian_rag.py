#!/usr/bin/env python3
"""Safely tune bounded Obsidian RAG ranking settings from the eval set."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.knowledge.vector_store import KnowledgeVectorStore, load_ranking_config
from scripts.evaluate_obsidian_rag import DEFAULT_EVAL_SET, evaluate_cases

DEFAULT_CONFIG = REPO_ROOT / "config" / "rag_ranking.json"
DEFAULT_REPORT = REPO_ROOT / "reports" / "rag_auto_fix_latest.json"

_PREFERRED_PREFIXES = (
    "リース知識/",
    "03-知識_業界/",
    "Projects/tune_lease_55/Asset Knowledge/",
    "Projects/tune_lease_55/Asset Finance/",
    "Projects/tune_lease_55/Cases/",
    "Projects/tune_lease_55/Feedback/",
    "Projects/tune_lease_55/Research/",
    "Projects/tune_lease_55/",
)
_PENALTY_PREFIXES = (
    "05-クリップ_記事/リースニュース/",
    "リースニュース/",
    "07-アーカイブ/",
    "Projects/tune_lease_55/AI Chat/",
    "Daily/",
    "Clippings/",
    "Humor/",
)


def _load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("RAG eval set must be a list")
    return data


def _preferred_prefix(path: str) -> str | None:
    return next((prefix for prefix in _PREFERRED_PREFIXES if path.startswith(prefix)), None)


def _penalty_prefix(pattern: str) -> str | None:
    return next(
        (prefix for prefix in _PENALTY_PREFIXES if pattern.startswith(prefix) or prefix.startswith(pattern)),
        None,
    )


def build_candidates(config: dict, cases: list[dict], baseline: dict) -> list[tuple[str, dict]]:
    cases_by_id = {str(case.get("id")): case for case in cases}
    candidates: list[tuple[str, dict]] = []
    targeted = copy.deepcopy(config)
    changed = False

    for result in baseline["cases"]:
        if result["rank"] == 0:
            case = cases_by_id[result["id"]]
            for expected in case.get("expected_path_any") or []:
                prefix = _preferred_prefix(str(expected))
                if not prefix:
                    continue
                boosts = targeted.setdefault("preferred_path_boosts", {})
                boosts[prefix] = round(min(0.20, float(boosts.get(prefix, 0.0)) + 0.02), 4)
                changed = True
        if result["forbidden_paths"]:
            case = cases_by_id[result["id"]]
            for forbidden in case.get("forbidden_path_any") or []:
                prefix = _penalty_prefix(str(forbidden))
                if not prefix:
                    continue
                penalties = targeted.setdefault("low_priority_path_penalties", {})
                penalties[prefix] = round(min(0.50, float(penalties.get(prefix, 0.0)) + 0.05), 4)
                changed = True

    if any("/lease-wiki-vault/" in path for result in baseline["cases"] for path in result["forbidden_paths"]):
        targeted["sync_copy_penalty"] = round(
            min(0.70, float(targeted.get("sync_copy_penalty", 0.35)) + 0.05),
            4,
        )
        changed = True

    if changed:
        targeted["keyword_pool_multiplier"] = min(
            8,
            int(targeted.get("keyword_pool_multiplier", 4)) + 1,
        )
        candidates.append(("targeted_failure_fix", targeted))

    for multiplier in (5, 6):
        candidate = copy.deepcopy(config)
        candidate["keyword_pool_multiplier"] = max(
            multiplier,
            int(candidate.get("keyword_pool_multiplier", 4)),
        )
        candidates.append((f"keyword_pool_{multiplier}", candidate))

    broad = copy.deepcopy(config)
    boosts = broad.setdefault("preferred_path_boosts", {})
    for prefix in ("リース知識/", "03-知識_業界/"):
        boosts[prefix] = round(min(0.20, float(boosts.get(prefix, 0.0)) + 0.02), 4)
    broad["keyword_pool_multiplier"] = min(8, int(broad.get("keyword_pool_multiplier", 4)) + 1)
    candidates.append(("curated_knowledge_boost", broad))

    unique: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for name, candidate in candidates:
        key = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        if key not in seen and candidate != config:
            seen.add(key)
            unique.append((name, candidate))
    return unique[:4]


def _objective(summary: dict) -> tuple[int, int, int, float]:
    return (
        int(summary["hit_at_k"]),
        -int(summary["forbidden_cases"]),
        int(summary["hit_at_1"]),
        float(summary["mrr"]),
    )


def is_safe_improvement(baseline: dict, candidate: dict) -> bool:
    baseline_passed = {item["id"] for item in baseline["cases"] if item["passed"]}
    candidate_passed = {item["id"] for item in candidate["cases"] if item["passed"]}
    if not baseline_passed.issubset(candidate_passed):
        return False
    if candidate["forbidden_cases"] > baseline["forbidden_cases"]:
        return False
    return _objective(candidate) > _objective(baseline)


def run_auto_fix(
    *,
    cases: list[dict],
    config: dict,
    evaluate_config: Callable[[dict], dict],
) -> dict:
    baseline = evaluate_config(config)
    report = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "healthy" if baseline["passed"] else "no_safe_improvement",
        "baseline": baseline,
        "trials": [],
        "selected": None,
    }
    if baseline["passed"]:
        return report

    best_summary = baseline
    best_config = None
    best_name = None
    for name, candidate_config in build_candidates(config, cases, baseline):
        summary = evaluate_config(candidate_config)
        safe = is_safe_improvement(baseline, summary)
        report["trials"].append({"name": name, "safe_improvement": safe, "summary": summary})
        if safe and _objective(summary) > _objective(best_summary):
            best_summary = summary
            best_config = candidate_config
            best_name = name

    if best_config is not None:
        report["status"] = "applied"
        report["selected"] = {
            "name": best_name,
            "config": best_config,
            "summary": best_summary,
        }
    return report


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = _load_cases(args.eval_set)
    config = load_ranking_config(str(args.config))
    store = KnowledgeVectorStore(ranking_config=config)

    def evaluate_config(candidate_config: dict) -> dict:
        store.set_ranking_config(candidate_config)
        return evaluate_cases(
            cases,
            lambda query, limit: store.search(query, top_k=limit),
            args.top_k,
        )

    report = run_auto_fix(cases=cases, config=config, evaluate_config=evaluate_config)
    report["eval_set"] = str(args.eval_set)
    report["config_path"] = str(args.config)
    report["dry_run"] = args.dry_run

    selected = report.get("selected")
    if selected and not args.dry_run:
        _atomic_write_json(args.config, selected["config"])
    _atomic_write_json(args.report, report)

    baseline = report["baseline"]
    print(
        "[rag-auto-fix] "
        f"status={report['status']} "
        f"hit@{args.top_k}={baseline['hit_at_k']}/{baseline['total']} "
        f"forbidden={baseline['forbidden_cases']}"
    )


if __name__ == "__main__":
    main()
