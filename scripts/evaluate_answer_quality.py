#!/usr/bin/env python3
"""Generate or score a small, deterministic lease-answer quality benchmark."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EVAL_SET = REPO_ROOT / "api" / "knowledge" / "answer_eval_set.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "answer_quality_latest.json"

UNCERTAINTY_TERMS = (
    "要確認",
    "確認が必要",
    "場合がある",
    "可能性",
    "公募要領",
    "個別",
    "一概に",
    "断定でき",
    "変動",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def score_answer(case: dict[str, Any], answer: str, source_paths: list[str] | None = None) -> dict[str, Any]:
    normalized = _normalize(answer)
    concept_results = []
    for aliases in case.get("required_concepts") or []:
        matched = [alias for alias in aliases if _normalize(alias) in normalized]
        concept_results.append(
            {
                "aliases": aliases,
                "matched": bool(matched),
                "matched_alias": matched[0] if matched else "",
            }
        )
    concept_total = len(concept_results)
    concept_hits = sum(1 for item in concept_results if item["matched"])
    concept_score = concept_hits / concept_total * 70 if concept_total else 70.0

    forbidden_hits = [
        claim
        for claim in case.get("forbidden_claims") or []
        if _normalize(claim) in normalized
    ]
    safety_score = 20.0 if not forbidden_hits else 0.0

    uncertainty_required = bool(case.get("require_uncertainty"))
    uncertainty_present = any(_normalize(term) in normalized for term in UNCERTAINTY_TERMS)
    uncertainty_score = 10.0 if not uncertainty_required or uncertainty_present else 0.0

    total_score = round(concept_score + safety_score + uncertainty_score, 1)
    uncertainty_ok = not uncertainty_required or uncertainty_present
    return {
        "id": case["id"],
        "query": case["query"],
        "answer": answer,
        "source_paths": source_paths or [],
        "score": total_score,
        "passed": total_score >= 80.0 and not forbidden_hits and uncertainty_ok,
        "concept_hits": concept_hits,
        "concept_total": concept_total,
        "concept_results": concept_results,
        "forbidden_hits": forbidden_hits,
        "uncertainty_required": uncertainty_required,
        "uncertainty_present": uncertainty_present,
    }


def evaluate_answers(
    cases: list[dict[str, Any]],
    answers: dict[str, dict[str, Any] | str],
) -> dict[str, Any]:
    results = []
    for case in cases:
        raw = answers.get(case["id"], "")
        if isinstance(raw, dict):
            answer = str(raw.get("answer") or "")
            source_paths = [str(path) for path in raw.get("source_paths") or []]
        else:
            answer = str(raw or "")
            source_paths = []
        results.append(score_answer(case, answer, source_paths))

    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    avg_score = sum(float(result["score"]) for result in results) / total if total else 0.0
    concept_hits = sum(int(result["concept_hits"]) for result in results)
    concept_total = sum(int(result["concept_total"]) for result in results)
    forbidden_cases = sum(1 for result in results if result["forbidden_hits"])
    uncertainty_misses = sum(
        1
        for result in results
        if result["uncertainty_required"] and not result["uncertainty_present"]
    )
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total * 100, 1) if total else 0.0,
        "average_score": round(avg_score, 1),
        "concept_coverage": round(concept_hits / concept_total * 100, 1) if concept_total else 0.0,
        "forbidden_cases": forbidden_cases,
        "uncertainty_misses": uncertainty_misses,
        "cases": results,
    }


def generate_answers(
    cases: list[dict[str, Any]],
    *,
    search_fn: Callable[[str, int], list[dict]],
    chat_fn: Callable[[str, list[dict], str], str],
) -> dict[str, dict[str, Any]]:
    system_prompt = """あなたはリース実務の回答品質評価用アシスタントです。
与えられた参照ナレッジだけを根拠に、日本語で簡潔に回答してください。
結論、重要な確認事項、注意点を含めます。審査質問では、抽象論だけでなく確認資料、
数値推移、受注・入金、支払原資、物件の流通性、保守・部品供給、撤去費、法規制のうち
質問に関係する実務観点を具体的に挙げてください。
参照情報だけで断定できない場合は「要確認」または「場合がある」と明示してください。
自動承認・自動否決はせず、根拠のない数値を作らないでください。"""
    answers: dict[str, dict[str, Any]] = {}
    for case in cases:
        hits = search_fn(case["query"], 5)
        context_parts = []
        source_paths = []
        for hit in hits:
            path = str(hit.get("file_path") or hit.get("file_name") or hit.get("ref") or "")
            text = str(hit.get("text") or "")[:900]
            if path:
                source_paths.append(path)
            if text:
                context_parts.append(f"参照: {path}\n{text}")
        user_message = f"質問: {case['query']}\n\n" + "\n\n---\n\n".join(context_parts)
        answers[case["id"]] = {
            "answer": chat_fn(system_prompt, [], user_message).strip(),
            "source_paths": source_paths,
        }
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases = json.loads(args.eval_set.read_text(encoding="utf-8"))
    if args.generate:
        from api.chat_memory import call_gemini_chat
        from api.knowledge.vector_store import get_store

        store = get_store()
        answers = generate_answers(
            cases,
            search_fn=lambda query, limit: store.search(query, top_k=limit),
            chat_fn=call_gemini_chat,
        )
    elif args.responses:
        answers = json.loads(args.responses.read_text(encoding="utf-8"))
    else:
        raise SystemExit("--generate または --responses が必要です")

    summary = evaluate_answers(cases, answers)
    summary["eval_set"] = str(args.eval_set)
    summary["generated"] = bool(args.generate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"answer_quality pass={summary['passed']}/{summary['total']} "
        f"avg={summary['average_score']} "
        f"concept={summary['concept_coverage']}% "
        f"forbidden={summary['forbidden_cases']} "
        f"uncertainty_miss={summary['uncertainty_misses']}"
    )
    raise SystemExit(0 if summary["passed"] == summary["total"] else 1)


if __name__ == "__main__":
    main()
