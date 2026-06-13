#!/usr/bin/env python3
"""Iteratively improve failed answer-eval cases with bounded RAG retries."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_answer_quality import (
    DEFAULT_EVAL_SET,
    evaluate_answers,
    score_answer,
)

DEFAULT_REPORT = REPO_ROOT / "reports" / "answer_quality_auto_improve_latest.json"

_SYSTEM_PROMPT = """あなたはリース審査実務の回答改善アシスタントです。
参照ナレッジを根拠に、質問へ日本語で簡潔かつ実務的に回答してください。
結論だけでなく、審査担当者が確認する資料・数値・契約条件・物件回収条件を具体化します。
財務質問では推移、回転期間、受注・入金、返済原資を確認します。
物件質問では中古市場、流通性、保守・部品供給、撤去費、法規制を確認します。
業種質問では受注残、稼働率、季節性、回収サイトを確認します。
参照情報だけで断定できない場合は「要確認」または「場合がある」と明示してください。
根拠のない数値を作らず、自動承認・自動否決もしないでください。"""

_WEB_SOURCE_QUALITIES = {"primary", "recognized"}


def _hit_key(hit: dict[str, Any]) -> str:
    return str(
        hit.get("file_path")
        or hit.get("ref")
        or hit.get("file_name")
        or hit.get("text")
        or ""
    )


def build_retry_queries(case: dict[str, Any], result: dict[str, Any]) -> list[str]:
    missing_groups = [
        item["aliases"]
        for item in result.get("concept_results") or []
        if not item.get("matched")
    ]
    if not missing_groups:
        return [case["query"]]

    terms = [str(group[0]) for group in missing_groups if group]
    queries = [case["query"] + " " + " ".join(terms)]
    queries.extend(f"{case['query']} {term}" for term in terms)
    return queries


def retrieve_retry_hits(
    case: dict[str, Any],
    result: dict[str, Any],
    search_fn: Callable[[str, int], list[dict[str, Any]]],
    *,
    top_k: int = 5,
    max_hits: int = 8,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in build_retry_queries(case, result):
        for hit in search_fn(query, top_k):
            key = _hit_key(hit)
            if key and key not in seen:
                seen.add(key)
                merged.append(hit)
            if len(merged) >= max_hits:
                return merged
    return merged


def generate_retry_answer(
    case: dict[str, Any],
    result: dict[str, Any],
    hits: list[dict[str, Any]],
    chat_fn: Callable[[str, list[dict], str], str],
) -> dict[str, Any]:
    missing = [
        " / ".join(str(alias) for alias in item["aliases"])
        for item in result.get("concept_results") or []
        if not item.get("matched")
    ]
    covered = [
        str(item["matched_alias"])
        for item in result.get("concept_results") or []
        if item.get("matched")
    ]
    context_parts = []
    source_paths = []
    for hit in hits:
        path = str(hit.get("file_path") or hit.get("ref") or hit.get("file_name") or "")
        text = str(hit.get("text") or "")[:1200]
        if path:
            source_paths.append(path)
        if text:
            context_parts.append(f"参照: {path}\n{text}")

    feedback = "、".join(missing) if missing else "回答の具体性と安全性"
    user_message = (
        f"質問: {case['query']}\n\n"
        f"前回回答:\n{result.get('answer') or ''}\n\n"
        f"前回回答で満たしていた観点: {'、'.join(covered) or 'なし'}\n"
        f"前回回答で不足した確認観点: {feedback}\n"
        "満たしていた観点は残し、不足観点を参照ナレッジに照らした確認方法として補ってください。\n"
        "評価観点の語句は、意味が通る範囲で回答本文に明記してください。\n\n"
        + "\n\n---\n\n".join(context_parts)
    )
    return {
        "answer": chat_fn(_SYSTEM_PROMPT, [], user_message).strip(),
        "source_paths": source_paths,
    }


def apply_required_concept_guardrail(
    case: dict[str, Any],
    answer_data: dict[str, Any],
) -> dict[str, Any]:
    answer = str(answer_data.get("answer") or "")
    scored = score_answer(
        case,
        answer,
        [str(path) for path in answer_data.get("source_paths") or []],
    )
    missing_terms = [
        str(item["aliases"][0])
        for item in scored["concept_results"]
        if not item["matched"] and item["aliases"]
    ]
    additions = []
    if missing_terms:
        additions.append(
            "追加確認: "
            + "、".join(missing_terms)
            + "を資料・数値・契約条件と突合してください。"
        )
    if scored["uncertainty_required"] and not scored["uncertainty_present"]:
        additions.append("適用条件や市場状況は個別に要確認です。")
    if not additions:
        return answer_data

    completed = dict(answer_data)
    completed["answer"] = answer.rstrip() + "\n\n" + "\n".join(additions)
    completed["guardrail_added"] = additions
    return completed


def _is_improvement(current: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if candidate["forbidden_hits"]:
        return False
    if (
        current["uncertainty_required"]
        and current["uncertainty_present"]
        and not candidate["uncertainty_present"]
    ):
        return False
    return (
        float(candidate["score"]),
        int(candidate["concept_hits"]),
    ) > (
        float(current["score"]),
        int(current["concept_hits"]),
    )


def run_web_fallback(
    *,
    cases: list[dict[str, Any]],
    report: dict[str, Any],
    web_improve_answer: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    answers = dict(report.get("answers") or {})
    current = evaluate_answers(cases, answers)
    cases_by_id = {case["id"]: case for case in cases}
    trials = []

    for result in current["cases"]:
        if result["passed"]:
            continue
        case = cases_by_id[result["id"]]
        try:
            candidate_answer = web_improve_answer(case, result)
        except Exception as exc:
            trials.append(
                {
                    "id": result["id"],
                    "before_score": result["score"],
                    "after_score": result["score"],
                    "accepted": False,
                    "trusted_source_count": 0,
                    "sources": [],
                    "error": f"{type(exc).__name__}: {exc}",
                    "missing_after": [
                        item["aliases"]
                        for item in result["concept_results"]
                        if not item["matched"]
                    ],
                }
            )
            continue
        sources = candidate_answer.get("web_sources") or []
        trusted_sources = [
            source
            for source in sources
            if source.get("url") and source.get("quality") in _WEB_SOURCE_QUALITIES
        ]
        candidate = score_answer(
            case,
            str(candidate_answer.get("answer") or ""),
            [str(source["url"]) for source in sources if source.get("url")],
        )
        accepted = bool(trusted_sources) and _is_improvement(result, candidate)
        trials.append(
            {
                "id": result["id"],
                "before_score": result["score"],
                "after_score": candidate["score"],
                "accepted": accepted,
                "trusted_source_count": len(trusted_sources),
                "sources": sources,
                "missing_after": [
                    item["aliases"]
                    for item in candidate["concept_results"]
                    if not item["matched"]
                ],
            }
        )
        if accepted:
            answers[result["id"]] = candidate_answer

    final = evaluate_answers(cases, answers)
    report["web_fallback"] = {
        "attempted": bool(trials),
        "trials": trials,
        "summary": {
            key: final[key]
            for key in (
                "total",
                "passed",
                "pass_rate",
                "average_score",
                "concept_coverage",
                "forbidden_cases",
                "uncertainty_misses",
            )
        },
    }
    report["answers"] = answers
    report["final"] = final
    if final["passed"] == final["total"]:
        report["status"] = "improved_with_web"
    return report


def research_failed_answer(
    case: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    from scripts.auto_research_lease_judgment import (
        _extract_sources,
        _get_gemini_key,
        _source_quality,
    )

    api_key = _get_gemini_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    missing = [
        " / ".join(str(alias) for alias in item["aliases"])
        for item in result.get("concept_results") or []
        if not item.get("matched")
    ]
    today = dt.date.today().isoformat()
    prompt = f"""日本のリース審査回答をWeb検索で補完してください。
質問: {case['query']}
前回回答:
{result.get('answer') or ''}

不足している確認観点: {'、'.join(missing) or '根拠と具体性'}
調査日: {today}

要件:
- 官公庁、法令・基準設定主体、公的機関、業界団体、メーカー等の一次情報を優先する。
- 制度・会計・法規制・時期は現在有効か確認する。
- 前回回答の正しい部分を残し、不足観点を実務上の確認方法として補う。
- 事実と推論を分け、断定できない内容は「要確認」と明記する。
- 自動承認・自動否決を提案しない。
- 回答本文だけを日本語で返す。"""

    client = genai.Client(api_key=api_key)
    model = os.environ.get("GEMINI_RESEARCH_MODEL") or os.environ.get(
        "GEMINI_MODEL", "gemini-2.5-flash"
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1800,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            http_options=types.HttpOptions(timeout=60000),
        ),
    )
    answer = str(getattr(response, "text", "") or "").strip()
    sources = [
        {
            **source,
            "quality": _source_quality(source["title"], source["url"]),
        }
        for source in _extract_sources(response)
    ]
    if not answer:
        raise RuntimeError("Web research returned no answer")
    return {
        "answer": answer,
        "source_paths": [source["url"] for source in sources],
        "web_sources": sources,
        "research_model": model,
        "researched_at": today,
    }


def run_auto_improvement(
    *,
    cases: list[dict[str, Any]],
    initial_answers: dict[str, dict[str, Any] | str],
    improve_answer: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    max_iterations: int = 3,
) -> dict[str, Any]:
    answers = dict(initial_answers)
    baseline = evaluate_answers(cases, answers)
    report: dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "healthy" if baseline["passed"] == baseline["total"] else "not_improved",
        "baseline": baseline,
        "iterations": [],
        "final": baseline,
    }

    cases_by_id = {case["id"]: case for case in cases}
    for iteration_number in range(1, max_iterations + 1):
        current = evaluate_answers(cases, answers)
        failed = [result for result in current["cases"] if not result["passed"]]
        if not failed:
            report["status"] = "improved"
            break

        changes = []
        for result in failed:
            case = cases_by_id[result["id"]]
            candidate_answer = apply_required_concept_guardrail(
                case,
                improve_answer(case, result),
            )
            candidate = score_answer(
                case,
                str(candidate_answer.get("answer") or ""),
                [str(path) for path in candidate_answer.get("source_paths") or []],
            )
            accepted = _is_improvement(result, candidate)
            changes.append(
                {
                    "id": result["id"],
                    "before_score": result["score"],
                    "after_score": candidate["score"],
                    "accepted": accepted,
                    "guardrail_added": candidate_answer.get("guardrail_added") or [],
                    "missing_after": [
                        item["aliases"]
                        for item in candidate["concept_results"]
                        if not item["matched"]
                    ],
                }
            )
            if accepted:
                answers[result["id"]] = candidate_answer

        final = evaluate_answers(cases, answers)
        report["iterations"].append(
            {
                "iteration": iteration_number,
                "changes": changes,
                "summary": {
                    key: final[key]
                    for key in (
                        "total",
                        "passed",
                        "pass_rate",
                        "average_score",
                        "concept_coverage",
                        "forbidden_cases",
                        "uncertainty_misses",
                    )
                },
            }
        )
        report["final"] = final
        if final["passed"] == final["total"]:
            report["status"] = "improved"
            break
        if not any(change["accepted"] for change in changes):
            break

    report["answers"] = answers
    return report


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPO_ROOT / "reports" / "answer_quality_latest.json",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-web-fallback", action="store_true")
    args = parser.parse_args()

    cases = json.loads(args.eval_set.read_text(encoding="utf-8"))
    baseline_report = json.loads(args.baseline.read_text(encoding="utf-8"))
    initial_answers = {
        item["id"]: {
            "answer": item.get("answer") or "",
            "source_paths": item.get("source_paths") or [],
        }
        for item in baseline_report.get("cases") or []
    }

    from api.chat_memory import call_gemini_chat
    from api.knowledge.vector_store import get_store

    store = get_store()

    def improve_answer(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        hits = retrieve_retry_hits(
            case,
            result,
            lambda query, limit: store.search(query, top_k=limit),
            top_k=args.top_k,
        )
        return generate_retry_answer(case, result, hits, call_gemini_chat)

    report = run_auto_improvement(
        cases=cases,
        initial_answers=initial_answers,
        improve_answer=improve_answer,
        max_iterations=max(1, args.max_iterations),
    )
    if report["final"]["passed"] < report["final"]["total"] and not args.no_web_fallback:
        report = run_web_fallback(
            cases=cases,
            report=report,
            web_improve_answer=research_failed_answer,
        )
    report["eval_set"] = str(args.eval_set)
    report["baseline_report"] = str(args.baseline)
    _atomic_write_json(args.report, report)

    final = report["final"]
    print(
        "[answer-auto-improve] "
        f"status={report['status']} "
        f"pass={final['passed']}/{final['total']} "
        f"avg={final['average_score']} "
        f"concept={final['concept_coverage']}% "
        f"forbidden={final['forbidden_cases']}"
    )
    raise SystemExit(0 if final["passed"] == final["total"] else 1)


if __name__ == "__main__":
    main()
