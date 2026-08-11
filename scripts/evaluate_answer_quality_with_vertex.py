#!/usr/bin/env python3
"""Evaluate Shion answer quality with an optional Vertex Gemini judge.

This runner is intentionally sidecar-only. It reads an existing answer response
JSON, applies the deterministic answer-quality checks, and optionally asks
Vertex Gemini for a richer judge report. The output is a report artifact only;
it is not connected to scoring, RAG, prompts, or automatic promotion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_answer_quality import DEFAULT_EVAL_SET, evaluate_answers

DEFAULT_OUTPUT = REPO_ROOT / "reports" / "vertex_answer_quality_latest.json"
DEFAULT_MODEL = "gemini-2.5-flash"
JUDGE_DIMENSIONS = (
    "grounding",
    "practical_usefulness",
    "shion_fit",
    "information_health",
    "judgment_asset_connection",
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _compact_text(value: Any, limit: int = 4000) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _answer_payload(raw: dict[str, Any] | str | None) -> tuple[str, list[str]]:
    if isinstance(raw, dict):
        answer = str(raw.get("answer") or "")
        source_paths = [str(path) for path in raw.get("source_paths") or []]
        return answer, source_paths
    return str(raw or ""), []


def coerce_answers_payload(payload: Any) -> dict[str, dict[str, Any] | str]:
    """Accept either a simple response map or an answer-quality report."""
    if not isinstance(payload, dict):
        return {}
    cases = payload.get("cases")
    if isinstance(cases, list):
        answers: dict[str, dict[str, Any] | str] = {}
        for item in cases:
            if not isinstance(item, dict):
                continue
            case_id = str(item.get("id") or "")
            if not case_id:
                continue
            answers[case_id] = {
                "answer": str(item.get("answer") or ""),
                "source_paths": [str(path) for path in item.get("source_paths") or []],
            }
        return answers
    return payload


def build_judge_prompt(
    case: dict[str, Any],
    deterministic_result: dict[str, Any],
) -> str:
    required = [
        aliases[0]
        for aliases in case.get("required_concepts") or []
        if aliases
    ]
    return f"""あなたは紫苑の回答品質を点検する Vertex 側の審査員です。
対象はリース審査AIの回答です。本文を改善するのではなく、評価JSONだけを返してください。

評価軸:
- grounding: 根拠・出典・断定抑制が適切か
- practical_usefulness: 実務で次に確認する資料・条件・反証が具体的か
- shion_fit: 紫苑らしく、抽象論だけでなく判断資産へ接続できているか
- information_health: 過剰参照、過剰自己言及、ノイズ、長文化がないか
- judgment_asset_connection: 再利用できる確認質問・承認条件・リスク信号が含まれるか

各軸は0から5の整数で採点してください。危険な断定、禁止主張、自動承認/自動否決、根拠なき数値があれば flags に入れてください。
JSON以外は返さないでください。

出力形式:
{{
  "scores": {{
    "grounding": 0,
    "practical_usefulness": 0,
    "shion_fit": 0,
    "information_health": 0,
    "judgment_asset_connection": 0
  }},
  "flags": ["..."],
  "review": "短い評価",
  "improvement_suggestion": "次に直すなら何か"
}}

評価ケースID: {case.get("id")}
質問: {case.get("query")}
必須観点: {", ".join(required)}
禁止主張: {", ".join(str(item) for item in case.get("forbidden_claims") or [])}
決定的評価スコア: {deterministic_result.get("score")}
決定的評価の未充足観点: {", ".join(
        str(item.get("aliases", [""])[0])
        for item in deterministic_result.get("concept_results") or []
        if not item.get("matched")
    )}
検出済み禁止主張: {", ".join(str(item) for item in deterministic_result.get("forbidden_hits") or [])}
回答:
{_compact_text(deterministic_result.get("answer"), 5000)}
"""


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object even when the model wraps it in fences or prose."""
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw_scores = payload.get("scores") if isinstance(payload.get("scores"), dict) else {}
    scores: dict[str, int] = {}
    for key in JUDGE_DIMENSIONS:
        try:
            value = int(raw_scores.get(key, 0))
        except (TypeError, ValueError):
            value = 0
        scores[key] = max(0, min(5, value))
    total = sum(scores.values())
    flags = [
        _compact_text(flag, 120)
        for flag in (payload.get("flags") or [])
        if str(flag or "").strip()
    ][:8]
    return {
        "scores": scores,
        "score": round(total / (len(JUDGE_DIMENSIONS) * 5) * 100, 1),
        "flags": flags,
        "review": _compact_text(payload.get("review"), 500),
        "improvement_suggestion": _compact_text(payload.get("improvement_suggestion"), 500),
    }


def call_vertex_gemini(prompt: str, *, model: str = DEFAULT_MODEL) -> str:
    """Call Vertex Gemini through the existing optional Vertex auth helper."""
    from api.vertex_agent_search import _post_json, get_config

    config = get_config()
    location = os.environ.get("VERTEX_AI_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
    url = (
        f"https://aiplatform.googleapis.com/v1/projects/{config.project_id}/locations/global/"
        f"publishers/google/models/{model}:generateContent"
        if location == "global"
        else f"https://{location}-aiplatform.googleapis.com/v1/projects/{config.project_id}/locations/{location}/"
        f"publishers/google/models/{model}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    response = _post_json(
        url,
        body,
        config,
        timeout_seconds=float(os.environ.get("VERTEX_ANSWER_QUALITY_TIMEOUT_SECONDS", "45") or 45),
    )
    candidate = (response.get("candidates") or [{}])[0]
    parts = candidate.get("content", {}).get("parts", []) or []
    return "\n".join(str(part.get("text") or "") for part in parts if part.get("text"))


def judge_case(
    case: dict[str, Any],
    deterministic_result: dict[str, Any],
    *,
    judge_fn: Callable[[str], str],
) -> dict[str, Any]:
    prompt = build_judge_prompt(case, deterministic_result)
    text = judge_fn(prompt)
    parsed = extract_json_object(text)
    normalized = normalize_judge_payload(parsed)
    normalized["status"] = "ok" if parsed else "parse_error"
    return normalized


def run_vertex_answer_quality_eval(
    *,
    cases: list[dict[str, Any]],
    answers: dict[str, dict[str, Any] | str],
    judge_fn: Callable[[str], str] | None = None,
    enable_vertex: bool = True,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    deterministic = evaluate_answers(cases, answers)
    case_by_id = {case["id"]: case for case in cases}
    vertex_cases = []
    vertex_scores: list[float] = []
    vertex_status = "disabled" if not enable_vertex else "ok"

    for deterministic_case in deterministic["cases"]:
        case = case_by_id[deterministic_case["id"]]
        vertex_result: dict[str, Any]
        if not enable_vertex:
            vertex_result = {"status": "disabled"}
        else:
            try:
                caller = judge_fn or (lambda prompt: call_vertex_gemini(prompt, model=model))
                vertex_result = judge_case(case, deterministic_case, judge_fn=caller)
                if vertex_result.get("status") == "ok":
                    vertex_scores.append(float(vertex_result["score"]))
            except Exception as exc:
                vertex_status = "partial_error"
                vertex_result = {
                    "status": "error",
                    "error": _compact_text(exc, 240),
                }
        vertex_cases.append(
            {
                "id": deterministic_case["id"],
                "query": deterministic_case["query"],
                "deterministic_score": deterministic_case["score"],
                "deterministic_passed": deterministic_case["passed"],
                "vertex": vertex_result,
            }
        )

    if enable_vertex and not vertex_scores and any(item["vertex"].get("status") == "error" for item in vertex_cases):
        vertex_status = "error"

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "needs_review" if deterministic["passed"] < deterministic["total"] else "passed",
        "model": model,
        "sidecar_only": True,
        "safety": {
            "connected_to_rag": False,
            "connected_to_scoring": False,
            "connected_to_prompts": False,
            "auto_promotion": False,
        },
        "deterministic": deterministic,
        "vertex": {
            "enabled": enable_vertex,
            "status": vertex_status,
            "average_score": round(sum(vertex_scores) / len(vertex_scores), 1) if vertex_scores else None,
            "judged": len(vertex_scores),
            "total": len(cases),
        },
        "cases": vertex_cases,
    }


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.environ.get("VERTEX_ANSWER_QUALITY_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--skip-vertex", action="store_true")
    args = parser.parse_args()

    cases = json.loads(args.eval_set.read_text(encoding="utf-8"))
    answers = coerce_answers_payload(json.loads(args.responses.read_text(encoding="utf-8")))
    enable_vertex = (
        not args.skip_vertex
        and _env_bool("VERTEX_ANSWER_QUALITY_EVAL_ENABLED", True)
    )
    report = run_vertex_answer_quality_eval(
        cases=cases,
        answers=answers,
        enable_vertex=enable_vertex,
        model=args.model,
    )
    report["eval_set"] = str(args.eval_set)
    report["responses"] = str(args.responses)
    _atomic_write_json(args.output, report)

    deterministic = report["deterministic"]
    vertex = report["vertex"]
    print(
        "[vertex-answer-quality] "
        f"status={report['status']} "
        f"deterministic={deterministic['passed']}/{deterministic['total']} "
        f"avg={deterministic['average_score']} "
        f"vertex={vertex['status']} "
        f"judged={vertex['judged']}/{vertex['total']} "
        f"vertex_avg={vertex['average_score']}"
    )
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
