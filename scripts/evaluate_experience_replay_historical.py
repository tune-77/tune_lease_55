#!/usr/bin/env python3
"""Score historical Shion answers against the experience replay eval set.

This is the no-network fallback for experience replay evaluation. It evaluates
responses already stored in local prompt feedback logs, so it does not send
workspace context or user queries to an external LLM.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_answer_quality import evaluate_answers

DEFAULT_EVAL_SET = REPO_ROOT / "api" / "knowledge" / "experience_replay_eval_set.json"
DEFAULT_PROMPT_FEEDBACK = REPO_ROOT / "data" / "prompt_feedback_log.jsonl"
DEFAULT_CHAT_LOG = REPO_ROOT / "data" / "cloudrun_chat_log.jsonl"
DEFAULT_MAX_AGE_DAYS = 30
# 同じ質問の言い換え（語尾・助詞の違い）だけを拾う。0.5前後だと「焼却炉はリースできる？」に
# 「中古車はリースできる？」が当たり、別物件の回答を採点してしまう
SIMILAR_QUESTION_MIN = 0.8
DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports" / "experience_replay_historical_quality_latest.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "experience_replay_historical_quality_latest.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _normalize_question(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    return re.sub(r"[\s。、，,.！!？?「」『』（）()…・ー〜~]", "", value)


def _bigrams(text: str) -> set[str]:
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _similarity(a: str, b: str) -> float:
    left, right = _bigrams(a), _bigrams(b)
    if not left or not right:
        return 1.0 if a and a == b else 0.0
    return len(left & right) / len(left | right)


def _answer_rows(
    prompt_feedback_rows: list[dict[str, Any]],
    chat_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """履歴を (質問, 回答, 日時, 出典) にそろえる。回答の無い行は捨てる。"""
    rows: list[dict[str, Any]] = []
    for row in prompt_feedback_rows:
        answer = str(row.get("response_text") or row.get("response_preview") or "")
        if answer:
            rows.append(
                {
                    "question": str(row.get("question") or ""),
                    "question_hash": str(row.get("question_hash") or ""),
                    "answer": answer,
                    "answered_at": str(row.get("timestamp") or row.get("ts") or ""),
                    "source": "prompt_feedback_log",
                }
            )
    for row in chat_rows or []:
        answer = str(row.get("assistant_reply") or "")
        if answer:
            rows.append(
                {
                    "question": str(row.get("user_message") or ""),
                    "question_hash": "",
                    "answer": answer,
                    "answered_at": str(row.get("ts") or row.get("timestamp") or ""),
                    "source": f"chat_log:{row.get('surface') or ''}",
                }
            )
    return rows


def build_historical_answers(
    cases: list[dict[str, Any]],
    prompt_feedback_rows: list[dict[str, Any]],
    chat_rows: list[dict[str, Any]] | None = None,
    *,
    fresh_since: str | None = None,
    min_similarity: float = SIMILAR_QUESTION_MIN,
) -> dict[str, dict[str, Any]]:
    """各ケースに、同じ/ごく近い質問への**最新の**回答を割り当てる（REV-497）。

    以前は同じ質問の最初の記録（6/12 の回答）を毎回使っていたため、新しい回答が評価に入らず、
    結果が 8/14 から毎日同じだった。fresh_since（YYYY-MM-DD）より古い回答しか無いケースは
    freshness="stale" として採点に使わず、回答が無いケースは "none" とする（古い回答で代用しない）。
    """
    pool = _answer_rows(prompt_feedback_rows, chat_rows)
    normalized_pool = [(_normalize_question(row["question"]), row) for row in pool]

    answers: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case.get("id") or "")
        source_id = str(case.get("_source_id") or "")
        query = _normalize_question(case.get("query"))
        matches: list[tuple[float, dict[str, Any]]] = []
        for normalized, row in normalized_pool:
            if source_id and row["question_hash"] == source_id:
                matches.append((1.0, row))
                continue
            similarity = 1.0 if normalized and normalized == query else _similarity(query, normalized)
            if similarity >= min_similarity:
                matches.append((similarity, row))
        if not matches:
            answers[case_id] = {
                "answer": "",
                "source_paths": [],
                "matched_historical_log": False,
                "freshness": "none",
            }
            continue
        similarity, latest = max(matches, key=lambda item: item[1]["answered_at"])
        stale = bool(fresh_since) and latest["answered_at"][:10] < str(fresh_since)
        answers[case_id] = {
            "answer": latest["answer"],
            "source_paths": [],
            "matched_historical_log": True,
            "freshness": "stale" if stale else "fresh",
            "answered_at": latest["answered_at"],
            "answer_source": latest["source"],
            "matched_question": latest["question"],
            "question_similarity": round(similarity, 2),
        }
    return answers


def build_report(
    *,
    cases: list[dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    eval_set_path: Path,
    prompt_feedback_path: Path,
) -> dict[str, Any]:
    def _freshness(case: dict[str, Any]) -> str:
        info = answers.get(str(case.get("id") or ""), {})
        return str(info.get("freshness") or ("fresh" if info.get("matched_historical_log") else "none"))

    scored_cases = [case for case in cases if _freshness(case) == "fresh"]
    summary = evaluate_answers(scored_cases, answers)
    for result in summary.get("cases") or []:
        info = answers.get(str(result.get("id") or ""), {})
        result["answered_at"] = info.get("answered_at", "")
        result["answer_source"] = info.get("answer_source", "")
    missing = [
        {"id": case.get("id"), "query": case.get("query")}
        for case in cases
        if _freshness(case) == "none"
    ]
    stale = [
        {
            "id": case.get("id"),
            "query": case.get("query"),
            "latest_answered_at": answers.get(str(case.get("id") or ""), {}).get("answered_at", ""),
        }
        for case in cases
        if _freshness(case) == "stale"
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "historical_local_no_network",
        "eval_set": str(eval_set_path),
        "prompt_feedback_log": str(prompt_feedback_path),
        "answer_coverage": {
            "total_cases": len(cases),
            "scored_with_latest": len(scored_cases),
            "stale": stale,
            "no_answer": missing,
        },
        "missing_historical_answers": missing,
        "final": summary,
    }


def render_markdown(report: dict[str, Any]) -> str:
    final = report["final"]
    lines = [
        "# Experience Replay Historical Quality",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Mode: `{report['mode']}`",
        f"- Passed: {final['passed']}/{final['total']}",
        f"- Pass rate: {final['pass_rate']}%",
        f"- Average score: {final['average_score']}",
        f"- Concept coverage: {final['concept_coverage']}%",
        f"- Forbidden cases: {final['forbidden_cases']}",
        f"- Uncertainty misses: {final['uncertainty_misses']}",
        f"- Missing historical answers: {len(report['missing_historical_answers'])}",
    ]
    coverage = report.get("answer_coverage")
    if coverage:
        lines += [
            f"- Scored with latest answers: {coverage['scored_with_latest']}/{coverage['total_cases']}",
            "",
            "## 最新の回答なし（採点していない）",
        ]
        if not coverage["stale"] and not coverage["no_answer"]:
            lines.append("- None")
        for item in coverage["stale"]:
            lines.append(f"- `{item['id']}` 最新の回答が古い（{item['latest_answered_at'][:10]}） / {item['query']}")
        for item in coverage["no_answer"]:
            lines.append(f"- `{item['id']}` 未回答（履歴に回答なし） / {item['query']}")
    lines += [
        "",
        "## Failed Cases",
    ]
    failed = [case for case in final["cases"] if not case["passed"]]
    if not failed:
        lines.append("- None")
    for case in failed:
        missing = [
            "/".join(result["aliases"])
            for result in case.get("concept_results") or []
            if not result.get("matched")
        ]
        lines.append(
            f"- `{case['id']}` score={case['score']} "
            f"missing={'; '.join(missing) or '-'} "
            f"uncertainty_present={case['uncertainty_present']} "
            f"answered_at={str(case.get('answered_at') or '')[:10] or '-'} / {case['query']}"
        )
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--prompt-feedback", type=Path, default=DEFAULT_PROMPT_FEEDBACK)
    parser.add_argument("--chat-log", type=Path, default=DEFAULT_CHAT_LOG)
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=DEFAULT_MAX_AGE_DAYS,
        help="これより古い回答しか無い質問は採点せず「最新なし」とする",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--strict", action="store_true", help="不合格が1件でもあれば終了コード1にする")
    args = parser.parse_args()

    cases = json.loads(args.eval_set.expanduser().read_text(encoding="utf-8"))
    fresh_since = (datetime.now() - timedelta(days=args.max_age_days)).strftime("%Y-%m-%d")
    answers = build_historical_answers(
        cases,
        read_jsonl(args.prompt_feedback.expanduser()),
        read_jsonl(args.chat_log.expanduser()),
        fresh_since=fresh_since,
    )
    report = build_report(
        cases=cases,
        answers=answers,
        eval_set_path=args.eval_set.expanduser(),
        prompt_feedback_path=args.prompt_feedback.expanduser(),
    )
    write_outputs(report, output_json=args.output_json.expanduser(), output_md=args.output_md.expanduser())
    final = report["final"]
    print(
        "[experience-replay-historical] "
        f"passed={final['passed']}/{final['total']} "
        f"avg={final['average_score']} "
        f"concept={final['concept_coverage']}% "
        f"forbidden={final['forbidden_cases']} "
        f"uncertainty_miss={final['uncertainty_misses']} "
        f"missing={len(report['missing_historical_answers'])}"
    )
    coverage = report["answer_coverage"]
    print(
        "[experience-replay-historical] "
        f"latest_scored={coverage['scored_with_latest']}/{coverage['total_cases']} "
        f"stale={len(coverage['stale'])} no_answer={len(coverage['no_answer'])} "
        f"(fresh_since={fresh_since})"
    )
    failed = final["total"] - final["passed"]
    if failed:
        # 不合格は回答品質の指標であって、パイプラインの障害ではない（REV-496）。
        # 10/3 に `|| true` を外して実際の終了コードを記録するようにした結果、8/14 から変わらない
        # 0/10 が毎日「パイプライン障害」として検出されていた。止めたい時だけ --strict を使う。
        print(f"[experience-replay-historical] warn: 不合格 {failed}/{final['total']} 件（品質の指標・障害ではない）")
    if args.strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
