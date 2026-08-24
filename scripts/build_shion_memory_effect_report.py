#!/usr/bin/env python3
"""Build an effect report for Shion memories.

The report explains which memory layers are being used, which memories are
stale/revised, and which records need review. It is observational only: no
prompt, scoring, RAG rank, Obsidian, or judgment asset store is changed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_ROOT / "data" / "shion_memory_index.json"
DEFAULT_USAGE_LOG = PROJECT_ROOT / "data" / "shion_memory_usage_log.jsonl"
DEFAULT_REVISIONS = PROJECT_ROOT / "data" / "shion_memory_revisions.jsonl"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "shion_memory_effect_latest.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "shion_memory_effect_latest.md"
GUARDRAIL = "observability_only_no_prompt_no_rag_rank_no_scoring_no_auto_promotion"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _feedback_triage(items: list[dict[str, Any]], *, states: set[str]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    selected = [item for item in items if str(item.get("utility_state") or "") in states]
    for item in selected:
        domain = str(item.get("domain") or "").strip()
        if not domain:
            source = str(item.get("source_path") or "").strip()
            domain = Path(source).stem if source else "unknown"
        layer = str(item.get("memory_layer") or "unknown")
        mtype = str(item.get("memory_type") or "unknown")
        key = (domain, layer, mtype)
        group = groups.setdefault(
            key,
            {
                "domain": domain,
                "memory_layer": layer,
                "memory_type": mtype,
                "count": 0,
                "used_count": 0,
                "impact_hint_count": 0,
                "sample_ids": [],
                "samples": [],
            },
        )
        group["count"] += 1
        group["used_count"] += int(item.get("used_count") or 0)
        group["impact_hint_count"] += int(item.get("impact_hint_count") or 0)
        if len(group["samples"]) < 3:
            group["sample_ids"].append(str(item.get("id") or ""))
            group["samples"].append(
                {
                    "id": str(item.get("id") or ""),
                    "used_count": int(item.get("used_count") or 0),
                    "utility_state": str(item.get("utility_state") or ""),
                    "reason": str(item.get("reason") or ""),
                    "content": str(item.get("content") or "")[:140],
                }
            )
    batches = sorted(groups.values(), key=lambda row: (-int(row["used_count"]), -int(row["count"]), str(row["domain"])))
    return {
        "record_count": len(selected),
        "batch_count": len(batches),
        "top_batches": batches[:20],
        "policy": "group_all_matching_memory_utility_records_by_domain_layer_type",
    }


def build_report(index: dict[str, Any], usage_rows: list[dict[str, Any]], revision_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    records = [r for r in index.get("records") or [] if isinstance(r, dict)]
    by_id = {str(r.get("id") or ""): r for r in records if r.get("id")}
    usage_by_id: Counter[str] = Counter()
    impact_by_id: Counter[str] = Counter()
    layer_usage: Counter[str] = Counter()
    type_usage: Counter[str] = Counter()
    routes_by_id: dict[str, set[str]] = {}
    questions_by_id: dict[str, set[str]] = {}
    latest_question: dict[str, str] = {}
    latest_used_at: dict[str, str] = {}
    explicit_feedback_by_id: dict[str, Counter[str]] = {}
    impact_hint_count = 0

    for row in usage_rows:
        refs = [str(ref) for ref in row.get("refs") or [] if ref]
        route = str(row.get("route") or "unknown")
        question = str(row.get("question") or "").strip()
        ts = str(row.get("ts") or row.get("timestamp") or "").strip()
        feedback = _usage_feedback_label(row)
        if row.get("impact_hints"):
            impact_hint_count += 1
            for hint in row.get("impact_hints") or []:
                if isinstance(hint, dict) and hint.get("id"):
                    impact_by_id[str(hint["id"])] += 1
        for ref in refs:
            usage_by_id[ref] += 1
            record = by_id.get(ref) or {}
            layer_usage[str(record.get("memory_layer") or "unknown")] += 1
            type_usage[str(record.get("memory_type") or "unknown")] += 1
            routes_by_id.setdefault(ref, set()).add(route)
            if question:
                questions_by_id.setdefault(ref, set()).add(question[:160])
                latest_question[ref] = question
            if ts:
                latest_used_at[ref] = max(latest_used_at.get(ref, ""), ts)
            if feedback:
                explicit_feedback_by_id.setdefault(ref, Counter())[feedback] += 1

    status_counts = Counter(str(r.get("status") or "active") for r in records)
    layer_counts = Counter(str(r.get("memory_layer") or "unknown") for r in records)
    unused_persistent = [
        _record_summary(r, usage_by_id, latest_question)
        for r in records
        if str(r.get("memory_layer") or "") == "persistent" and usage_by_id[str(r.get("id") or "")] == 0
    ]
    review_candidates = [
        _record_summary(r, usage_by_id, latest_question)
        for r in records
        if str(r.get("status") or "") in {"stale", "revised"}
    ][:30]
    top_used = [
        _record_summary(by_id[rid], usage_by_id, latest_question)
        for rid, _count in usage_by_id.most_common(20)
        if rid in by_id
    ]
    utility_records = [
        _utility_summary(
            record,
            usage_by_id,
            impact_by_id,
            routes_by_id,
            questions_by_id,
            latest_question,
            latest_used_at,
            explicit_feedback_by_id,
        )
        for record in records
    ]
    utility_counts = Counter(item["utility_state"] for item in utility_records)
    likely_helpful = [
        item for item in utility_records if item["utility_state"] in {"validated", "likely_helpful"}
    ]
    likely_helpful.sort(key=lambda item: (-float(item["utility_score"]), item["id"]))
    needs_feedback = [
        item for item in utility_records if item["utility_state"] in {"needs_feedback", "observed_no_impact"}
    ]
    needs_feedback.sort(key=lambda item: (-int(item["used_count"]), item["id"]))
    possible_noise = [
        item for item in utility_records if item["utility_state"] in {"challenged", "needs_review"}
    ]
    possible_noise.sort(key=lambda item: (-float(item["utility_score"]), item["id"]))
    revisions = revision_rows or []
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "guardrail": GUARDRAIL,
        "summary": {
            "records": len(records),
            "usage_events": len(usage_rows),
            "used_memory_ids": len(usage_by_id),
            "usage_events_with_impact_hints": impact_hint_count,
            "by_layer": dict(sorted(layer_counts.items())),
            "usage_by_layer": dict(sorted(layer_usage.items())),
            "usage_by_type": dict(sorted(type_usage.items())),
            "by_status": dict(sorted(status_counts.items())),
            "revision_events": len(revisions),
            "utility_by_state": dict(sorted(utility_counts.items())),
            "likely_helpful_memory_ids": sum(
                utility_counts.get(state, 0) for state in ("validated", "likely_helpful")
            ),
            "needs_feedback_memory_ids": sum(
                utility_counts.get(state, 0) for state in ("needs_feedback", "observed_no_impact")
            ),
            "possible_noise_memory_ids": sum(
                utility_counts.get(state, 0) for state in ("challenged", "needs_review")
            ),
        },
        "top_used": top_used,
        "likely_helpful": likely_helpful[:20],
        "needs_feedback": needs_feedback[:20],
        "needs_feedback_triage": _feedback_triage(
            needs_feedback,
            states={"needs_feedback", "observed_no_impact"},
        ),
        "possible_noise": possible_noise[:20],
        "possible_noise_triage": _feedback_triage(
            possible_noise,
            states={"challenged", "needs_review"},
        ),
        "review_candidates": review_candidates,
        "unused_persistent": unused_persistent,
        "next_actions": [
            "likely_helpful は回答へ効いた可能性が高い記憶として、同種質問で再利用を観測する。",
            "needs_feedback は想起されているが効き方の証跡が薄いので、回答後の helped / neutral / challenged を取る。",
            "possible_noise は stale/revised の使用や否定フィードバックを優先確認する。",
            "stale/revised は削除せず、必要なら scripts/revise_shion_memory.py で後継記憶を登録する。",
            "unused_persistent は強い原則なのに使われていないため、強すぎる/不要/参照条件が狭すぎる可能性を見る。",
        ],
    }


def _usage_feedback_label(row: dict[str, Any]) -> str:
    for key in ("memory_feedback", "feedback", "outcome", "result", "helpfulness"):
        value = str(row.get(key) or "").strip().lower()
        if value in {"helped", "useful", "役に立った", "good", "positive"}:
            return "helped"
        if value in {"neutral", "微妙", "unknown"}:
            return "neutral"
        if value in {"challenged", "rejected", "wrong", "違う", "要修正", "negative"}:
            return "challenged"
    return ""


def _utility_summary(
    record: dict[str, Any],
    usage_by_id: Counter[str],
    impact_by_id: Counter[str],
    routes_by_id: dict[str, set[str]],
    questions_by_id: dict[str, set[str]],
    latest_question: dict[str, str],
    latest_used_at: dict[str, str],
    explicit_feedback_by_id: dict[str, Counter[str]],
) -> dict[str, Any]:
    rid = str(record.get("id") or "")
    used = int(usage_by_id.get(rid, 0))
    impact = int(impact_by_id.get(rid, 0))
    route_count = len(routes_by_id.get(rid, set()))
    question_count = len(questions_by_id.get(rid, set()))
    feedback = explicit_feedback_by_id.get(rid, Counter())
    helped = int(feedback.get("helped", 0))
    neutral = int(feedback.get("neutral", 0))
    challenged = int(feedback.get("challenged", 0))
    status = str(record.get("status") or "active")

    score = 0.0
    score += min(35.0, used * 4.0)
    score += min(30.0, impact * 10.0)
    score += min(12.0, route_count * 3.0)
    score += min(12.0, question_count * 2.0)
    score += min(30.0, helped * 15.0)
    score += min(8.0, neutral * 4.0)
    score -= min(35.0, challenged * 18.0)
    if status == "stale":
        score -= 12.0
    elif status == "revised":
        score -= 18.0
    score = round(max(0.0, min(100.0, score)), 1)

    if challenged:
        state = "challenged"
    elif status in {"stale", "revised"} and used:
        state = "needs_review"
    elif helped:
        state = "validated"
    elif impact >= 2 or (impact >= 1 and used >= 2):
        state = "likely_helpful"
    elif used >= 3 and impact == 0:
        state = "needs_feedback"
    elif used > 0:
        state = "observed_no_impact"
    else:
        state = "unused"

    payload = _record_summary(record, usage_by_id, latest_question)
    payload.update(
        {
            "utility_state": state,
            "utility_score": score,
            "impact_hint_count": impact,
            "route_count": route_count,
            "distinct_question_count": question_count,
            "latest_used_at": latest_used_at.get(rid, ""),
            "domain": str(record.get("domain") or ""),
            "use_when": str(record.get("use_when") or ""),
            "explicit_feedback": {
                "helped": helped,
                "neutral": neutral,
                "challenged": challenged,
            },
            "reason": _utility_reason(state, used=used, impact=impact, status=status, challenged=challenged),
        }
    )
    return payload


def _utility_reason(state: str, *, used: int, impact: int, status: str, challenged: int) -> str:
    if state == "challenged":
        return f"否定/修正系フィードバックが {challenged} 件ある"
    if state == "needs_review":
        return f"{status} 状態だが想起されている"
    if state == "validated":
        return "明示的な helped フィードバックがある"
    if state == "likely_helpful":
        return f"想起 {used} 回、impact_hints {impact} 回で回答への効き方が記録されている"
    if state == "needs_feedback":
        return f"想起 {used} 回だが impact_hints が無く、効いたか不明"
    if state == "observed_no_impact":
        return "想起はされたが回答への効き方の証跡が薄い"
    return "まだ実回答での利用証跡がない"


def _record_summary(record: dict[str, Any], usage_by_id: Counter[str], latest_question: dict[str, str]) -> dict[str, Any]:
    rid = str(record.get("id") or "")
    return {
        "id": rid,
        "memory_layer": str(record.get("memory_layer") or ""),
        "memory_type": str(record.get("memory_type") or ""),
        "status": str(record.get("status") or "active"),
        "source_path": str(record.get("source_path") or ""),
        "used_count": int(usage_by_id.get(rid, 0)),
        "last_used_at": str(record.get("last_used_at") or ""),
        "latest_question": latest_question.get(rid, ""),
        "content": " ".join(str(record.get("content") or "").split())[:180],
    }


def markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Shion Memory Effect Report",
        "",
        f"- Guardrail: `{report.get('guardrail')}`",
        f"- Records: {summary.get('records', 0)}",
        f"- Usage events: {summary.get('usage_events', 0)}",
        f"- Used memory ids: {summary.get('used_memory_ids', 0)}",
        f"- Impact-hint events: {summary.get('usage_events_with_impact_hints', 0)}",
        f"- Usage by layer: {summary.get('usage_by_layer', {})}",
        f"- Utility by state: {summary.get('utility_by_state', {})}",
        "",
        "## Top Used",
    ]
    for item in report.get("top_used") or []:
        lines.append(f"- {item.get('id')} [{item.get('memory_layer')}/{item.get('status')}] used={item.get('used_count')} {item.get('content')}")
    lines.extend(["", "## Likely Helpful"])
    for item in report.get("likely_helpful") or []:
        lines.append(
            f"- {item.get('id')} score={item.get('utility_score')} impact={item.get('impact_hint_count')} "
            f"domain={item.get('domain')} {item.get('content')}"
        )
    lines.extend(["", "## Needs Feedback"])
    for item in report.get("needs_feedback") or []:
        lines.append(
            f"- {item.get('id')} used={item.get('used_count')} impact={item.get('impact_hint_count')} "
            f"reason={item.get('reason')} {item.get('content')}"
        )
    triage = report.get("needs_feedback_triage") or {}
    lines.extend(["", "## Needs Feedback Triage"])
    for batch in triage.get("top_batches") or []:
        lines.append(
            f"- {batch.get('domain')} {batch.get('memory_layer')}/{batch.get('memory_type')}: "
            f"{batch.get('count')} records, used={batch.get('used_count')}"
        )
    lines.extend(["", "## Possible Noise"])
    for item in report.get("possible_noise") or []:
        lines.append(
            f"- {item.get('id')} [{item.get('status')}] state={item.get('utility_state')} reason={item.get('reason')} {item.get('content')}"
        )
    lines.extend(["", "## Review Candidates"])
    for item in report.get("review_candidates") or []:
        lines.append(f"- {item.get('id')} [{item.get('status')}] {item.get('content')}")
    lines.extend(["", "## Unused Persistent"])
    for item in report.get("unused_persistent") or []:
        lines.append(f"- {item.get('id')} {item.get('content')}")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--usage-log", type=Path, default=DEFAULT_USAGE_LOG)
    parser.add_argument("--revisions", type=Path, default=DEFAULT_REVISIONS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = build_report(_read_json(args.index), _read_jsonl(args.usage_log), _read_jsonl(args.revisions))
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(report), encoding="utf-8")
    print(f"wrote={args.output_json}")
    print(f"wrote={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
