#!/usr/bin/env python3
"""Build Shion's read-only experience flywheel report.

This turns existing traces into reviewable experience candidates:
context -> decision -> feedback -> gate decision.

The report is intentionally non-mutating. It does not fine-tune models, promote
memory, or edit skills. It only decides what is worth human review, replay eval,
observation, or quarantine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPT_FEEDBACK = REPO_ROOT / "data" / "prompt_feedback_log.jsonl"
DEFAULT_EXPERIENCE_EVENTS = REPO_ROOT / "data" / "shion_experience_events.jsonl"
DEFAULT_SCREENING_FEEDBACK = REPO_ROOT / "data" / "screening_loop_feedback.jsonl"
DEFAULT_JUDGMENT_ASSET_FEEDBACK = REPO_ROOT / "data" / "judgment_asset_usage_feedback.jsonl"

DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports" / "experience_flywheel_latest.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "experience_flywheel_latest.md"

POSITIVE_FEEDBACK = {"helped", "使える", "合っている", "accepted", "approved", "good"}
NEGATIVE_FEEDBACK = {"not_helped", "使えない", "違う", "incorrect", "bad", "harmful"}
NEUTRAL_FEEDBACK = {"neutral", "微妙", "保留", "unclear"}

PROMOTION_REVIEW_LIMIT = 30
REPLAY_LIMIT = 30
OBSERVE_LIMIT = 30
QUARANTINE_LIMIT = 30


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
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


def build_experience_flywheel(
    *,
    prompt_feedback_rows: list[dict[str, Any]] | None = None,
    experience_event_rows: list[dict[str, Any]] | None = None,
    screening_feedback_rows: list[dict[str, Any]] | None = None,
    judgment_asset_feedback_rows: list[dict[str, Any]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    candidates.extend(normalize_prompt_feedback(prompt_feedback_rows or []))
    candidates.extend(normalize_experience_events(experience_event_rows or []))
    candidates.extend(normalize_screening_feedback(screening_feedback_rows or []))
    candidates.extend(normalize_judgment_asset_feedback(judgment_asset_feedback_rows or []))

    deduped, duplicate_summary = deduplicate_candidates(candidates)
    for candidate in deduped:
        candidate["scores"] = score_candidate(candidate)
        candidate["gate"] = gate_candidate(candidate)

    ordered = sorted(
        deduped,
        key=lambda item: (
            item["gate"]["priority"],
            item["scores"]["total"],
            item["scores"]["feedback_sufficiency"],
        ),
        reverse=True,
    )
    by_gate = Counter(item["gate"]["status"] for item in ordered)
    by_source = Counter(item["source"] for item in ordered)
    by_action = Counter(item["gate"]["recommended_action"] for item in ordered)

    return {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "read_only_experience_flywheel_gate",
        "guardrail": "no model weight update, no memory promotion, no skill edit",
        "summary": {
            "raw_candidates": len(candidates),
            "deduped_candidates": len(ordered),
            "duplicates_collapsed": len(duplicate_summary),
            "by_gate": dict(sorted(by_gate.items())),
            "by_source": dict(sorted(by_source.items())),
            "by_recommended_action": dict(sorted(by_action.items())),
        },
        "promotion_review_candidates": [
            slim_candidate(item) for item in ordered if item["gate"]["status"] == "promote_to_review"
        ][:PROMOTION_REVIEW_LIMIT],
        "replay_eval_candidates": [
            slim_candidate(item) for item in ordered if item["gate"]["status"] == "replay_eval"
        ][:REPLAY_LIMIT],
        "observe_only_candidates": [
            slim_candidate(item) for item in ordered if item["gate"]["status"] == "observe_only"
        ][:OBSERVE_LIMIT],
        "quarantined_candidates": [
            slim_candidate(item) for item in ordered if item["gate"]["status"] == "quarantine"
        ][:QUARANTINE_LIMIT],
        "duplicates": duplicate_summary[:30],
        "recommendations": build_recommendations(ordered),
    }


def normalize_prompt_feedback(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        question = _text(row.get("question"))
        response = _text(row.get("response_text") or row.get("response_preview"))
        if not question and not response:
            continue
        response_changed = bool(_text(row.get("response_diff_from_previous")))
        pdca_applied = bool(row.get("pdca_applied"))
        feedback_label = "prompt_changed" if response_changed else ("pdca_applied" if pdca_applied else "implicit")
        candidate = base_candidate(
            source="prompt_feedback",
            source_id=_first(row, "question_hash", "response_hash", "id"),
            timestamp=_first(row, "timestamp", "ts"),
            context={
                "question": question,
                "surface": _text(row.get("surface")),
                "category": _text(row.get("category")),
                "intent": _text(row.get("intent")),
            },
            decision={
                "response_preview": _truncate(response, 420),
                "prompt_final_hash": _text(row.get("prompt_final_hash")),
                "pdca_applied": pdca_applied,
            },
            feedback={
                "label": feedback_label,
                "response_changed": response_changed,
                "prompt_diff_available": bool(_text(row.get("prompt_diff"))),
            },
        )
        candidates.append(candidate)
    return candidates


def normalize_experience_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        message = _text(row.get("message_preview"))
        response = _text(row.get("response_start"))
        if not message and not response:
            continue
        scene = row.get("practical_scene") if isinstance(row.get("practical_scene"), dict) else {}
        signals = row.get("signals") if isinstance(row.get("signals"), dict) else {}
        candidate = base_candidate(
            source="shion_experience",
            source_id=_first(row, "id"),
            timestamp=_first(row, "ts"),
            context={
                "question": message,
                "route": _text(row.get("route")),
                "scene": _text(scene.get("label")),
                "learned_entry_count": int(scene.get("learned_entry_count") or 0),
            },
            decision={
                "response_preview": _truncate(response, 420),
                "judgment_directive": _text(row.get("judgment_directive")),
                "delta": _text(row.get("delta")),
            },
            feedback={
                "label": "implicit_trace",
                "signals": signals,
                "memory_refs_count": int(row.get("memory_refs_count") or 0),
                "knowledge_refs_count": int(row.get("knowledge_refs_count") or 0),
            },
        )
        candidates.append(candidate)
    return candidates


def normalize_screening_feedback(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        rating = _text(row.get("rating"))
        candidate = base_candidate(
            source="screening_feedback",
            source_id=_first(row, "id", "event_id"),
            timestamp=_first(row, "ts"),
            context={
                "surface": _text(row.get("surface")),
                "target": _text(row.get("target")),
                "score": row.get("score"),
                "hantei": _text(row.get("hantei")),
                "case_context": row.get("context") if isinstance(row.get("context"), dict) else {},
            },
            decision={
                "issue_text": _text(row.get("issue_text")),
                "ringi_policy_text": _text(row.get("ringi_policy_text")),
            },
            feedback={
                "label": rating,
                "comment": _text(row.get("comment")),
                "source": _text(row.get("source")),
            },
        )
        candidates.append(candidate)
    return candidates


def normalize_judgment_asset_feedback(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        outcome = _text(row.get("outcome"))
        rule_id = _text(row.get("rule_id") or row.get("judgment_asset_id") or row.get("asset_id"))
        if not rule_id and not outcome:
            continue
        candidate = base_candidate(
            source="judgment_asset_feedback",
            source_id=_first(row, "id", "case_id", "used_at"),
            timestamp=_first(row, "used_at", "ts"),
            context={
                "case_id": _text(row.get("case_id")),
                "rule_id": rule_id,
                "source": _text(row.get("source")),
            },
            decision={
                "used_judgment_asset": rule_id,
            },
            feedback={
                "label": outcome,
                "note": _text(row.get("note")),
            },
        )
        candidates.append(candidate)
    return candidates


def base_candidate(
    *,
    source: str,
    source_id: Any,
    timestamp: Any,
    context: dict[str, Any],
    decision: dict[str, Any],
    feedback: dict[str, Any],
) -> dict[str, Any]:
    seed = json.dumps(
        {
            "source": source,
            "source_id": _text(source_id),
            "context": context,
            "decision": decision,
            "feedback": feedback,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    candidate_id = f"xfly_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"
    return {
        "id": candidate_id,
        "source": source,
        "source_id": _text(source_id),
        "timestamp": _text(timestamp),
        "context": context,
        "decision": decision,
        "feedback": feedback,
        "signature": build_signature(source, context, decision, feedback),
        "duplicate_count": 0,
    }


def build_signature(
    source: str,
    context: dict[str, Any],
    decision: dict[str, Any],
    feedback: dict[str, Any],
) -> str:
    signature_payload = {
        "source": source,
        "question": _truncate(_text(context.get("question")), 80),
        "target": _text(context.get("target")),
        "route": _text(context.get("route")),
        "scene": _text(context.get("scene")),
        "rule_id": _text(context.get("rule_id")),
        "decision": _truncate(_text(decision.get("issue_text") or decision.get("ringi_policy_text") or decision.get("judgment_directive")), 120),
        "feedback": normalized_feedback_label(feedback),
    }
    raw = json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def deduplicate_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_signature: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for candidate in candidates:
        signature = str(candidate.get("signature") or "")
        if signature not in by_signature:
            by_signature[signature] = candidate
            continue
        kept = by_signature[signature]
        kept["duplicate_count"] = int(kept.get("duplicate_count") or 0) + 1
        duplicates.append(
            {
                "signature": signature,
                "kept_id": kept.get("id"),
                "duplicate_id": candidate.get("id"),
                "source": candidate.get("source"),
                "duplicate_count": kept["duplicate_count"],
            }
        )
    return list(by_signature.values()), duplicates


def score_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    feedback = candidate.get("feedback") if isinstance(candidate.get("feedback"), dict) else {}
    context = candidate.get("context") if isinstance(candidate.get("context"), dict) else {}
    decision = candidate.get("decision") if isinstance(candidate.get("decision"), dict) else {}
    label = normalized_feedback_label(feedback)
    explicit_feedback = label in POSITIVE_FEEDBACK | NEGATIVE_FEEDBACK | NEUTRAL_FEEDBACK
    has_comment = bool(_text(feedback.get("comment") or feedback.get("note")))
    trace_depth = trace_depth_score(candidate)
    feedback_sufficiency = 0
    if explicit_feedback:
        feedback_sufficiency = 2 + int(has_comment)
    elif feedback.get("response_changed") or feedback.get("prompt_diff_available"):
        feedback_sufficiency = 1

    decision_specificity = min(
        3,
        int(bool(_text(decision.get("issue_text"))))
        + int(bool(_text(decision.get("ringi_policy_text"))))
        + int(bool(_text(decision.get("judgment_directive"))))
        + int(bool(_text(decision.get("response_preview")))),
    )
    context_specificity = min(
        3,
        int(bool(_text(context.get("question"))))
        + int(bool(_text(context.get("scene"))))
        + int(bool(_text(context.get("rule_id"))))
        + int(bool(context.get("case_context"))),
    )
    diversity_bonus = 1 if candidate.get("duplicate_count", 0) == 0 else 0
    total = feedback_sufficiency * 3 + trace_depth + decision_specificity + context_specificity + diversity_bonus
    if label in NEGATIVE_FEEDBACK:
        total += 2
    return {
        "total": total,
        "feedback_sufficiency": feedback_sufficiency,
        "trace_depth": trace_depth,
        "decision_specificity": decision_specificity,
        "context_specificity": context_specificity,
        "diversity_bonus": diversity_bonus,
        "label": label,
    }


def gate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    scores = candidate.get("scores") if isinstance(candidate.get("scores"), dict) else score_candidate(candidate)
    source = _text(candidate.get("source"))
    label = _text(scores.get("label"))
    feedback_sufficiency = int(scores.get("feedback_sufficiency") or 0)
    total = int(scores.get("total") or 0)

    if source == "prompt_feedback" and feedback_sufficiency <= 1:
        return {
            "status": "replay_eval",
            "priority": 70,
            "recommended_action": "add_or_update_eval_case",
            "reason": "prompt changed without explicit outcome feedback",
        }

    if label in NEGATIVE_FEEDBACK:
        return {
            "status": "replay_eval",
            "priority": 95,
            "recommended_action": "add_failure_case_to_eval",
            "reason": "explicit negative feedback should become a replay case",
        }

    if label in NEUTRAL_FEEDBACK:
        return {
            "status": "replay_eval",
            "priority": 82,
            "recommended_action": "add_boundary_case_to_eval",
            "reason": "neutral feedback marks a boundary case",
        }

    if label in POSITIVE_FEEDBACK and feedback_sufficiency >= 2 and total >= 9:
        action = "review_judgment_asset_reinforcement"
        if source == "screening_feedback":
            action = "review_screening_judgment_pattern"
        return {
            "status": "promote_to_review",
            "priority": 90,
            "recommended_action": action,
            "reason": "explicit positive feedback with enough context",
        }

    if total >= 8:
        return {
            "status": "observe_only",
            "priority": 45,
            "recommended_action": "keep_as_trace_until_feedback_arrives",
            "reason": "rich trace but no explicit feedback",
        }

    return {
        "status": "quarantine",
        "priority": 10,
        "recommended_action": "do_not_learn",
        "reason": "insufficient feedback or context",
    }


def trace_depth_score(candidate: dict[str, Any]) -> int:
    source = _text(candidate.get("source"))
    feedback = candidate.get("feedback") if isinstance(candidate.get("feedback"), dict) else {}
    context = candidate.get("context") if isinstance(candidate.get("context"), dict) else {}
    if source == "shion_experience":
        return min(
            3,
            int(feedback.get("memory_refs_count") or 0)
            + int(feedback.get("knowledge_refs_count") or 0)
            + int(context.get("learned_entry_count") or 0),
        )
    if source == "screening_feedback":
        return 3 if context.get("case_context") else 2
    if source == "judgment_asset_feedback":
        return 2 + int(bool(context.get("case_id")))
    if source == "prompt_feedback":
        return 1 + int(bool(feedback.get("prompt_diff_available")))
    return 0


def build_recommendations(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(item["gate"]["status"] for item in candidates)
    recommendations: list[dict[str, Any]] = []
    if counts.get("replay_eval", 0):
        recommendations.append(
            {
                "action": "build_replay_eval_set",
                "reason": "feedback contains changed, neutral, or failed decisions",
                "count": counts["replay_eval"],
            }
        )
    if counts.get("promote_to_review", 0):
        recommendations.append(
            {
                "action": "human_review_before_skill_or_memory_promotion",
                "reason": "positive signals are strong enough to inspect, not auto-merge",
                "count": counts["promote_to_review"],
            }
        )
    if counts.get("quarantine", 0):
        recommendations.append(
            {
                "action": "improve_feedback_capture",
                "reason": "some traces lack enough context or explicit feedback to learn from",
                "count": counts["quarantine"],
            }
        )
    return recommendations


def slim_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate.get("id"),
        "source": candidate.get("source"),
        "source_id": candidate.get("source_id"),
        "timestamp": candidate.get("timestamp"),
        "context": candidate.get("context"),
        "decision": candidate.get("decision"),
        "feedback": candidate.get("feedback"),
        "scores": candidate.get("scores"),
        "gate": candidate.get("gate"),
        "signature": candidate.get("signature"),
        "duplicate_count": candidate.get("duplicate_count", 0),
    }


def build_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines: list[str] = []
    lines.append("# Experience Flywheel Report")
    lines.append("")
    lines.append(f"- Generated at: `{report.get('generated_at')}`")
    lines.append(f"- Mode: `{report.get('mode')}`")
    lines.append(f"- Guardrail: {report.get('guardrail')}")
    lines.append(f"- Raw candidates: {summary.get('raw_candidates', 0)}")
    lines.append(f"- Deduped candidates: {summary.get('deduped_candidates', 0)}")
    lines.append(f"- Duplicates collapsed: {summary.get('duplicates_collapsed', 0)}")
    lines.append("")
    lines.append("## Gate Summary")
    for key, count in (summary.get("by_gate") or {}).items():
        lines.append(f"- {key}: {count}")
    lines.append("")
    lines.append("## Promotion Review")
    _append_candidate_lines(lines, report.get("promotion_review_candidates") or [])
    lines.append("")
    lines.append("## Replay Eval")
    _append_candidate_lines(lines, report.get("replay_eval_candidates") or [])
    lines.append("")
    lines.append("## Quarantine")
    _append_candidate_lines(lines, report.get("quarantined_candidates") or [])
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(build_markdown(report), encoding="utf-8")


def normalized_feedback_label(feedback: dict[str, Any]) -> str:
    label = _text(feedback.get("label") or feedback.get("outcome") or feedback.get("rating")).strip()
    lowered = label.lower()
    if lowered in POSITIVE_FEEDBACK:
        return lowered
    if lowered in NEGATIVE_FEEDBACK:
        return lowered
    if lowered in NEUTRAL_FEEDBACK:
        return lowered
    return label


def _append_candidate_lines(lines: list[str], candidates: list[dict[str, Any]]) -> None:
    if not candidates:
        lines.append("- None")
        return
    for item in candidates[:10]:
        gate = item.get("gate") or {}
        scores = item.get("scores") or {}
        context = item.get("context") or {}
        title = (
            _text(context.get("question"))
            or _text((item.get("decision") or {}).get("issue_text"))
            or _text(context.get("rule_id"))
            or _text(item.get("source_id"))
        )
        lines.append(
            f"- `{item.get('source')}` score={scores.get('total')} "
            f"action={gate.get('recommended_action')} / {_truncate(title, 80)}"
        )


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", {}):
            return value
    return ""


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _truncate(value: str, limit: int) -> str:
    text = _text(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-feedback", type=Path, default=DEFAULT_PROMPT_FEEDBACK)
    parser.add_argument("--experience-events", type=Path, default=DEFAULT_EXPERIENCE_EVENTS)
    parser.add_argument("--screening-feedback", type=Path, default=DEFAULT_SCREENING_FEEDBACK)
    parser.add_argument("--judgment-asset-feedback", type=Path, default=DEFAULT_JUDGMENT_ASSET_FEEDBACK)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    report = build_experience_flywheel(
        prompt_feedback_rows=read_jsonl(args.prompt_feedback.expanduser()),
        experience_event_rows=read_jsonl(args.experience_events.expanduser()),
        screening_feedback_rows=read_jsonl(args.screening_feedback.expanduser()),
        judgment_asset_feedback_rows=read_jsonl(args.judgment_asset_feedback.expanduser()),
    )
    write_outputs(report, output_json=args.output_json.expanduser(), output_md=args.output_md.expanduser())
    print(f"saved: {args.output_json.expanduser()}")
    print(f"saved: {args.output_md.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
