#!/usr/bin/env python3
"""Review and publish accepted experience replay checklist items.

Candidates are regenerated reports. Reviews are durable append-only decisions.
Only accepted/revised items are published to the active checklist consumed by
Reflection Gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.loop_engineering_common import DATA_DIR, append_jsonl, load_jsonl

DEFAULT_CANDIDATES_JSON = REPO_ROOT / "reports" / "experience_replay_checklist_candidates_latest.json"
DEFAULT_REVIEWS_JSONL = DATA_DIR / "experience_replay_checklist_reviews.jsonl"
DEFAULT_ACTIVE_JSON = DATA_DIR / "experience_replay_accepted_checklist.json"
DEFAULT_REPORT_MD = REPO_ROOT / "reports" / "experience_replay_checklist_review_latest.md"

VALID_DECISIONS = {"accepted", "revised", "held", "rejected"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text(value: Any) -> str:
    return str(value or "").strip()


def load_candidates(path: Path = DEFAULT_CANDIDATES_JSON) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def normalize_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in payload.get("global_checklist_candidates") or []:
        if not isinstance(item, dict):
            continue
        checklist_item = _text(item.get("checklist_item"))
        candidate_id = _text(item.get("id"))
        if not candidate_id or not checklist_item:
            continue
        candidates.append(
            {
                "id": candidate_id,
                "scope": "global",
                "trigger": _text(item.get("trigger")),
                "topic": "",
                "source_case_id": "",
                "query": "",
                "checklist_items": [checklist_item],
                "priority": int(item.get("evidence_count") or 0),
                "source": item,
            }
        )
    for item in payload.get("case_checklist_candidates") or []:
        if not isinstance(item, dict):
            continue
        checklist_items = [_text(value) for value in item.get("checklist_items") or [] if _text(value)]
        candidate_id = _text(item.get("id"))
        if not candidate_id or not checklist_items:
            continue
        candidates.append(
            {
                "id": candidate_id,
                "scope": "case",
                "trigger": _text(item.get("topic") or item.get("query")),
                "topic": _text(item.get("topic")),
                "source_case_id": _text(item.get("source_case_id")),
                "query": _text(item.get("query")),
                "checklist_items": checklist_items,
                "priority": int(item.get("priority") or 0),
                "source": item,
            }
        )
    return candidates


def latest_reviews(reviews_path: Path = DEFAULT_REVIEWS_JSONL) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(reviews_path, newest_first=False):
        candidate_id = _text(row.get("candidate_id"))
        if candidate_id:
            latest[candidate_id] = row
    return latest


def overlay_reviews(
    candidates: list[dict[str, Any]],
    *,
    reviews_path: Path = DEFAULT_REVIEWS_JSONL,
) -> list[dict[str, Any]]:
    reviews = latest_reviews(reviews_path)
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        review = reviews.get(item["id"])
        if review:
            item["review"] = review
            item["review_decision"] = review.get("decision")
            item["reviewed_at"] = review.get("reviewed_at")
            item["review_note"] = review.get("note", "")
            item["status"] = review.get("decision")
        else:
            item["status"] = "candidate_needs_human_review"
        out.append(item)
    return out


def record_review(
    *,
    candidate_id: str,
    decision: str,
    candidates_payload: dict[str, Any] | None = None,
    note: str = "",
    edited_items: list[str] | None = None,
    reviewer: str = "user",
    reviews_path: Path = DEFAULT_REVIEWS_JSONL,
) -> dict[str, Any]:
    clean_candidate_id = _text(candidate_id)
    clean_decision = _text(decision)
    if not clean_candidate_id:
        raise ValueError("candidate_id is required")
    if clean_decision not in VALID_DECISIONS:
        raise ValueError("decision must be accepted, revised, held, or rejected")
    candidates = normalize_candidates(candidates_payload or load_candidates())
    source = next((item for item in candidates if item["id"] == clean_candidate_id), None)
    if not source:
        raise ValueError("candidate not found")
    clean_edited = [_text(item) for item in edited_items or [] if _text(item)]
    review = {
        "candidate_id": clean_candidate_id,
        "decision": clean_decision,
        "note": _text(note),
        "edited_checklist_items": clean_edited,
        "reviewer": _text(reviewer) or "user",
        "reviewed_at": _now_iso(),
        "source_scope": source["scope"],
        "source_case_id": source.get("source_case_id", ""),
        "policy": "human_review_only_accepted_items_feed_reflection_gate",
    }
    append_jsonl(reviews_path, review)
    return {"status": "recorded", "review": review}


def build_active_checklist(
    candidates_payload: dict[str, Any],
    *,
    reviews_path: Path = DEFAULT_REVIEWS_JSONL,
) -> dict[str, Any]:
    reviewed = overlay_reviews(normalize_candidates(candidates_payload), reviews_path=reviews_path)
    counts = {decision: 0 for decision in sorted(VALID_DECISIONS)}
    active_items: list[dict[str, Any]] = []
    for candidate in reviewed:
        decision = _text(candidate.get("review_decision"))
        if decision in counts:
            counts[decision] += 1
        if decision not in {"accepted", "revised"}:
            continue
        review = candidate.get("review") if isinstance(candidate.get("review"), dict) else {}
        edited = [_text(item) for item in review.get("edited_checklist_items") or [] if _text(item)]
        active_items.append(
            {
                "id": candidate["id"],
                "scope": candidate["scope"],
                "trigger": candidate.get("trigger", ""),
                "topic": candidate.get("topic", ""),
                "source_case_id": candidate.get("source_case_id", ""),
                "query": candidate.get("query", ""),
                "checklist_items": edited if decision == "revised" and edited else candidate["checklist_items"],
                "review_decision": decision,
                "reviewed_at": candidate.get("reviewed_at", ""),
            }
        )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "experience_replay_accepted_reflection_gate_checklist",
        "guardrail": "human-reviewed accepted/revised checklist only; inference-time auxiliary; no scoring, rag, memory, or skill changes",
        "source_candidates_mode": candidates_payload.get("mode", ""),
        "summary": {
            "candidate_count": len(reviewed),
            "active_count": len(active_items),
            "review_counts": counts,
            "unreviewed_count": sum(1 for item in reviewed if not item.get("review_decision")),
        },
        "active_items": active_items,
    }


def render_markdown(active_payload: dict[str, Any], reviewed_candidates: list[dict[str, Any]]) -> str:
    summary = active_payload.get("summary") or {}
    lines = [
        "# Experience Replay Checklist Review",
        "",
        f"- Generated at: `{active_payload.get('generated_at')}`",
        f"- Mode: `{active_payload.get('mode')}`",
        f"- Guardrail: {active_payload.get('guardrail')}",
        f"- Candidates: {summary.get('candidate_count', 0)}",
        f"- Active accepted/revised: {summary.get('active_count', 0)}",
        f"- Unreviewed: {summary.get('unreviewed_count', 0)}",
        f"- Review counts: `{json.dumps(summary.get('review_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Active Reflection Gate Items",
    ]
    if not active_payload.get("active_items"):
        lines.append("- None")
    for item in active_payload.get("active_items") or []:
        lines.append(f"- `{item['id']}` {item.get('topic') or item.get('trigger')} ({item['review_decision']})")
        for checklist_item in item.get("checklist_items") or []:
            lines.append(f"  - {checklist_item}")
    lines.extend(["", "## Review Queue"])
    if not reviewed_candidates:
        lines.append("- None")
    for candidate in reviewed_candidates[:30]:
        lines.append(
            f"- `{candidate['id']}` status={candidate.get('status')} "
            f"scope={candidate.get('scope')} topic={candidate.get('topic') or candidate.get('trigger')}"
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    active_payload: dict[str, Any],
    reviewed_candidates: list[dict[str, Any]],
    *,
    active_json: Path = DEFAULT_ACTIVE_JSON,
    report_md: Path = DEFAULT_REPORT_MD,
) -> None:
    active_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    active_json.write_text(json.dumps(active_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown(active_payload, reviewed_candidates), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-json", type=Path, default=DEFAULT_CANDIDATES_JSON)
    parser.add_argument("--reviews-jsonl", type=Path, default=DEFAULT_REVIEWS_JSONL)
    parser.add_argument("--active-json", type=Path, default=DEFAULT_ACTIVE_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--review-id", default="")
    parser.add_argument("--decision", choices=sorted(VALID_DECISIONS), default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--reviewer", default="user")
    parser.add_argument("--edited-item", action="append", default=[])
    args = parser.parse_args()

    candidates_payload = load_candidates(args.candidates_json.expanduser())
    if args.review_id or args.decision:
        if not args.review_id or not args.decision:
            parser.error("--review-id and --decision must be provided together")
        record_review(
            candidate_id=args.review_id,
            decision=args.decision,
            candidates_payload=candidates_payload,
            note=args.note,
            edited_items=args.edited_item,
            reviewer=args.reviewer,
            reviews_path=args.reviews_jsonl.expanduser(),
        )

    reviewed = overlay_reviews(
        normalize_candidates(candidates_payload),
        reviews_path=args.reviews_jsonl.expanduser(),
    )
    active = build_active_checklist(candidates_payload, reviews_path=args.reviews_jsonl.expanduser())
    write_outputs(
        active,
        reviewed,
        active_json=args.active_json.expanduser(),
        report_md=args.report_md.expanduser(),
    )
    print(
        "experience replay checklist review: "
        f"active={active['summary']['active_count']} unreviewed={active['summary']['unreviewed_count']} "
        f"-> {args.active_json.expanduser()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
