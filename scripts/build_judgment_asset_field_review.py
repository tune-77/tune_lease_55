#!/usr/bin/env python3
"""Build a field-review report for active judgment assets.

This is a local review sidecar. It reads canonical judgment assets and manual
field feedback, then writes JSON/Markdown reports. It does not promote, demote,
inject into prompts, change scoring, write to Obsidian, call GCS, or call Cloud
Run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_JSON = PROJECT_ROOT / "data" / "canonical_judgment_rules.json"
DEFAULT_FEEDBACK_JSONL = PROJECT_ROOT / "data" / "judgment_asset_usage_feedback.jsonl"
DEFAULT_NEXT_CASE_TARGETS_JSON = PROJECT_ROOT / "data" / "judgment_asset_next_case_targets.json"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "judgment_asset_field_review_latest.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "judgment_asset_field_review_latest.md"

GUARDRAIL = "review_only_no_promotion_no_prompt_no_scoring_no_gcs_no_cloudrun_no_obsidian_write"
VALID_OUTCOMES = {"used", "helped", "challenged", "rejected", "neutral"}


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _canonical_rules(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    rules = canonical.get("rules")
    if isinstance(rules, list):
        return [item for item in rules if isinstance(item, dict)]
    rules = canonical.get("canonical_rules")
    if isinstance(rules, list):
        return [item for item in rules if isinstance(item, dict)]
    return []


def active_rules(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for rule in _canonical_rules(canonical):
        if rule.get("status") != "active":
            continue
        if rule.get("private") is True:
            continue
        if not str(rule.get("id") or "").strip():
            continue
        rules.append(rule)
    return sorted(rules, key=lambda item: str(item.get("concept") or item.get("id") or ""))


def _feedback_rule_id(row: dict[str, Any]) -> str:
    for key in ("rule_id", "judgment_asset_id", "asset_id", "id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _feedback_outcome(row: dict[str, Any]) -> str:
    outcome = str(row.get("outcome") or row.get("status") or "").strip().lower()
    if outcome in VALID_OUTCOMES:
        return outcome
    if row.get("helped") is True:
        return "helped"
    if row.get("challenged") is True or row.get("wrong") is True:
        return "challenged"
    if row.get("used") is True:
        return "used"
    return ""


def _is_simulation_feedback(row: dict[str, Any]) -> bool:
    source = str(row.get("source") or "").strip().lower()
    case_id = str(row.get("case_id") or row.get("case") or "").strip().lower()
    return source == "simulation" or case_id.startswith("sim-")


def _short(value: Any, length: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= length:
        return text
    return text[: length - 1] + "..."


def _last_signal(rows: list[dict[str, Any]]) -> dict[str, str]:
    if not rows:
        return {"case_id": "", "used_at": "", "note": ""}
    row = sorted(rows, key=lambda item: str(item.get("used_at") or item.get("timestamp") or ""))[-1]
    return {
        "case_id": str(row.get("case_id") or row.get("case") or "").strip(),
        "used_at": str(row.get("used_at") or row.get("timestamp") or row.get("date") or "").strip(),
        "note": str(row.get("note") or "").strip(),
    }


def _review_bucket(counts: Counter[str]) -> tuple[str, str]:
    helped = counts.get("helped", 0)
    challenged = counts.get("challenged", 0)
    rejected = counts.get("rejected", 0)
    neutral = counts.get("neutral", 0)
    used = counts.get("used", 0)
    negative = challenged + rejected
    total = helped + neutral + used + negative
    if negative:
        return "review", "challenged/rejected があるため文面・適用条件を見直す"
    if helped:
        return "grow", "helped があるため再利用候補として伸ばす"
    if total == 0:
        return "sleeping", "active だが実利用フィードバックが未記録"
    return "hold", "使われているが有効性シグナルはまだ弱い"


def _priority_score(item: dict[str, Any]) -> float:
    confidence = item.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    user_evidence = int(item.get("user_evidence_count") or 0)
    return round(float(confidence) * 10 + min(user_evidence, 5), 3)


def _fixed_next_case_targets(
    target_config: dict[str, Any],
    rules_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    targets = target_config.get("targets")
    if not isinstance(targets, list):
        return []

    fixed: list[dict[str, Any]] = []
    for row in targets:
        if not isinstance(row, dict):
            continue
        rule_id = str(row.get("rule_id") or "").strip()
        if not rule_id or rule_id not in rules_by_id:
            continue
        item = rules_by_id[rule_id]
        fixed.append(
            {
                "rule_id": item.get("rule_id"),
                "concept": item.get("concept"),
                "statement": item.get("statement"),
                "reason": str(row.get("reason") or "Userが次の実案件で試す対象として固定").strip(),
                "priority_score": _priority_score(item),
                "fixed": True,
                "fixed_at": str(target_config.get("fixed_at") or "").strip(),
                "owner": str(target_config.get("owner") or "").strip(),
            }
        )
    return fixed


def _build_action_plan(
    buckets: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
    next_case_target_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sleeping = sorted(buckets.get("sleeping", []), key=_priority_score, reverse=True)
    grow = sorted(buckets.get("grow", []), key=lambda item: int((item.get("counts") or {}).get("helped") or 0), reverse=True)
    review = sorted(
        buckets.get("review", []),
        key=lambda item: int((item.get("counts") or {}).get("challenged") or 0) + int((item.get("counts") or {}).get("rejected") or 0),
        reverse=True,
    )
    all_items = grow + review + sleeping + buckets.get("hold", [])
    rules_by_id = {str(item.get("rule_id") or ""): item for item in all_items}
    next_case_targets = _fixed_next_case_targets(next_case_target_config or {}, rules_by_id)
    selection_mode = "fixed" if next_case_targets else "auto"
    if not next_case_targets:
        target_rules = sleeping[:3] or grow[:3] or review[:3]
        next_case_targets = [
            {
                "rule_id": item.get("rule_id"),
                "concept": item.get("concept"),
                "statement": item.get("statement"),
                "reason": (
                    "未使用だが confidence/user evidence が高いため、次の実案件で1回だけ意識して試す"
                    if item.get("bucket") == "sleeping"
                    else item.get("reason")
                ),
                "priority_score": _priority_score(item),
                "fixed": False,
            }
            for item in target_rules
        ]
    return {
        "label": "Field Feedback Action Plan",
        "status": "needs_real_case_feedback" if summary.get("sleeping") else "ok",
        "selection_mode": selection_mode,
        "real_case_feedback_gap": int(summary.get("sleeping") or 0),
        "next_case_targets": next_case_targets,
        "feedback_template": {
            "outcomes": ["helped", "challenged", "neutral", "rejected", "used"],
            "command": "python scripts/record_judgment_asset_feedback.py <rule_id> <outcome> --case-id <case_id> --note '<what changed>' --source real_case --recompute-growth",
            "note_rule": "note には『回答/確認/稟議条件の何が変わったか』を1文で残す。",
        },
        "cleanup_rule": "sleeping は削除候補ではなく、次回実案件で試す候補。challenged/rejected が出たものだけ文面・適用条件を見直す。",
    }


def build_review(
    *,
    target_date: str,
    canonical: dict[str, Any],
    feedback_rows: list[dict[str, Any]] | None = None,
    include_simulation: bool = False,
    next_case_target_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feedback_rows = feedback_rows or []
    rules = active_rules(canonical)
    active_ids = {str(rule.get("id") or "").strip() for rule in rules}
    rows_by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown_feedback = 0
    invalid_feedback = 0
    simulation_feedback = 0

    for row in feedback_rows:
        if _is_simulation_feedback(row):
            simulation_feedback += 1
            if not include_simulation:
                continue
        rule_id = _feedback_rule_id(row)
        outcome = _feedback_outcome(row)
        if not rule_id or not outcome:
            invalid_feedback += 1
            continue
        if rule_id not in active_ids:
            unknown_feedback += 1
            continue
        rows_by_rule[rule_id].append({**row, "outcome": outcome})

    buckets: dict[str, list[dict[str, Any]]] = {"grow": [], "review": [], "sleeping": [], "hold": []}
    for rule in rules:
        rule_id = str(rule.get("id") or "").strip()
        rule_rows = rows_by_rule.get(rule_id, [])
        counts: Counter[str] = Counter(str(row.get("outcome") or "") for row in rule_rows)
        bucket, reason = _review_bucket(counts)
        signal = _last_signal(rule_rows)
        item = {
            "rule_id": rule_id,
            "concept": str(rule.get("concept") or "").strip(),
            "statement": str(rule.get("canonical_statement") or "").strip(),
            "bucket": bucket,
            "reason": reason,
            "counts": {
                "used": int(counts.get("used", 0) + counts.get("helped", 0) + counts.get("challenged", 0) + counts.get("rejected", 0) + counts.get("neutral", 0)),
                "helped": int(counts.get("helped", 0)),
                "challenged": int(counts.get("challenged", 0)),
                "rejected": int(counts.get("rejected", 0)),
                "neutral": int(counts.get("neutral", 0)),
            },
            "last_signal": signal,
            "confidence": rule.get("confidence"),
            "user_evidence_count": int(rule.get("user_evidence_count") or 0),
        }
        buckets[bucket].append(item)

    summary = {
        "active_rules": len(rules),
        "grow": len(buckets["grow"]),
        "review": len(buckets["review"]),
        "sleeping": len(buckets["sleeping"]),
        "hold": len(buckets["hold"]),
        "unknown_feedback": unknown_feedback,
        "invalid_feedback": invalid_feedback,
        "simulation_feedback": simulation_feedback,
        "include_simulation": include_simulation,
    }
    action_plan = _build_action_plan(buckets, summary, next_case_target_config)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date": target_date,
        "schema_version": 1,
        "mode": "local_review_only",
        "guardrail": GUARDRAIL,
        "summary": summary,
        "buckets": buckets,
        "action_plan": action_plan,
        "notes": [
            "この棚卸しは実利用フィードバックの見える化だけを行う。",
            "source=simulation または sim-* case は既定では試運転として除外する。",
            "grow は昇格ではなく、次回案件で優先して試す候補。",
            "review は自動修正せず、人間が文面・適用条件・使わない条件を見る。",
            "sleeping は削除候補ではなく、実案件でまだ試されていない active ルール。",
        ],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    buckets = payload["buckets"]
    labels = [
        ("伸ばす", "grow"),
        ("見直す", "review"),
        ("眠ってる", "sleeping"),
        ("保留", "hold"),
    ]
    lines = [
        "# Judgment Asset Field Review",
        "",
        f"- Date: {payload['date']}",
        f"- Mode: {payload['mode']}",
        f"- Guardrail: {payload['guardrail']}",
        f"- Active rules: {summary['active_rules']}",
        f"- Grow: {summary['grow']} / Review: {summary['review']} / Sleeping: {summary['sleeping']} / Hold: {summary['hold']}",
        f"- Simulation feedback: {summary.get('simulation_feedback', 0)} / included: {summary.get('include_simulation', False)}",
        f"- Unknown feedback rows: {summary['unknown_feedback']}",
        "",
    ]
    for label, key in labels:
        lines.extend([f"## {label}", ""])
        items = buckets.get(key) if isinstance(buckets.get(key), list) else []
        if not items:
            lines.extend(["- None", ""])
            continue
        for item in items:
            counts = item.get("counts") if isinstance(item.get("counts"), dict) else {}
            signal = item.get("last_signal") if isinstance(item.get("last_signal"), dict) else {}
            lines.append(f"### {item.get('concept') or item.get('rule_id')}")
            lines.append(f"- Rule ID: `{item.get('rule_id')}`")
            lines.append(f"- Statement: {_short(item.get('statement'))}")
            lines.append(
                "- Feedback: "
                f"used={int(counts.get('used') or 0)}, "
                f"helped={int(counts.get('helped') or 0)}, "
                f"challenged={int(counts.get('challenged') or 0)}, "
                f"rejected={int(counts.get('rejected') or 0)}, "
                f"neutral={int(counts.get('neutral') or 0)}"
            )
            lines.append(f"- Reason: {item.get('reason')}")
            if signal.get("case_id") or signal.get("note"):
                lines.append(f"- Last signal: {signal.get('case_id') or '-'} / {_short(signal.get('note'), 80)}")
            lines.append("")
    action_plan = payload.get("action_plan") if isinstance(payload.get("action_plan"), dict) else {}
    targets = action_plan.get("next_case_targets") if isinstance(action_plan.get("next_case_targets"), list) else []
    lines.extend(["## Next Real Case Feedback", ""])
    if action_plan.get("selection_mode") == "fixed":
        lines.extend(["- Selection: fixed by User for the next real case", ""])
    if targets:
        for item in targets:
            lines.append(f"- `{item.get('rule_id')}` {item.get('concept')}: {_short(item.get('statement'), 100)}")
            lines.append(f"  - Why: {_short(item.get('reason'), 100)}")
    else:
        lines.append("- None")
    template = action_plan.get("feedback_template") if isinstance(action_plan.get("feedback_template"), dict) else {}
    if template.get("command"):
        lines.extend(["", "### Feedback command", "", f"`{template.get('command')}`", ""])
    lines.extend(["## Notes", ""])
    for note in payload.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--feedback-jsonl", type=Path, default=DEFAULT_FEEDBACK_JSONL)
    parser.add_argument("--next-case-targets-json", type=Path, default=DEFAULT_NEXT_CASE_TARGETS_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--include-simulation",
        action="store_true",
        help="Include source=simulation / sim-* feedback in buckets for trial review",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_review(
        target_date=args.date,
        canonical=_read_json(args.canonical_json),
        feedback_rows=_read_jsonl(args.feedback_jsonl),
        include_simulation=args.include_simulation,
        next_case_target_config=_read_json(args.next_case_targets_json),
    )
    _write_json(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(payload), encoding="utf-8")
    print(f"wrote {args.output_md} and {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
