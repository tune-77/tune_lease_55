#!/usr/bin/env python3
"""Evaluate whether Shion has grown over an operating period.

This is a local reporting sidecar. It reads existing measurement artifacts and
feedback ledgers, then writes a period-level judgment report. It does not write
to Obsidian, connect to RAG, change prompts, promote rules, or alter scoring.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROWTH_HISTORY_JSONL = PROJECT_ROOT / "data" / "judgment_asset_growth_history.jsonl"
DEFAULT_GROWTH_LATEST_JSON = PROJECT_ROOT / "reports" / "judgment_asset_growth_latest.json"
DEFAULT_FEEDBACK_JSONL = PROJECT_ROOT / "data" / "judgment_asset_usage_feedback.jsonl"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "shion_growth_evaluation_latest.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "shion_growth_evaluation_latest.md"

JUDGMENT_LABELS = {
    "grown": "育った",
    "partial": "育っている途中",
    "inventory_only": "在庫は増えたが実戦検証不足",
    "insufficient": "判定保留",
    "regressed": "後退・要点検",
}


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


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return None


def _period_rows(rows: list[dict[str, Any]], start_date: str, end_date: str) -> list[dict[str, Any]]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if not start or not end:
        return []
    selected = []
    for row in rows:
        row_date = _parse_date(str(row.get("date") or ""))
        if row_date and start <= row_date <= end:
            selected.append(row)
    return sorted(selected, key=lambda item: str(item.get("date") or ""))


def _latest_snapshot_from_report(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    latest = report.get("latest")
    return latest if isinstance(latest, dict) else {}


def _score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _component(row: dict[str, Any], key: str) -> float:
    components = row.get("components") if isinstance(row.get("components"), dict) else {}
    return _score(components.get(key))


def _count(row: dict[str, Any], key: str) -> int:
    counts = row.get("counts") if isinstance(row.get("counts"), dict) else {}
    try:
        return int(counts.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _feedback_outcome(row: dict[str, Any]) -> str:
    return str(row.get("outcome") or row.get("status") or "").strip().lower()


def _feedback_date(row: dict[str, Any]) -> date | None:
    for key in ("used_at", "timestamp", "date"):
        value = str(row.get(key) or "").strip()
        if value:
            parsed = _parse_date(value)
            if parsed:
                return parsed
    return None


def _feedback_summary(rows: list[dict[str, Any]], start_date: str, end_date: str) -> dict[str, Any]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    totals = {"used": 0, "helped": 0, "challenged": 0, "neutral": 0, "rejected": 0}
    rule_ids: set[str] = set()
    case_ids: set[str] = set()
    if not start or not end:
        return {"totals": totals, "distinct_rules": 0, "distinct_cases": 0}
    for row in rows:
        used_on = _feedback_date(row)
        if not used_on or not (start <= used_on <= end):
            continue
        outcome = _feedback_outcome(row)
        if outcome not in totals:
            continue
        totals["used"] += 1
        if outcome != "used":
            totals[outcome] += 1
        rule_id = str(row.get("rule_id") or row.get("judgment_asset_id") or "").strip()
        case_id = str(row.get("case_id") or row.get("case") or "").strip()
        if rule_id:
            rule_ids.add(rule_id)
        if case_id:
            case_ids.add(case_id)
    return {
        "totals": totals,
        "distinct_rules": len(rule_ids),
        "distinct_cases": len(case_ids),
    }


def _trend(values: list[float]) -> dict[str, float]:
    if not values:
        return {"first": 0.0, "last": 0.0, "delta": 0.0, "average": 0.0}
    first = round(values[0], 1)
    last = round(values[-1], 1)
    average = round(sum(values) / len(values), 1)
    return {"first": first, "last": last, "delta": round(last - first, 1), "average": average}


def _bar(value: float, width: int = 20) -> str:
    bounded = max(0.0, min(100.0, value))
    filled = int(round(bounded / 100 * width))
    return "█" * filled + "░" * (width - filled)


def _grade_dimension(score: float) -> str:
    if score >= 75:
        return "strong"
    if score >= 50:
        return "moderate"
    if score > 0:
        return "weak"
    return "none"


def build_growth_evaluation(
    *,
    start_date: str,
    end_date: str,
    growth_rows: list[dict[str, Any]],
    latest_snapshot: dict[str, Any] | None = None,
    feedback_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    period = _period_rows(growth_rows, start_date, end_date)
    if not period and latest_snapshot:
        latest_date = str(latest_snapshot.get("date") or "")
        if start_date <= latest_date <= end_date:
            period = [latest_snapshot]
    feedback = _feedback_summary(feedback_rows or [], start_date, end_date)

    if not period:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "schema_version": 1,
            "mode": "local_measurement_only",
            "guardrail": "no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun_no_obsidian_write",
            "period": {"start_date": start_date, "end_date": end_date, "days_measured": 0},
            "judgment": {
                "code": "insufficient",
                "label": JUDGMENT_LABELS["insufficient"],
                "score": 0.0,
                "summary": "期間内の測定履歴がないため、紫苑が育ったかは判定できない。",
            },
            "dimensions": {},
            "evidence": {"feedback": feedback, "history_points": 0},
            "next_actions": ["日次の Judgment Asset Growth Score を先に記録する。"],
        }

    scores = [_score(row.get("score")) for row in period]
    field_values = [_component(row, "field_validation") for row in period]
    reuse_values = [_component(row, "reuse_proxy") for row in period]
    judgment_change_values = [_component(row, "judgment_change_proxy") for row in period]
    alignment_values = [_component(row, "human_alignment_proxy") for row in period]
    negative_values = [_component(row, "negative_signal") for row in period]

    first = period[0]
    last = period[-1]
    active_rules_delta = _count(last, "active_rules") - _count(first, "active_rules")
    user_evidence_delta = _count(last, "user_evidence") - _count(first, "user_evidence")
    concepts_delta = _count(last, "concepts") - _count(first, "concepts")
    risk_axes_delta = _count(last, "risk_axes") - _count(first, "risk_axes")

    feedback_totals = feedback["totals"]
    field_last = field_values[-1]
    reuse_last = reuse_values[-1]
    judgment_change_last = judgment_change_values[-1]
    alignment_last = alignment_values[-1]
    negative_last = negative_values[-1]
    score_trend = _trend(scores)

    inventory_score = max(
        0.0,
        min(
            100.0,
            _count(last, "active_rules") * 5
            + _count(last, "risk_axes") * 6
            + _count(last, "concepts") * 3
            + max(0, user_evidence_delta) * 4
            + max(0, active_rules_delta) * 6,
        ),
    )
    reuse_score = min(100.0, reuse_last + min(20, feedback_totals["used"] * 4))
    judgment_change_score = min(
        100.0,
        judgment_change_last + max(0, concepts_delta) * 3 + max(0, risk_axes_delta) * 4,
    )
    field_score = max(
        0.0,
        min(
            100.0,
            field_last
            + feedback_totals["helped"] * 12
            + feedback_totals["neutral"] * 3
            - feedback_totals["challenged"] * 10
            - feedback_totals["rejected"] * 14,
        ),
    )
    alignment_score = min(100.0, alignment_last + max(0, user_evidence_delta) * 3)
    noise_score = max(0.0, 100.0 - negative_last)

    overall = round(
        inventory_score * 0.15
        + reuse_score * 0.20
        + judgment_change_score * 0.20
        + field_score * 0.25
        + alignment_score * 0.10
        + noise_score * 0.10,
        1,
    )

    field_has_evidence = feedback_totals["used"] > 0 or field_last > 0
    regression = score_trend["delta"] <= -10 or negative_last >= 75
    if regression:
        code = "regressed"
    elif overall >= 70 and field_has_evidence and feedback_totals["helped"] >= feedback_totals["challenged"] + feedback_totals["rejected"]:
        code = "grown"
    elif inventory_score >= 50 and reuse_score >= 45 and judgment_change_score >= 45 and field_has_evidence:
        code = "partial"
    elif inventory_score >= 50 and not field_has_evidence:
        code = "inventory_only"
    else:
        code = "insufficient"

    if code == "grown":
        summary = "判断資産が増え、再利用され、実案件・人間反応で効いた証跡もある。"
    elif code == "partial":
        summary = "判断資産と再利用の兆候はあるが、実戦検証の厚みはまだ十分ではない。"
    elif code == "inventory_only":
        summary = "判断資産の在庫と整理は進んだが、実案件で効いた証跡が不足している。"
    elif code == "regressed":
        summary = "期間内でスコア低下またはノイズ増加が強く、成長より点検を優先する状態。"
    else:
        summary = "測定点または成長証跡が不足しており、育ったとはまだ言えない。"

    dimensions = {
        "inventory": {
            "score": round(inventory_score, 1),
            "grade": _grade_dimension(inventory_score),
            "active_rules_delta": active_rules_delta,
            "user_evidence_delta": user_evidence_delta,
            "concepts_delta": concepts_delta,
            "risk_axes_delta": risk_axes_delta,
        },
        "reuse": {"score": round(reuse_score, 1), "grade": _grade_dimension(reuse_score)},
        "judgment_change": {
            "score": round(judgment_change_score, 1),
            "grade": _grade_dimension(judgment_change_score),
        },
        "field_validation": {
            "score": round(field_score, 1),
            "grade": _grade_dimension(field_score),
            "used": feedback_totals["used"],
            "helped": feedback_totals["helped"],
            "challenged": feedback_totals["challenged"],
            "rejected": feedback_totals["rejected"],
        },
        "human_alignment": {"score": round(alignment_score, 1), "grade": _grade_dimension(alignment_score)},
        "noise_control": {
            "score": round(noise_score, 1),
            "grade": _grade_dimension(noise_score),
            "negative_signal": round(negative_last, 1),
        },
    }

    next_actions: list[str] = []
    if not field_has_evidence:
        next_actions.append("紫苑レビューで使われた判断資産に used/helped/challenged/rejected を記録する。")
    if dimensions["reuse"]["grade"] in {"weak", "none"}:
        next_actions.append("昇格済み判断資産が次の回答・レビューで想起されたかをログ化する。")
    if dimensions["noise_control"]["grade"] in {"weak", "none"}:
        next_actions.append("Mana/curator の指摘を確認し、重複・古い前提・過剰なルールを棚卸しする。")
    if not next_actions:
        next_actions.append("次月は helped/challenged の実案件照合を増やし、在庫量より効果率を見る。")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "mode": "local_measurement_only",
        "guardrail": "no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun_no_obsidian_write",
        "period": {
            "start_date": start_date,
            "end_date": end_date,
            "days_measured": len(period),
            "first_measured_date": str(first.get("date") or ""),
            "last_measured_date": str(last.get("date") or ""),
        },
        "judgment": {
            "code": code,
            "label": JUDGMENT_LABELS[code],
            "score": overall,
            "summary": summary,
        },
        "dimensions": dimensions,
        "evidence": {
            "score_trend": score_trend,
            "feedback": feedback,
            "history_points": len(period),
            "first_counts": first.get("counts") if isinstance(first.get("counts"), dict) else {},
            "last_counts": last.get("counts") if isinstance(last.get("counts"), dict) else {},
        },
        "next_actions": next_actions,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    judgment = payload["judgment"]
    period = payload["period"]
    dimensions = payload.get("dimensions") if isinstance(payload.get("dimensions"), dict) else {}
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    feedback = evidence.get("feedback") if isinstance(evidence.get("feedback"), dict) else {}
    feedback_totals = feedback.get("totals") if isinstance(feedback.get("totals"), dict) else {}
    lines = [
        "# Shion Growth Evaluation",
        "",
        "## Judgment",
        "",
        f"- Period: {period.get('start_date')} to {period.get('end_date')}",
        f"- Days measured: {period.get('days_measured')}",
        f"- Result: {judgment.get('label')} ({judgment.get('score')})",
        f"- Summary: {judgment.get('summary')}",
        f"- Mode: {payload.get('mode')}",
        f"- Guardrail: {payload.get('guardrail')}",
        "",
        "## Dimensions",
        "",
    ]
    labels = [
        ("Inventory", "inventory"),
        ("Reuse", "reuse"),
        ("Judgment change", "judgment_change"),
        ("Field validation", "field_validation"),
        ("Human alignment", "human_alignment"),
        ("Noise control", "noise_control"),
    ]
    for label, key in labels:
        item = dimensions.get(key) if isinstance(dimensions.get(key), dict) else {}
        value = _score(item.get("score"))
        lines.append(f"- {label}: `{_bar(value)}` {value:.1f} / {item.get('grade') or 'none'}")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- Score delta: {evidence.get('score_trend', {}).get('delta', 0.0)}",
            f"- Feedback used: {int(feedback_totals.get('used') or 0)}",
            f"- Feedback helped: {int(feedback_totals.get('helped') or 0)}",
            f"- Feedback challenged: {int(feedback_totals.get('challenged') or 0)}",
            f"- Feedback rejected: {int(feedback_totals.get('rejected') or 0)}",
            f"- Distinct rules with feedback: {int(feedback.get('distinct_rules') or 0)}",
            f"- Distinct cases with feedback: {int(feedback.get('distinct_cases') or 0)}",
            "",
            "## Next Actions",
            "",
        ]
    )
    for action in payload.get("next_actions") or []:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def _default_start(end_date: str) -> str:
    parsed = _parse_date(end_date) or date.today()
    return (parsed - timedelta(days=29)).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--start-date", default="")
    parser.add_argument("--growth-history-jsonl", type=Path, default=DEFAULT_GROWTH_HISTORY_JSONL)
    parser.add_argument("--growth-latest-json", type=Path, default=DEFAULT_GROWTH_LATEST_JSON)
    parser.add_argument("--feedback-jsonl", type=Path, default=DEFAULT_FEEDBACK_JSONL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    end_date = args.end_date
    start_date = args.start_date or _default_start(end_date)
    payload = build_growth_evaluation(
        start_date=start_date,
        end_date=end_date,
        growth_rows=_read_jsonl(args.growth_history_jsonl),
        latest_snapshot=_latest_snapshot_from_report(args.growth_latest_json),
        feedback_rows=_read_jsonl(args.feedback_jsonl),
    )
    _write_json(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(payload), encoding="utf-8")
    print(
        "Shion Growth Evaluation: "
        f"{payload['judgment']['label']} ({payload['judgment']['score']}) "
        f"{start_date}..{end_date}"
    )
    print(f"report: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
