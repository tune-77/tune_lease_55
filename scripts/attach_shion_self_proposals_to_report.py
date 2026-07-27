#!/usr/bin/env python3
"""Attach Shion self proposals to daily improvement reports as a separate section."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_LATEST = REPORTS_DIR / "latest.json"

LAYER_LABELS = {
    "usage_based": "利用ログ由来",
    "feedback_based": "人間反応・判断ログ由来",
    "system_audit_based": "システム監査由来",
}

SOURCES = [
    {
        "path": DATA_DIR / "usage_loop_proposals.jsonl",
        "source": "usage_loop",
        "kind": "画面利用",
        "evidence_layer": "usage_based",
        "summary_keys": ("hypothesis", "evidence", "reason"),
    },
    {
        "path": DATA_DIR / "feedback_pattern_proposals.jsonl",
        "source": "feedback_pattern_loop",
        "kind": "人間反応",
        "evidence_layer": "feedback_based",
        "summary_keys": ("hypothesis", "evidence", "pattern", "suggestion"),
    },
    {
        "path": DATA_DIR / "judgment_divergence_proposals.jsonl",
        "source": "judgment_divergence_loop",
        "kind": "審査判断乖離",
        "evidence_layer": "feedback_based",
        "summary_keys": ("observation", "review_point", "suggestion"),
    },
    {
        "path": DATA_DIR / "outcome_drift_proposals.jsonl",
        "source": "outcome_drift_loop",
        "kind": "実績ドリフト",
        "evidence_layer": "feedback_based",
        "summary_keys": ("observation", "review_point", "suggestion"),
    },
    {
        "path": DATA_DIR / "knowledge_gap_proposals.jsonl",
        "source": "knowledge_gap_loop",
        "kind": "ナレッジ穴探し",
        "evidence_layer": "feedback_based",
        "summary_keys": ("reason", "search_hint"),
    },
    {
        "path": REPORTS_DIR / "lease_system_gap_analysis.json",
        "source": "lease_system_gap_analysis",
        "kind": "システム監査",
        "evidence_layer": "system_audit_based",
        "format": "system_gap_report",
        "summary_keys": ("impact", "recommended_action", "guardrail"),
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _load_system_gap_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    rows = payload.get("gaps") if isinstance(payload.get("gaps"), list) else []
    normalized: list[dict[str, Any]] = []
    generated_at = str(payload.get("generated_at") or "")
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(
            {
                "title": row.get("title") or row.get("id") or "",
                "topic": row.get("id") or row.get("category") or "",
                "status": "system_audit_candidate",
                "priority": row.get("priority") or "",
                "generated_at": generated_at,
                "hypothesis": row.get("impact") or "",
                "evidence": " / ".join(str(x) for x in (row.get("evidence") or []) if str(x).strip()),
                "proposed_change": row.get("recommended_action") or "",
                "success_metric": row.get("suggested_program") or "",
                "verification_plan": "システム監査レポートを再生成し、同一GAPが改善・解消したか確認する。",
                "risk": row.get("guardrail") or "",
                "proposal_schema": "shion_self_hypothesis_v1",
                "human_decision_status": "needs_human_review",
                "impact": row.get("impact") or "",
                "recommended_action": row.get("recommended_action") or "",
                "guardrail": row.get("guardrail") or "",
            }
        )
    return normalized


def _load_source_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    path = source["path"]
    if source.get("format") == "system_gap_report":
        return _load_system_gap_rows(path)
    return _load_jsonl(path)


def _summary(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    parts = []
    for key in keys:
        value = str(item.get(key) or "").strip()
        if value:
            parts.append(value)
    text = " / ".join(parts).strip()
    return text[:240].rstrip()


def _proposal_sort_key(item: dict[str, Any]) -> str:
    return str(item.get("generated_at") or item.get("ts") or "")


def _infer_evidence_layer(source: dict[str, Any]) -> str:
    explicit = str(source.get("evidence_layer") or "").strip()
    if explicit:
        return explicit
    source_name = str(source.get("source") or "")
    if source_name == "usage_loop":
        return "usage_based"
    if source_name in {"lease_system_gap_analysis", "system_audit"}:
        return "system_audit_based"
    return "feedback_based"


def collect_shion_self_proposals(limit: int = 10) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    counts_by_kind: dict[str, int] = {}
    counts_by_layer: dict[str, int] = {key: 0 for key in LAYER_LABELS}

    for source in SOURCES:
        rows = _load_source_rows(source)
        counts_by_kind[source["kind"]] = len(rows)
        evidence_layer = _infer_evidence_layer(source)
        counts_by_layer[evidence_layer] = counts_by_layer.get(evidence_layer, 0) + len(rows)
        for row in rows:
            title = str(row.get("title") or row.get("topic") or "").strip()
            if not title:
                continue
            items.append(
                {
                    "kind": source["kind"],
                    "source": source["source"],
                    "evidence_layer": evidence_layer,
                    "evidence_layer_label": LAYER_LABELS.get(evidence_layer, evidence_layer),
                    "title": title,
                    "status": str(row.get("status") or "proposed"),
                    "priority": str(row.get("priority") or ""),
                    "generated_at": str(row.get("generated_at") or row.get("ts") or ""),
                    "target": str(row.get("target_page") or row.get("topic") or ""),
                    "hypothesis": str(row.get("hypothesis") or row.get("suggestion") or row.get("reason") or "").strip(),
                    "evidence": str(row.get("evidence") or row.get("reason") or "").strip(),
                    "proposed_change": str(row.get("proposed_change") or row.get("suggestion") or "").strip(),
                    "success_metric": str(row.get("success_metric") or "").strip(),
                    "verification_plan": str(row.get("verification_plan") or "").strip(),
                    "risk": str(row.get("risk") or "").strip(),
                    "proposal_schema": str(row.get("proposal_schema") or ""),
                    "human_decision_status": str(row.get("human_decision_status") or row.get("status") or ""),
                    "summary": _summary(row, source["summary_keys"]),
                }
            )

    items.sort(key=_proposal_sort_key, reverse=True)
    return {
        "label": "紫苑の自己提案",
        "note": "通常のneeds_reviewではなく、紫苑がログから出した改善仮説。採用判断は人間が行う。",
        "count": len(items),
        "counts_by_kind": counts_by_kind,
        "counts_by_layer": counts_by_layer,
        "layer_labels": LAYER_LABELS,
        "items": items[: max(0, limit)],
    }


def attach_to_report(path: Path, proposal_section: dict[str, Any]) -> bool:
    report = _load_json(path)
    if not report:
        return False

    section = dict(proposal_section)
    section["attached_at"] = datetime.now().isoformat(timespec="seconds")
    report["shion_self_proposals"] = section
    report["shion_self_proposal_count"] = int(section.get("count") or 0)
    summary = report.setdefault("summary", {})
    if isinstance(summary, dict):
        summary["shion_self_proposal_count"] = int(section.get("count") or 0)
        summary["shion_self_proposal_counts_by_layer"] = section.get("counts_by_layer") or {}
    _write_json(path, report)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, help="Specific daily improvement report JSON.")
    parser.add_argument("--latest", type=Path, default=DEFAULT_LATEST, help="reports/latest.json path.")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    section = collect_shion_self_proposals(limit=args.limit)
    targets = []
    if args.report:
        targets.append(args.report.expanduser())
    if args.latest:
        targets.append(args.latest.expanduser())

    updated = []
    for target in dict.fromkeys(targets):
        if attach_to_report(target, section):
            updated.append(str(target))

    print(f"shion_self_proposals={section['count']} updated={len(updated)}")
    for path in updated:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
