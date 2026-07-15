#!/usr/bin/env python3
"""Record and visualize daily Judgment Asset Growth Score.

This is a measurement sidecar. It writes only local data/report artifacts and
does not connect judgment assets to RAG, prompts, scoring, GCS, Cloud Run, or
Obsidian.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURATOR_JSON = PROJECT_ROOT / "reports" / "obsidian_curator_latest.json"
DEFAULT_MANA_JSON = PROJECT_ROOT / "reports" / "mana_obsidian_curator_latest.json"
DEFAULT_CANONICAL_JSON = PROJECT_ROOT / "data" / "canonical_judgment_rules.json"
DEFAULT_HISTORY_JSONL = PROJECT_ROOT / "data" / "judgment_asset_growth_history.jsonl"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "judgment_asset_growth_latest.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "judgment_asset_growth_latest.md"

SCORE_WEIGHTS = {
    "coverage": 0.25,
    "reuse": 0.25,
    "judgment_change": 0.20,
    "human_alignment": 0.20,
    "negative_signal": -0.10,
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_history(path: Path) -> list[dict[str, Any]]:
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


def _write_history(path: Path, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _read_history(path)
    target_date = str(snapshot.get("date") or "")
    replaced = False
    updated: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("date") or "") == target_date:
            if not replaced:
                updated.append(snapshot)
                replaced = True
            continue
        updated.append(row)
    if not replaced:
        updated.append(snapshot)
    updated.sort(key=lambda item: str(item.get("date") or ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in updated),
        encoding="utf-8",
    )
    return updated


def _clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(max(0.0, min(100.0, value)) / 100 * width))
    return "█" * filled + "░" * (width - filled)


def _canonical_rules(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    rules = canonical.get("rules")
    if isinstance(rules, list):
        return [item for item in rules if isinstance(item, dict)]
    rules = canonical.get("canonical_rules")
    if isinstance(rules, list):
        return [item for item in rules if isinstance(item, dict)]
    return []


def _count_active_rules(rules: list[dict[str, Any]]) -> int:
    return sum(1 for item in rules if item.get("status") == "active")


def _count_distinct(values: list[Any]) -> int:
    normalized = {str(value).strip() for value in values if str(value).strip()}
    return len(normalized)


def _negative_signal(mana: dict[str, Any], curator: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    status = str(mana.get("status") or curator.get("mana_status") or "missing")
    status_penalty = {
        "allow": 0,
        "watch": 15,
        "hold": 35,
        "stop": 60,
        "missing": 20,
    }.get(status, 25)
    findings = mana.get("findings") if isinstance(mana.get("findings"), list) else []
    mana_review_items = curator.get("mana_review_items") if isinstance(curator.get("mana_review_items"), list) else []
    duplicate_clusters = curator.get("duplicate_clusters") if isinstance(curator.get("duplicate_clusters"), list) else []
    penalty = status_penalty + min(25, len(findings) * 5) + min(20, len(mana_review_items) * 4) + min(15, len(duplicate_clusters) * 2)
    details = {
        "mana_status": status,
        "findings": len(findings),
        "mana_review_items": len(mana_review_items),
        "duplicate_clusters": len(duplicate_clusters),
    }
    return _clamp_score(penalty), details


def build_growth_snapshot(
    *,
    target_date: str,
    curator: dict[str, Any],
    mana: dict[str, Any],
    canonical: dict[str, Any],
) -> dict[str, Any]:
    material_counts = curator.get("material_counts") if isinstance(curator.get("material_counts"), dict) else {}
    inbox_candidates = curator.get("inbox_candidates") if isinstance(curator.get("inbox_candidates"), list) else []
    rules = _canonical_rules(canonical)
    active_rules = _count_active_rules(rules)
    active = [item for item in rules if item.get("status") == "active"]
    all_axes = [axis for item in active for axis in (item.get("risk_axis") or [])]
    all_concepts = [item.get("concept") for item in active]
    all_domains = [domain for item in active for domain in (item.get("domains") or [item.get("domain")])]

    material_type_count = len(material_counts)
    inbox_count = len(inbox_candidates)
    axis_count = _count_distinct(all_axes)
    concept_count = _count_distinct(all_concepts)
    domain_count = _count_distinct(all_domains)
    total_evidence = sum(int(item.get("evidence_count") or 0) for item in active)
    total_user_evidence = sum(int(item.get("user_evidence_count") or 0) for item in active)
    rules_with_axis = sum(1 for item in active if item.get("risk_axis"))
    high_confidence_rules = sum(1 for item in active if float(item.get("confidence") or 0) >= 0.85)

    coverage = _clamp_score(
        material_type_count * 10
        + min(22, inbox_count * 0.9)
        + active_rules * 3
        + axis_count * 4
        + domain_count * 3
    )
    reuse = min(60.0, _clamp_score(active_rules * 4 + total_evidence * 0.7 + total_user_evidence * 2))
    judgment_change = min(
        75.0,
        _clamp_score(rules_with_axis * 5 + concept_count * 3 + high_confidence_rules * 2 + min(15, inbox_count * 0.5)),
    )
    human_alignment = min(70.0, _clamp_score(active_rules * 3 + total_user_evidence * 5 + high_confidence_rules * 2))
    negative, negative_details = _negative_signal(mana, curator)

    score = _clamp_score(
        coverage * SCORE_WEIGHTS["coverage"]
        + reuse * SCORE_WEIGHTS["reuse"]
        + judgment_change * SCORE_WEIGHTS["judgment_change"]
        + human_alignment * SCORE_WEIGHTS["human_alignment"]
        + negative * SCORE_WEIGHTS["negative_signal"]
    )
    components = {
        "coverage": coverage,
        "reuse_proxy": reuse,
        "judgment_change_proxy": judgment_change,
        "human_alignment_proxy": human_alignment,
        "negative_signal": negative,
    }
    return {
        "date": target_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "score_name": "Judgment Asset Growth Score",
        "score": score,
        "weights": SCORE_WEIGHTS,
        "components": components,
        "counts": {
            "materials_count": int(curator.get("materials_count") or sum(int(v or 0) for v in material_counts.values())),
            "material_type_count": material_type_count,
            "inbox_candidates": inbox_count,
            "active_rules": active_rules,
            "risk_axes": axis_count,
            "concepts": concept_count,
            "domains": domain_count,
            "total_evidence": total_evidence,
            "user_evidence": total_user_evidence,
            "high_confidence_rules": high_confidence_rules,
        },
        "negative_details": negative_details,
        "notes": [
            "reuse_proxy, judgment_change_proxy, human_alignment_proxy は現時点の保存証跡からの代理指標。",
            "実利用ログ・結果登録での的中検証がまだ弱い間は、代理指標の上限を抑えて過大評価しない。",
            "ハッカソン中は測定とローカル可視化のみ。RAG・プロンプト・スコアリング・GCS・Cloud Runへ自動接続しない。",
        ],
    }


def build_payload(snapshot: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "mode": "local_measurement_only",
        "guardrail": "no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun_no_obsidian_write",
        "latest": snapshot,
        "history": history[-30:],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    latest = payload["latest"]
    components = latest["components"]
    counts = latest["counts"]
    history = payload.get("history") or []
    lines = [
        "# Judgment Asset Growth Score",
        "",
        "## Current",
        "",
        f"- Date: {latest['date']}",
        f"- Score: {latest['score']}",
        f"- Mode: {payload['mode']}",
        f"- Guardrail: {payload['guardrail']}",
        "",
        "## Components",
        "",
    ]
    labels = [
        ("Coverage", "coverage"),
        ("Reuse proxy", "reuse_proxy"),
        ("Judgment change proxy", "judgment_change_proxy"),
        ("Human alignment proxy", "human_alignment_proxy"),
        ("Negative signal", "negative_signal"),
    ]
    for label, key in labels:
        value = float(components.get(key) or 0)
        lines.append(f"- {label}: `{_bar(value)}` {value:.1f}")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Materials: {counts['materials_count']}",
            f"- Inbox candidates: {counts['inbox_candidates']}",
            f"- Active rules: {counts['active_rules']}",
            f"- Risk axes: {counts['risk_axes']}",
            f"- Concepts: {counts['concepts']}",
            f"- User evidence: {counts['user_evidence']}",
            "",
            "## Trend",
            "",
        ]
    )
    if history:
        for row in history[-14:]:
            score = float(row.get("score") or 0)
            lines.append(f"- {row.get('date')}: `{_bar(score)}` {score:.1f}")
    else:
        lines.append("- No history yet.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for note in latest.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--curator-json", type=Path, default=DEFAULT_CURATOR_JSON)
    parser.add_argument("--mana-json", type=Path, default=DEFAULT_MANA_JSON)
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--history-jsonl", type=Path, default=DEFAULT_HISTORY_JSONL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = build_growth_snapshot(
        target_date=args.date,
        curator=_read_json(args.curator_json),
        mana=_read_json(args.mana_json),
        canonical=_read_json(args.canonical_json),
    )
    history = _write_history(args.history_jsonl, snapshot)
    payload = build_payload(snapshot, history)
    _write_json(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(payload), encoding="utf-8")
    print(f"Judgment Asset Growth Score: {snapshot['score']} ({args.date})")
    print(f"history: {args.history_jsonl}")
    print(f"report: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
