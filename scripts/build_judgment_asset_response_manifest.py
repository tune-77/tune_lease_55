#!/usr/bin/env python3
"""Build a per-response manifest of which judgment assets a Shion review cited.

紫苑レビュー(shion_screening_reviews)の本文から判断資産citationを抽出し、
実際に judgment_asset_usage_feedback.jsonl / judgment_asset_feedback_drops.jsonl
へ記録されたかを突き合わせる read-only レポート。

このレポートは正規判断資産系統(architecture audit の "agents" 層で提案された
manifest)の可視化専用で、回答生成・スコアリング・RAG・プロンプトには一切接続しない。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from judgment_asset_citation import classify_citations  # noqa: E402
from scripts.record_judgment_asset_feedback import load_active_rules  # noqa: E402

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "lease_data.db"
DEFAULT_CANONICAL_JSON = PROJECT_ROOT / "data" / "canonical_judgment_rules.json"
DEFAULT_USAGE_FEEDBACK_JSONL = PROJECT_ROOT / "data" / "judgment_asset_usage_feedback.jsonl"
DEFAULT_FEEDBACK_DROPS_JSONL = PROJECT_ROOT / "data" / "judgment_asset_feedback_drops.jsonl"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "judgment_asset_response_manifest_latest.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "judgment_asset_response_manifest_latest.md"

GUARDRAIL = "review_only_no_promotion_no_prompt_no_scoring_no_gcs_no_cloudrun_no_obsidian_write"


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


def load_reviews(db_path: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    """shion_screening_reviews を新しい順に読む。テーブルが無ければ空リスト。"""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, case_id, review_text, user_feedback, created_at
            FROM shion_screening_reviews
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _review_manifest_entry(
    review: dict[str, Any],
    *,
    active_rules: dict[str, dict[str, Any]],
    usage_rule_ids_by_case: dict[str, set[str]],
    drop_reasons_by_review_id: dict[int, list[str]],
) -> dict[str, Any]:
    review_id = review.get("id")
    case_id = str(review.get("case_id") or "")
    review_text = str(review.get("review_text") or "")
    classification = classify_citations(review_text, active_rules.keys())
    resolved = classification["resolved"]
    cited_assets = [
        {"rule_id": rule_id, "concept": str(active_rules.get(rule_id, {}).get("concept") or "")}
        for rule_id in resolved
    ]
    recorded_case_rule_ids = usage_rule_ids_by_case.get(case_id, set())
    recorded_rule_ids = [rule_id for rule_id in resolved if rule_id in recorded_case_rule_ids]
    unrecorded_rule_ids = [rule_id for rule_id in resolved if rule_id not in recorded_case_rule_ids]
    drop_reasons = drop_reasons_by_review_id.get(review_id, [])

    user_feedback = str(review.get("user_feedback") or "")
    if not user_feedback:
        status = "no_feedback_yet"
    elif not resolved:
        status = "feedback_without_citation"
    elif unrecorded_rule_ids and not recorded_rule_ids:
        status = "citation_not_recorded"
    elif unrecorded_rule_ids:
        status = "partially_recorded"
    else:
        status = "recorded"

    return {
        "review_id": review_id,
        "case_id": case_id,
        "created_at": review.get("created_at"),
        "user_feedback": user_feedback,
        "cited_assets": cited_assets,
        "ambiguous_citation_prefixes": classification["ambiguous"],
        "unresolved_citation_prefixes": classification["unresolved"],
        "recorded_rule_ids": recorded_rule_ids,
        "unrecorded_rule_ids": unrecorded_rule_ids,
        "drop_reasons": drop_reasons,
        "status": status,
    }


def build_manifest(
    *,
    target_date: str,
    reviews: list[dict[str, Any]],
    active_rules: dict[str, dict[str, Any]],
    usage_feedback_rows: list[dict[str, Any]],
    drop_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    usage_rule_ids_by_case: dict[str, set[str]] = defaultdict(set)
    for row in usage_feedback_rows:
        case_id = str(row.get("case_id") or "")
        rule_id = str(row.get("rule_id") or "")
        if case_id and rule_id:
            usage_rule_ids_by_case[case_id].add(rule_id)

    drop_reasons_by_review_id: dict[int, list[str]] = defaultdict(list)
    for row in drop_rows:
        review_id = row.get("review_id")
        reason = str(row.get("reason") or "")
        if review_id is not None and reason:
            drop_reasons_by_review_id[int(review_id)].append(reason)

    entries = [
        _review_manifest_entry(
            review,
            active_rules=active_rules,
            usage_rule_ids_by_case=usage_rule_ids_by_case,
            drop_reasons_by_review_id=drop_reasons_by_review_id,
        )
        for review in reviews
    ]

    status_counts = Counter(entry["status"] for entry in entries)
    summary = {
        "reviews_scanned": len(entries),
        "active_rules": len(active_rules),
        "recorded": status_counts.get("recorded", 0),
        "partially_recorded": status_counts.get("partially_recorded", 0),
        "citation_not_recorded": status_counts.get("citation_not_recorded", 0),
        "feedback_without_citation": status_counts.get("feedback_without_citation", 0),
        "no_feedback_yet": status_counts.get("no_feedback_yet", 0),
        "ambiguous_citations_total": sum(len(entry["ambiguous_citation_prefixes"]) for entry in entries),
        "unresolved_citations_total": sum(len(entry["unresolved_citation_prefixes"]) for entry in entries),
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date": target_date,
        "schema_version": 1,
        "mode": "local_review_only",
        "guardrail": GUARDRAIL,
        "summary": summary,
        "entries": entries,
        "notes": [
            "citation_not_recorded / partially_recorded は本文に判断資産出典があるのに"
            "judgment_asset_usage_feedback.jsonl へ届いていない件。原因調査の起点にする。",
            "ambiguous_citation_prefixes は短縮citationが複数の能動ルールに一致し得た件。"
            "誤帰属を避けるため resolved には含めていない。",
            "このレポートは観測専用。回答生成・スコアリング・判断資産の昇格には一切使わない。",
        ],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Judgment Asset Response Manifest",
        "",
        f"- Date: {payload['date']}",
        f"- Mode: {payload['mode']}",
        f"- Guardrail: {payload['guardrail']}",
        f"- Reviews scanned: {summary['reviews_scanned']} / Active rules: {summary['active_rules']}",
        f"- Recorded: {summary['recorded']} / Partially recorded: {summary['partially_recorded']} / "
        f"Citation not recorded: {summary['citation_not_recorded']}",
        f"- Feedback without citation: {summary['feedback_without_citation']} / "
        f"No feedback yet: {summary['no_feedback_yet']}",
        f"- Ambiguous citations: {summary['ambiguous_citations_total']} / "
        f"Unresolved citations: {summary['unresolved_citations_total']}",
        "",
        "## 要調査（本文に出典があるのに記録されていない）",
        "",
    ]
    gap_entries = [
        entry
        for entry in payload["entries"]
        if entry["status"] in {"citation_not_recorded", "partially_recorded"}
    ]
    if not gap_entries:
        lines.extend(["- None", ""])
    else:
        for entry in gap_entries[:20]:
            assets = "、".join(f"{a['rule_id']}({a['concept']})" for a in entry["cited_assets"]) or "-"
            reasons = "、".join(entry["drop_reasons"]) or "記録なし（drop理由も見つからず）"
            lines.append(f"- review#{entry['review_id']} case={entry['case_id']}: {assets}")
            lines.append(f"  - 未記録rule_id: {entry['unrecorded_rule_ids']} / drop理由: {reasons}")
        lines.append("")
    lines.extend(["## Notes", ""])
    for note in payload.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--usage-feedback-jsonl", type=Path, default=DEFAULT_USAGE_FEEDBACK_JSONL)
    parser.add_argument("--feedback-drops-jsonl", type=Path, default=DEFAULT_FEEDBACK_DROPS_JSONL)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_manifest(
        target_date=args.date,
        reviews=load_reviews(args.db_path, limit=args.limit),
        active_rules=load_active_rules(path=args.canonical_json),
        usage_feedback_rows=_read_jsonl(args.usage_feedback_jsonl),
        drop_rows=_read_jsonl(args.feedback_drops_jsonl),
    )
    _write_json(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(payload), encoding="utf-8")
    print(f"wrote {args.output_md} and {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
