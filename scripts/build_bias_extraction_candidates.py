#!/usr/bin/env python3
"""Build review-only cognitive-bias candidates from judgment_feedback comments.

協議コメント（judgment_feedback.reason）を api/crystallizer/bias_extractor.extract_bias() で
分析し、人間の判断バイアス・スルーされたリスクを「判断資産候補」として抽出する。
scripts/build_autoresearch_judgment_asset_candidates.py と同じ「候補JSONL + stateファイル」
パターンを踏襲し、人間が有用性を確認するまでは判断資産へ昇格させない。

新規コメントのみを処理するため、最後に処理した judgment_feedback.id を
data/bias_extraction_checkpoint.json に保存し、次回実行時はそこから再開する
（--since-id で明示的に上書き可能）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api.crystallizer.bias_extractor import extract_bias

PROJECT_ROOT = _REPO_ROOT
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_DB_PATH = str(DATA_DIR / "lease_data.db")
DEFAULT_OUTPUT_JSONL = DATA_DIR / "bias_extraction_candidates.jsonl"
DEFAULT_STATE_JSON = DATA_DIR / "bias_extraction_candidate_state.json"
DEFAULT_CHECKPOINT_JSON = DATA_DIR / "bias_extraction_checkpoint.json"

MIN_CLAIM_LENGTH = 8

DEFAULT_CANDIDATE_STATE = {
    "use_count": 0,
    "useful_count": 0,
    "rejected_count": 0,
    "neutral_count": 0,
    "last_used_at": "",
    "last_feedback_at": "",
    "verified_status": "unverified",
    "edited_claim": "",
    "edit_count": 0,
    "last_edited_at": "",
}


def _candidate_state(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    state = dict(DEFAULT_CANDIDATE_STATE)
    if raw:
        for key in state:
            if key not in raw:
                continue
            if key.endswith("_count"):
                try:
                    state[key] = max(0, int(raw[key]))
                except (TypeError, ValueError):
                    state[key] = 0
            else:
                state[key] = str(raw[key] or "")
    if state["verified_status"] not in {"unverified", "supported", "contradicted", "unclear"}:
        state["verified_status"] = "unverified"
    return state


def load_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: _candidate_state(v) for k, v in raw.items() if isinstance(k, str) and isinstance(v, dict)}


def write_state(path: Path, candidates: list[dict[str, Any]], existing: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(existing)
    for item in candidates:
        merged.setdefault(str(item["id"]), _candidate_state(item))
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("last_processed_id", 0))
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return 0


def write_checkpoint(path: Path, last_processed_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"last_processed_id": last_processed_id}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_existing_candidates(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    items: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("id"):
            items[str(row["id"])] = row
    return items


def _candidate_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _promotion_status(state: dict[str, Any], asset_quality: str) -> str:
    if asset_quality != "actionable":
        return "not_promoted_too_short"
    if state["verified_status"] == "supported" and int(state["useful_count"]) > 0:
        return "ready_for_promotion"
    if state["verified_status"] == "contradicted" or int(state["rejected_count"]) > int(state["useful_count"]):
        return "rejected_or_deprioritized"
    return "not_promoted"


def _load_comments(db_path: str, *, since_id: int, limit: int) -> list[dict[str, Any]]:
    if not os.path.exists(db_path):
        return []
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, case_id, recorded_at, source, reason, score
                FROM judgment_feedback
                WHERE id > ?
                ORDER BY id
                LIMIT ?
                """,
                (since_id, limit),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {"id": r[0], "case_id": r[1], "recorded_at": r[2], "source": r[3], "reason": r[4], "score": r[5]}
        for r in rows
    ]


def _dismissed_risk_candidates(row: dict[str, Any], analysis: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for risk in analysis.get("dismissed_risks") or []:
        claim = str(risk.get("risk_summary") or "").strip()
        if len(claim) < MIN_CLAIM_LENGTH:
            continue
        candidate_id = _candidate_id("judgment_feedback", str(row["id"]), "dismissed_risk", claim)
        out.append(
            {
                "id": candidate_id,
                "material_type": "cognitive_bias",
                "candidate_type": "dismissed_risk",
                "case_id": row["case_id"],
                "source_table": "judgment_feedback",
                "source_record_id": row["id"],
                "claim": claim,
                "risk_category": risk.get("risk_category", "OTHER"),
                "excuse_pattern": risk.get("excuse_pattern", ""),
                "quoted_text": risk.get("quoted_text", ""),
                "severity": risk.get("severity", "MEDIUM"),
                "decision_context": analysis.get("overall_sentiment", {}),
                "recorded_at": row["recorded_at"],
            }
        )
    return out


def _bias_pattern_candidates(row: dict[str, Any], analysis: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for bias in analysis.get("inferred_biases") or []:
        bias_type = str(bias.get("bias_type") or "NONE")
        if bias_type == "NONE":
            continue
        claim = str(bias.get("explanation") or "").strip()
        if len(claim) < MIN_CLAIM_LENGTH:
            continue
        candidate_id = _candidate_id("judgment_feedback", str(row["id"]), "bias_pattern", bias_type, claim)
        out.append(
            {
                "id": candidate_id,
                "material_type": "cognitive_bias",
                "candidate_type": "bias_pattern",
                "case_id": row["case_id"],
                "source_table": "judgment_feedback",
                "source_record_id": row["id"],
                "claim": claim,
                "bias_type": bias_type,
                "decision_context": analysis.get("overall_sentiment", {}),
                "recorded_at": row["recorded_at"],
            }
        )
    return out


def build_candidates(
    rows: list[dict[str, Any]],
    *,
    states: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        analysis = extract_bias(row["reason"], row["case_id"])
        raw_items = _dismissed_risk_candidates(row, analysis) + _bias_pattern_candidates(row, analysis)
        for item in raw_items:
            if item["id"] in seen_ids:
                continue
            seen_ids.add(item["id"])
            state = _candidate_state(states.get(item["id"]))
            asset_quality = "actionable"
            item.update(
                {
                    "review_status": "candidate",
                    "asset_quality": asset_quality,
                    "promotion_status": _promotion_status(state, asset_quality),
                    **state,
                    "requires_human_use_feedback": True,
                    "requires_result_verification": True,
                    "use_policy": (
                        "抽出されたバイアス指摘は参考情報。個別案件の自動承認・自動否決の根拠にはしない。"
                        "人間が実際の案件で確認し、有用性フィードバックが得られたものだけ判断資産へ昇格を検討する。"
                    ),
                }
            )
            candidates.append(item)
    return candidates


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _markdown_report(candidates: list[dict[str, Any]], *, new_count: int) -> str:
    bias_counts: dict[str, int] = {}
    for item in candidates:
        if item["candidate_type"] == "bias_pattern":
            bias_counts[item["bias_type"]] = bias_counts.get(item["bias_type"], 0) + 1
    lines = [
        "# Bias Extraction Candidates",
        "",
        "## Summary",
        "",
        f"- Total candidates: {len(candidates)} (new this run: {new_count})",
        f"- dismissed_risk: {sum(1 for c in candidates if c['candidate_type'] == 'dismissed_risk')}",
        f"- bias_pattern: {sum(1 for c in candidates if c['candidate_type'] == 'bias_pattern')}",
    ]
    for bias_type, count in sorted(bias_counts.items()):
        lines.append(f"  - {bias_type}: {count}")
    lines += [
        "",
        "## Promotion Policy",
        "",
        "- バイアス指摘は候補のまま。人間が実案件で有用性を確認し verified_status=supported になるまで昇格しない。",
        "- 本抽出は既存スコア・審査ロジックに一切介入しない（気づきレイヤーのみ）。",
        "",
        "## Candidates",
        "",
    ]
    for item in candidates:
        lines += [
            f"### case={item['case_id']} / {item['candidate_type']}",
            "",
            f"- Claim: {item['claim']}",
            f"- Status: {item['review_status']} / {item['promotion_status']}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def write_report(candidates: list[dict[str, Any]], *, new_count: int) -> dict[str, str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    latest_md = REPORTS_DIR / "bias_extraction_candidates_latest.md"
    latest_json = REPORTS_DIR / "bias_extraction_candidates_latest.json"
    md = _markdown_report(candidates, new_count=new_count)
    latest_md.write_text(md, encoding="utf-8")
    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "total_candidates": len(candidates),
        "new_candidates_this_run": new_count,
        "dismissed_risk": sum(1 for c in candidates if c["candidate_type"] == "dismissed_risk"),
        "bias_pattern": sum(1 for c in candidates if c["candidate_type"] == "bias_pattern"),
    }
    latest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"markdown": str(latest_md), "summary_json": str(latest_json)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build review-only cognitive-bias candidates from judgment_feedback comments."
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--limit", type=int, default=50, help="Max new judgment_feedback rows to scan this run")
    parser.add_argument("--since-id", type=int, default=None, help="Override checkpoint; process rows with id > this")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--state", default=str(DEFAULT_STATE_JSON))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_JSON))
    args = parser.parse_args()

    output_path = Path(args.output)
    state_path = Path(args.state)
    checkpoint_path = Path(args.checkpoint)

    since_id = args.since_id if args.since_id is not None else load_checkpoint(checkpoint_path)
    rows = _load_comments(args.db, since_id=since_id, limit=args.limit)

    existing_state = load_state(state_path)
    existing_candidates = load_existing_candidates(output_path)
    new_candidates = build_candidates(rows, states=existing_state)

    merged = dict(existing_candidates)
    for item in new_candidates:
        merged[item["id"]] = item
    all_candidates = list(merged.values())

    write_state(state_path, all_candidates, existing_state)
    write_jsonl(output_path, all_candidates)
    paths = write_report(all_candidates, new_count=len(new_candidates))

    if rows:
        write_checkpoint(checkpoint_path, max(row["id"] for row in rows))

    print(
        json.dumps(
            {
                "scanned_comments": len(rows),
                "new_candidates": len(new_candidates),
                "total_candidates": len(all_candidates),
                "output_jsonl": str(output_path),
                "paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
