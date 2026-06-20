#!/usr/bin/env python3
"""Build privacy-bounded learning candidates from real-case decision differences."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DB = REPO_ROOT / "data" / "lease_data.db"
DEFAULT_REPORT = REPO_ROOT / "reports" / "real_case_difference_learning_latest.json"

_SAFE_INPUT_KEYS = (
    "grade",
    "contract_type",
    "lease_term",
    "acquisition_cost",
    "lease_asset_score",
    "asset_score",
    "customer_type",
    "industry_major",
    "industry_sub",
)

_SIGNAL_DEFINITIONS = {
    "high_score_lost": {
        "where": "sr.total_score >= 70 AND sr.outcome = 'lost'",
        "question": (
            "高スコアでも失注した非スコア要因を確認する。価格・競合・銀行支援・"
            "補助金タイミング・営業導線を、人間が案件記録で判定する。"
        ),
    },
    "low_score_contracted": {
        "where": "sr.total_score < 50 AND sr.outcome = 'contracted'",
        "question": (
            "低スコアでも成約した補強要因を確認する。保証・前受金・銀行支援・"
            "物件換金性・取引実績を、人間が案件記録で判定する。"
        ),
    },
    "high_score_delinquent": {
        "where": "sr.total_score >= 70 AND sr.outcome = 'delinquent'",
        "question": (
            "高スコア後に延滞した見落としを確認する。資金繰り推移・受注残・"
            "返済原資・集中リスク・外部環境を、人間が案件記録で判定する。"
        ),
    },
}


def _open_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return bool(row)


def _anonymous_case_id(case_id: str) -> str:
    digest = hashlib.sha256(f"shion-case:{case_id}".encode("utf-8")).hexdigest()
    return f"case-{digest[:12]}"


def _safe_snapshot(raw: str | None) -> dict[str, Any]:
    try:
        data = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    inputs = data.get("inputs") if isinstance(data, dict) else {}
    if not isinstance(inputs, dict):
        inputs = {}
    if not inputs and isinstance(data, dict):
        inputs = data
    result = {}
    for key in _SAFE_INPUT_KEYS:
        value = inputs.get(key)
        if value not in (None, ""):
            result[key] = value
    return result


def load_explicit_differences(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(conn, "judgment_feedback"):
        return []
    rows = conn.execute(
        """
        SELECT id, case_id, recorded_at, source, model_decision, human_decision,
               reason, score, input_snapshot, evidence_snapshot, review_status
        FROM judgment_feedback
        WHERE changed=1
        ORDER BY id DESC
        """
    ).fetchall()
    result = []
    for row in rows:
        result.append(
            {
                "feedback_id": row["id"],
                "case_ref": _anonymous_case_id(str(row["case_id"])),
                "recorded_at": row["recorded_at"],
                "source": row["source"],
                "model_decision": row["model_decision"],
                "human_decision": row["human_decision"],
                "reason": row["reason"],
                "score": row["score"],
                "input_snapshot": _safe_snapshot(row["input_snapshot"]),
                "evidence_available": bool(row["evidence_snapshot"]),
                "review_status": row["review_status"],
                "eligible_for_training": row["review_status"] == "approved",
            }
        )
    return result


def load_outcome_signals(
    conn: sqlite3.Connection,
    *,
    per_signal_limit: int = 20,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts: dict[str, int] = {}
    candidates: list[dict[str, Any]] = []
    for signal_type, definition in _SIGNAL_DEFINITIONS.items():
        where = definition["where"]
        counts[signal_type] = int(
            conn.execute(
                f"SELECT COUNT(*) FROM screening_records sr WHERE {where}"
            ).fetchone()[0]
            or 0
        )
        rows = conn.execute(
            f"""
            SELECT sr.case_id, sr.total_score, sr.outcome, sr.screened_at,
                   pc.industry_sub, pc.data
            FROM screening_records sr
            LEFT JOIN past_cases pc ON pc.id = sr.case_id
            WHERE {where}
            ORDER BY sr.id DESC
            LIMIT ?
            """,
            (max(1, per_signal_limit),),
        ).fetchall()
        for row in rows:
            snapshot = _safe_snapshot(row["data"])
            if row["industry_sub"] and "industry_sub" not in snapshot:
                snapshot["industry_sub"] = row["industry_sub"]
            candidates.append(
                {
                    "signal_type": signal_type,
                    "case_ref": _anonymous_case_id(str(row["case_id"])),
                    "score": round(float(row["total_score"]), 1),
                    "actual_outcome": row["outcome"],
                    "screened_at": row["screened_at"],
                    "safe_snapshot": snapshot,
                    "review_question": definition["question"],
                    "review_status": "needs_human_review",
                    "eligible_for_training": False,
                    "warning": (
                        "成約・失注は人間の与信判断ラベルではないため、"
                        "理由確認前に教師データへ使用しない。"
                    ),
                }
            )
    return counts, candidates


def build_difference_report(
    conn: sqlite3.Connection,
    *,
    per_signal_limit: int = 20,
) -> dict[str, Any]:
    explicit = load_explicit_differences(conn)
    signal_counts, review_queue = load_outcome_signals(
        conn,
        per_signal_limit=per_signal_limit,
    )
    approved = [
        item for item in explicit if item["eligible_for_training"]
    ]
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "training_ready" if approved else "needs_human_feedback",
        "policy": {
            "explicit_human_correction_required_for_training": True,
            "outcome_signals_are_not_decision_labels": True,
            "pii_exported": False,
        },
        "summary": {
            "explicit_differences": len(explicit),
            "approved_training_candidates": len(approved),
            "pending_feedback_candidates": sum(
                1 for item in explicit if item["review_status"] == "candidate"
            ),
            "outcome_signal_counts": signal_counts,
            "review_queue_size": len(review_queue),
        },
        "approved_training_candidates": approved,
        "explicit_differences": explicit,
        "outcome_review_queue": review_queue,
        "next_action": (
            "画面またはAPIでモデル判断を人間が変更した際に理由を記録し、"
            "レビューでapprovedにした差分だけを紫苑の教材へ昇格する。"
        ),
    }


_LEDGER_PATH = REPO_ROOT / "api" / "rule_engine" / "ledger_rules.json"

_SIGNAL_RISK = {
    "high_score_lost": "medium",
    "low_score_contracted": "medium",
    "high_score_delinquent": "high",
}


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_ledger(path: Path, rules: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _max_rev_num(rules: list[dict[str, Any]]) -> int:
    import re as _re
    max_num = 0
    for r in rules:
        m = _re.match(r"REV-(\d+)", str(r.get("rev_id") or ""))
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num


def push_to_ledger(report: dict[str, Any], ledger_path: Path = _LEDGER_PATH) -> int:
    """outcome_review_queue の内容を ledger_rules.json に pending_review: true で追記する。

    既存の rev_id （DIFF-SIGNAL-* 形式）と重複するシグナル種別はスキップ。
    Returns:
        追記した件数。
    """
    signal_counts: dict[str, int] = (
        report.get("summary", {}).get("outcome_signal_counts") or {}
    )
    if not any(v > 0 for v in signal_counts.values()):
        return 0

    rules = _load_ledger(ledger_path)
    existing_ids = {str(r.get("rev_id") or "") for r in rules}
    base_rev = _max_rev_num(rules)
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    added = 0

    for signal_type, count in signal_counts.items():
        if count == 0:
            continue
        rev_id = f"DIFF-SIGNAL-{signal_type.upper()}"
        if rev_id in existing_ids:
            print(f"[learn_from_case_differences] スキップ（既存）: {rev_id}", flush=True)
            continue
        base_rev += 1
        definition = _SIGNAL_DEFINITIONS.get(signal_type, {})
        rule = {
            "rev_id": rev_id,
            "type": "manual",
            "pending_review": True,
            "category": "score_gap_learning",
            "description": (
                f"[スコア乖離自動検出] {signal_type}: {count}件 — "
                f"{definition.get('question', '')}"
            ),
            "status": "pending_review",
            "source": "learn_from_case_differences",
            "detected_at": now_iso,
            "signal_count": count,
            "affected_files": [],
            "risk": _SIGNAL_RISK.get(signal_type, "medium"),
            "auto_fix_allowed": False,
        }
        rules.append(rule)
        existing_ids.add(rev_id)
        added += 1
        print(
            f"[learn_from_case_differences] 台帳追記: {rev_id} ({count}件)",
            flush=True,
        )

    if added > 0:
        _save_ledger(ledger_path, rules)
    return added


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--per-signal-limit", type=int, default=20)
    args = parser.parse_args()

    with _open_readonly(args.db) as conn:
        report = build_difference_report(
            conn,
            per_signal_limit=max(1, args.per_signal_limit),
        )
    report["db"] = str(args.db)
    _atomic_write_json(args.report, report)

    ledger_added = push_to_ledger(report)
    if ledger_added:
        print(
            f"[learn_from_case_differences] ledger_rules.json に {ledger_added} 件追記",
            flush=True,
        )

    summary = report["summary"]
    signals = summary["outcome_signal_counts"]
    print(
        "[real-case-difference-learning] "
        f"status={report['status']} "
        f"explicit={summary['explicit_differences']} "
        f"approved={summary['approved_training_candidates']} "
        f"high_score_lost={signals['high_score_lost']} "
        f"low_score_contracted={signals['low_score_contracted']} "
        f"high_score_delinquent={signals['high_score_delinquent']} "
        f"review_queue={summary['review_queue_size']}"
    )


if __name__ == "__main__":
    main()
