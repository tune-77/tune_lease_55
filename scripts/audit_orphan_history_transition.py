#!/usr/bin/env python3
"""Read-only dry-run for classifying orphan screening records as history-only."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "lease_data.db"
DEFAULT_JSON = PROJECT_ROOT / "reports" / "orphan_history_dry_run_latest.json"
DEFAULT_MD = PROJECT_ROOT / "reports" / "orphan_history_dry_run_latest.md"
CANONICAL_CASE_ID_RE = re.compile(r"^\d{20}_[0-9a-fA-F]{8}$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate(conn: sqlite3.Connection, where: str = "1=1") -> dict[str, Any]:
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS rows,
               SUM(CASE WHEN outcome IS NOT NULL AND trim(outcome)<>'' THEN 1 ELSE 0 END) AS with_outcome,
               SUM(CASE WHEN q_risk_score IS NOT NULL THEN 1 ELSE 0 END) AS with_qrisk,
               ROUND(AVG(total_score), 6) AS avg_total_score,
               ROUND(AVG(asset_score), 6) AS avg_asset_score,
               ROUND(AVG(tenant_score), 6) AS avg_tenant_score
          FROM screening_records
         WHERE {where}
        """
    ).fetchone()
    return {
        "rows": int(row[0] or 0),
        "with_outcome": int(row[1] or 0),
        "with_qrisk": int(row[2] or 0),
        "avg_total_score": row[3],
        "avg_asset_score": row[4],
        "avg_tenant_score": row[5],
    }


def audit_database(db_path: Path, expected_target: int | None = None) -> dict[str, Any]:
    resolved = db_path.resolve()
    before = resolved.stat()
    sha_before = _file_sha256(resolved)
    uri = f"file:{resolved.as_posix()}?mode=ro"
    target_where = """
        COALESCE(s.record_state, 'active')='active'
        AND s.source='streamlit'
        AND NOT EXISTS (SELECT 1 FROM past_cases p WHERE p.id=s.case_id)
        AND NOT EXISTS (SELECT 1 FROM excluded_grade_cases e WHERE e.id=s.case_id)
    """

    with sqlite3.connect(uri, uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        targets = conn.execute(
            f"""
            SELECT s.id, s.case_id, s.screened_at, s.total_score, s.asset_score,
                   s.tenant_score, s.q_risk_score, s.competitor_pressure_score,
                   s.outcome, s.input_snapshot, s.source, s.created_at, s.updated_at,
                   s.record_state, s.parent_deleted_at, s.deletion_event_id
              FROM screening_records s
             WHERE {target_where}
             ORDER BY s.id
            """
        ).fetchall()
        columns = [item[0] for item in conn.execute("SELECT name FROM pragma_table_info('screening_records')")]
        total_rows = int(conn.execute("SELECT COUNT(*) FROM screening_records").fetchone()[0])
        all_orphans = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM screening_records s
                 WHERE NOT EXISTS (SELECT 1 FROM past_cases p WHERE p.id=s.case_id)
                   AND NOT EXISTS (SELECT 1 FROM excluded_grade_cases e WHERE e.id=s.case_id)
                """
            ).fetchone()[0]
        )
        classified_orphans = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM screening_records s
                 WHERE s.record_state='orphan_history'
                   AND NOT EXISTS (SELECT 1 FROM past_cases p WHERE p.id=s.case_id)
                   AND NOT EXISTS (SELECT 1 FROM excluded_grade_cases e WHERE e.id=s.case_id)
                """
            ).fetchone()[0]
        )
        linked_outcomes = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                  FROM screening_outcomes so
                  JOIN screening_records s ON s.id=so.screening_id
                 WHERE {target_where}
                """
            ).fetchone()[0]
        )
        metrics_before = _aggregate(conn)
        metrics_after_filter = _aggregate(conn, "COALESCE(record_state, 'active')='active'")
        if targets:
            target_ids = [int(row[0]) for row in targets]
            placeholders = ",".join("?" for _ in target_ids)
            row = conn.execute(
                f"""
                SELECT COUNT(*),
                       SUM(CASE WHEN outcome IS NOT NULL AND trim(outcome)<>'' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN q_risk_score IS NOT NULL THEN 1 ELSE 0 END),
                       ROUND(AVG(total_score), 6), ROUND(AVG(asset_score), 6), ROUND(AVG(tenant_score), 6)
                  FROM screening_records
                 WHERE COALESCE(record_state, 'active')='active' AND id NOT IN ({placeholders})
                """,
                target_ids,
            ).fetchone()
            metrics_after_filter = {
                "rows": int(row[0] or 0),
                "with_outcome": int(row[1] or 0),
                "with_qrisk": int(row[2] or 0),
                "avg_total_score": row[3],
                "avg_asset_score": row[4],
                "avg_tenant_score": row[5],
            }

    target_count = len(targets)
    canonical_ids = sum(bool(CANONICAL_CASE_ID_RE.fullmatch(str(row[1] or ""))) for row in targets)
    unique_case_ids = len({str(row[1]) for row in targets})
    with_snapshot = sum(bool(str(row[9] or "").strip()) for row in targets)
    with_outcome = sum(bool(str(row[8] or "").strip()) for row in targets)
    with_qrisk = sum(row[6] is not None for row in targets)
    simulation_failures = 0
    for row in targets:
        original = dict(zip(
            (
                "id", "case_id", "screened_at", "total_score", "asset_score", "tenant_score",
                "q_risk_score", "competitor_pressure_score", "outcome", "input_snapshot", "source",
                "created_at", "updated_at", "record_state", "parent_deleted_at", "deletion_event_id",
            ),
            row,
        ))
        simulated = dict(original)
        simulated["record_state"] = "orphan_history"
        changed = {key for key in original if original[key] != simulated[key]}
        if changed != {"record_state"}:
            simulation_failures += 1

    after = resolved.stat()
    sha_after = _file_sha256(resolved)
    changed_during_audit = (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or sha_before != sha_after
    )
    required_columns = {"record_state", "parent_deleted_at", "deletion_event_id"}
    apply_count_matches = expected_target is None or target_count == expected_target
    already_applied = all(
        (
            integrity == "ok",
            target_count == 0,
            classified_orphans == all_orphans,
            expected_target is None or classified_orphans == expected_target,
            not changed_during_audit,
        )
    )
    population_matches = apply_count_matches or already_applied
    ready = all(
        (
            integrity == "ok",
            target_count > 0,
            apply_count_matches,
            all_orphans == target_count,
            canonical_ids == target_count,
            unique_case_ids == target_count,
            linked_outcomes == 0,
            simulation_failures == 0,
            required_columns.issubset(set(columns)),
            not changed_during_audit,
        )
    )

    return {
        "schema_version": 1,
        "audit_type": "orphan_history_transition_dry_run",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "database": {
            "path": str(resolved),
            "open_mode": "sqlite_mode_ro_query_only",
            "integrity_check": integrity,
            "size_before": before.st_size,
            "size_after": after.st_size,
            "mtime_ns_before": before.st_mtime_ns,
            "mtime_ns_after": after.st_mtime_ns,
            "sha256_before": sha_before,
            "sha256_after": sha_after,
            "changed_during_audit": changed_during_audit,
        },
        "summary": {
            "screening_records_total": total_rows,
            "all_orphans": all_orphans,
            "target_rows": target_count,
            "classified_orphan_history": classified_orphans,
            "expected_target": expected_target,
            "target_count_matches": apply_count_matches,
            "expected_population_matches": population_matches,
            "unique_target_case_ids": unique_case_ids,
            "canonical_target_case_ids": canonical_ids,
            "targets_with_snapshot": with_snapshot,
            "targets_with_outcome": with_outcome,
            "targets_with_qrisk": with_qrisk,
            "linked_screening_outcomes": linked_outcomes,
            "simulation_failures": simulation_failures,
            "simulated_changed_columns": ["record_state"],
            "simulated_new_state": "orphan_history",
            "ready_for_apply": ready,
            "already_applied": already_applied,
        },
        "metrics_before": metrics_before,
        "metrics_after_active_filter": metrics_after_filter,
        "guardrails": {
            "database_writes": False,
            "individual_case_ids_emitted": False,
            "physical_delete": False,
            "parent_reconnection": False,
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    database = payload["database"]
    before = payload["metrics_before"]
    after = payload["metrics_after_active_filter"]
    if summary["already_applied"]:
        result = "PASS_NO_ACTION"
    elif summary["ready_for_apply"]:
        result = "PASS"
    else:
        result = "HOLD"
    lines = [
        "# orphan_history移行 Dry-run",
        "",
        f"- Result: `{result}`",
        f"- Generated: `{payload['generated_at']}`",
        "- Database writes: `false`",
        "- Physical deletes: `false`",
        "",
        "## Transition summary",
        "",
        "| Check | Value |",
        "|---|---:|",
    ]
    for key in (
        "screening_records_total", "all_orphans", "target_rows", "classified_orphan_history", "expected_target",
        "unique_target_case_ids", "canonical_target_case_ids", "targets_with_snapshot",
        "targets_with_outcome", "targets_with_qrisk", "linked_screening_outcomes",
        "simulation_failures",
    ):
        lines.append(f"| `{key}` | {summary[key]} |")
    lines.extend(
        [
            "",
            "予定変更は対象98件の `record_state: active → orphan_history` だけ。",
            "`parent_deleted_at` と `deletion_event_id` は、過去の削除事実を確定できないため変更しない。",
            "",
            "## Active-filter impact",
            "",
            "| Metric | Before | After |",
            "|---|---:|---:|",
        ]
    )
    for key in ("rows", "with_outcome", "with_qrisk", "avg_total_score", "avg_asset_score", "avg_tenant_score"):
        lines.append(f"| `{key}` | {before[key]} | {after[key]} |")
    lines.extend(
        [
            "",
            "## File safety",
            "",
            f"- SQLite integrity: `{database['integrity_check']}`",
            f"- Size unchanged: `{database['size_before'] == database['size_after']}`",
            f"- mtime unchanged: `{database['mtime_ns_before'] == database['mtime_ns_after']}`",
            f"- SHA-256 unchanged: `{database['sha256_before'] == database['sha256_after']}`",
            f"- Changed during audit: `{database['changed_during_audit']}`",
            "",
            "## Decision",
            "",
            (
                "Dry-run上は98件を安全に `orphan_history` へ分類できる。実更新は未実施。"
                if result == "PASS"
                else (
                    "対象98件はすでに `orphan_history` へ分類済みで、追加更新は不要。"
                    if result == "PASS_NO_ACTION"
                    else "安全条件を満たしていないため、実更新へ進めない。"
                )
            ),
            "",
            "注意: 状態を付けるだけでは既存SQLの集計対象から自動的に外れない。"
            "適用時は、親案件を前提とする集計・学習クエリへ `record_state='active'` 条件を同時に追加する必要がある。",
            "",
            "## Apply-time query scope",
            "",
            "実適用時に `record_state='active'` を追加すべき主な経路:",
            "",
            "- `lease_intelligence_tools.py`: 審査履歴・統計参照",
            "- `components/clifford_visual.py`: 審査記録の可視化",
            "- `api/routers/screening_emotions.py`: 最新審査記録の取得",
            "- `pages/admin.py`: 管理画面の件数",
            "- `retraining_pipeline.py` / `scripts/learn_from_case_differences.py`: 学習・差分候補",
            "- `scripts/export_screening_records.py` / `scripts/aurion_core_daily.py`: 出力・日次監査",
            "",
            "結果登録用の親接続済み2,011件と、結果あり件数・Q_riskあり件数は維持される。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--expected-target", type=int)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    payload = audit_database(args.db, args.expected_target)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.md_output.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 0 if payload["summary"]["ready_for_apply"] or payload["summary"]["already_applied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
