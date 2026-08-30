#!/usr/bin/env python3
"""Read-only dry-run for backfilling ``past_cases.data.id``.

The script never executes UPDATE/INSERT/DELETE and opens SQLite with
``mode=ro&immutable=1``.  It simulates the proposed JSON change in memory,
verifies that only the top-level ``id`` field changes, and writes audit reports.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "lease_data.db"
DEFAULT_JSON = PROJECT_ROOT / "reports" / "past_cases_json_id_backfill_dry_run_latest.json"
DEFAULT_MD = PROJECT_ROOT / "reports" / "past_cases_json_id_backfill_dry_run_latest.md"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_id(payload: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(payload)
    copied.pop("id", None)
    return copied


def _table_count(conn: sqlite3.Connection, table: str) -> int | None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if not exists:
        return None
    return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def audit_database(db_path: Path) -> dict[str, Any]:
    resolved = db_path.resolve()
    before = resolved.stat()
    sha256_before = _file_sha256(resolved)
    uri = f"file:{resolved.as_posix()}?mode=ro&immutable=1"

    rows_total = 0
    row_ids: list[str] = []
    valid_json = 0
    object_json = 0
    missing_inner_id = 0
    matching_inner_id = 0
    mismatching_inner_id = 0
    simulation_failures = 0
    only_id_changed = 0
    simulated_matching_id = 0
    size_delta_bytes = 0
    missing_by_month: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()

    with sqlite3.connect(uri, uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        rows = conn.execute("SELECT id, timestamp, data FROM past_cases ORDER BY id").fetchall()
        rows_total = len(rows)

        for raw_id, timestamp, raw_data in rows:
            row_id = str(raw_id or "").strip()
            row_ids.append(row_id)
            try:
                payload = json.loads(raw_data)
                valid_json += 1
            except (TypeError, json.JSONDecodeError):
                invalid_reasons["invalid_json"] += 1
                continue
            if not isinstance(payload, dict):
                invalid_reasons["json_root_not_object"] += 1
                continue
            object_json += 1

            inner_id = str(payload.get("id") or "").strip()
            if inner_id:
                if inner_id == row_id:
                    matching_inner_id += 1
                else:
                    mismatching_inner_id += 1
                continue

            missing_inner_id += 1
            month = str(timestamp or "")[:7] or "unknown"
            missing_by_month[month] += 1

            original_without_id = _without_id(payload)
            original_hash = _canonical_hash(original_without_id)
            simulated = copy.deepcopy(payload)
            simulated["id"] = row_id

            if _canonical_hash(_without_id(simulated)) != original_hash:
                simulation_failures += 1
                continue
            if str(simulated.get("id") or "").strip() != row_id:
                simulation_failures += 1
                continue

            changed_keys = {
                key
                for key in set(payload) | set(simulated)
                if payload.get(key) != simulated.get(key)
            }
            if changed_keys != {"id"}:
                simulation_failures += 1
                continue

            only_id_changed += 1
            simulated_matching_id += 1
            before_size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            after_size = len(json.dumps(simulated, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            size_delta_bytes += after_size - before_size

        related_table_counts = {
            table: _table_count(conn, table)
            for table in (
                "screening_records",
                "screening_outcomes",
                "shion_screening_reviews",
                "shion_followup_sessions",
                "judgment_lifecycle_events",
            )
        }

    after = resolved.stat()
    sha256_after = _file_sha256(resolved)
    unique_nonblank_ids = len({item for item in row_ids if item})
    blank_row_ids = sum(not item for item in row_ids)
    duplicate_row_ids = rows_total - unique_nonblank_ids - blank_row_ids
    projected_matching_inner_id = matching_inner_id + simulated_matching_id
    already_consistent = all(
        (
            rows_total > 0,
            valid_json == rows_total,
            object_json == rows_total,
            blank_row_ids == 0,
            duplicate_row_ids == 0,
            mismatching_inner_id == 0,
            missing_inner_id == 0,
            matching_inner_id == rows_total,
        )
    )
    ready_for_backfill = all(
        (
            rows_total > 0,
            valid_json == rows_total,
            object_json == rows_total,
            blank_row_ids == 0,
            duplicate_row_ids == 0,
            mismatching_inner_id == 0,
            missing_inner_id > 0,
            simulation_failures == 0,
            only_id_changed == missing_inner_id,
            projected_matching_inner_id == rows_total,
        )
    )

    return {
        "schema_version": 1,
        "audit_type": "past_cases_json_id_backfill_dry_run",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "database": {
            "path": str(resolved),
            "open_mode": "sqlite_mode_ro_immutable_query_only",
            "size_before": before.st_size,
            "size_after": after.st_size,
            "mtime_ns_before": before.st_mtime_ns,
            "mtime_ns_after": after.st_mtime_ns,
            "sha256_before": sha256_before,
            "sha256_after": sha256_after,
            "changed_during_audit": (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or sha256_before != sha256_after
            ),
        },
        "summary": {
            "rows_total": rows_total,
            "unique_nonblank_row_ids": unique_nonblank_ids,
            "blank_row_ids": blank_row_ids,
            "duplicate_row_ids": duplicate_row_ids,
            "valid_json": valid_json,
            "object_json": object_json,
            "matching_inner_id": matching_inner_id,
            "missing_inner_id": missing_inner_id,
            "mismatching_inner_id": mismatching_inner_id,
            "simulated_rows": only_id_changed,
            "simulation_failures": simulation_failures,
            "only_id_changed": only_id_changed,
            "projected_matching_inner_id": projected_matching_inner_id,
            "projected_missing_inner_id": missing_inner_id - simulated_matching_id,
            "projected_mismatching_inner_id": mismatching_inner_id,
            "projected_json_size_delta_bytes": size_delta_bytes,
            "ready_for_backfill": ready_for_backfill,
            "already_consistent": already_consistent,
        },
        "missing_by_month": dict(sorted(missing_by_month.items())),
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "related_table_counts_unchanged_reference": related_table_counts,
        "guardrails": {
            "database_writes": False,
            "sql_mutations": False,
            "simulated_in_memory_only": True,
            "individual_case_ids_emitted": False,
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    database = payload["database"]
    safe = not database["changed_during_audit"]
    if summary["already_consistent"] and safe:
        result = "PASS_NO_ACTION"
    elif summary["ready_for_backfill"] and safe:
        result = "PASS"
    else:
        result = "HOLD"
    lines = [
        "# past_cases JSON内案件IDバックフィル Dry-run",
        "",
        f"- Result: `{result}`",
        f"- Generated: `{payload['generated_at']}`",
        f"- Open mode: `{database['open_mode']}`",
        "- Database writes: `false`",
        "",
        "## Summary",
        "",
        "| Check | Value |",
        "|---|---:|",
    ]
    for key in (
        "rows_total",
        "unique_nonblank_row_ids",
        "blank_row_ids",
        "duplicate_row_ids",
        "valid_json",
        "object_json",
        "matching_inner_id",
        "missing_inner_id",
        "mismatching_inner_id",
        "simulated_rows",
        "simulation_failures",
        "only_id_changed",
        "projected_matching_inner_id",
        "projected_missing_inner_id",
        "projected_mismatching_inner_id",
        "projected_json_size_delta_bytes",
    ):
        lines.append(f"| `{key}` | {summary[key]} |")
    lines.extend(
        [
            "",
            "## Missing IDs by month",
            "",
            "| Month | Rows |",
            "|---|---:|",
        ]
    )
    for month, count in payload["missing_by_month"].items():
        lines.append(f"| `{month}` | {count} |")
    lines.extend(
        [
            "",
            "## File safety",
            "",
            f"- Size unchanged: `{database['size_before'] == database['size_after']}`",
            f"- mtime unchanged: `{database['mtime_ns_before'] == database['mtime_ns_after']}`",
            f"- SHA-256 unchanged: `{database['sha256_before'] == database['sha256_after']}`",
            f"- Changed during audit: `{database['changed_during_audit']}`",
            "",
            "## Decision",
            "",
            (
                "Dry-run上は、欠損行すべてでトップレベル `id` だけを補完できる。"
                "実更新はまだ行わず、バックアップと明示承認後に実施する。"
                if result == "PASS"
                else (
                    "全行のトップレベル `id` が行IDと一致しており、追加更新は不要。"
                    if result == "PASS_NO_ACTION"
                    else "安全条件を満たしていないため、実更新へ進めない。"
                )
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    payload = audit_database(args.db)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.md_output.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    safe = not payload["database"]["changed_during_audit"]
    return 0 if safe and (payload["summary"]["ready_for_backfill"] or payload["summary"]["already_consistent"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
