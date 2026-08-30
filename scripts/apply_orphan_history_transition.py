#!/usr/bin/env python3
"""Apply the reviewed orphan-history transition with backup and rollback guards."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_orphan_history_transition import audit_database  # noqa: E402

DEFAULT_DB = PROJECT_ROOT / "data" / "lease_data.db"
TARGET_WHERE = """
    COALESCE(s.record_state, 'active')='active'
    AND s.source='streamlit'
    AND NOT EXISTS (SELECT 1 FROM past_cases p WHERE p.id=s.case_id)
    AND NOT EXISTS (SELECT 1 FROM excluded_grade_cases e WHERE e.id=s.case_id)
"""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_hash(row: sqlite3.Row) -> str:
    payload = {key: row[key] for key in row.keys() if key != "record_state"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _backup_database(source: sqlite3.Connection, backup_path: Path) -> dict[str, Any]:
    if backup_path.exists():
        raise FileExistsError(f"Backup already exists: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(backup_path) as destination:
        source.backup(destination)
    with sqlite3.connect(f"file:{backup_path.resolve().as_posix()}?mode=ro", uri=True) as backup:
        integrity = str(backup.execute("PRAGMA integrity_check").fetchone()[0])
        total = int(backup.execute("SELECT COUNT(*) FROM screening_records").fetchone()[0])
        targets = int(
            backup.execute(f"SELECT COUNT(*) FROM screening_records s WHERE {TARGET_WHERE}").fetchone()[0]
        )
    return {
        "path": str(backup_path.resolve()),
        "integrity_check": integrity,
        "screening_records": total,
        "active_orphan_targets": targets,
        "size_bytes": backup_path.stat().st_size,
        "sha256": _file_sha256(backup_path),
    }


def apply_transition(db_path: Path, backup_path: Path, expected_target: int) -> dict[str, Any]:
    preflight = audit_database(db_path, expected_target)
    if not preflight["summary"]["ready_for_apply"]:
        raise RuntimeError(f"Dry-run guard failed: {preflight['summary']}")

    resolved = db_path.resolve()
    with sqlite3.connect(resolved, timeout=10.0) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        backup = _backup_database(conn, backup_path)
        if backup["integrity_check"] != "ok":
            raise RuntimeError("Backup integrity_check failed")
        if backup["active_orphan_targets"] != expected_target:
            raise RuntimeError("Backup target count differs from approved count")

        conn.execute("BEGIN IMMEDIATE")
        try:
            targets = conn.execute(
                f"SELECT s.* FROM screening_records s WHERE {TARGET_WHERE} ORDER BY s.id"
            ).fetchall()
            if len(targets) != expected_target:
                raise RuntimeError(f"Target count changed before update: {len(targets)} != {expected_target}")
            hashes_before = {int(row["id"]): _row_hash(row) for row in targets}
            target_ids = sorted(hashes_before)

            changes_before = conn.total_changes
            conn.executemany(
                "UPDATE screening_records SET record_state='orphan_history' "
                "WHERE id=? AND COALESCE(record_state, 'active')='active'",
                [(record_id,) for record_id in target_ids],
            )
            changed = conn.total_changes - changes_before
            if changed != expected_target:
                raise RuntimeError(f"Updated row count mismatch: {changed} != {expected_target}")

            placeholders = ",".join("?" for _ in target_ids)
            updated = conn.execute(
                f"SELECT * FROM screening_records WHERE id IN ({placeholders}) ORDER BY id",
                target_ids,
            ).fetchall()
            for row in updated:
                record_id = int(row["id"])
                if row["record_state"] != "orphan_history":
                    raise RuntimeError(f"State verification failed for record {record_id}")
                if _row_hash(row) != hashes_before[record_id]:
                    raise RuntimeError(f"Non-state field changed for record {record_id}")

            remaining_active = int(
                conn.execute(f"SELECT COUNT(*) FROM screening_records s WHERE {TARGET_WHERE}").fetchone()[0]
            )
            classified = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM screening_records s
                     WHERE s.record_state='orphan_history' AND s.source='streamlit'
                       AND NOT EXISTS (SELECT 1 FROM past_cases p WHERE p.id=s.case_id)
                       AND NOT EXISTS (SELECT 1 FROM excluded_grade_cases e WHERE e.id=s.case_id)
                    """
                ).fetchone()[0]
            )
            if remaining_active != 0 or classified != expected_target:
                raise RuntimeError(
                    f"Post-update classification failed: active={remaining_active}, history={classified}"
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        integrity_after = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        total_after = int(conn.execute("SELECT COUNT(*) FROM screening_records").fetchone()[0])
        active_after = int(
            conn.execute("SELECT COUNT(*) FROM screening_records WHERE record_state='active'").fetchone()[0]
        )

    return {
        "schema_version": 1,
        "operation": "orphan_history_transition",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "database_path": str(resolved),
        "backup": backup,
        "updated_rows": changed,
        "record_state": "orphan_history",
        "non_state_fields_changed": 0,
        "screening_records_after": total_after,
        "active_records_after": active_after,
        "orphan_history_after": classified,
        "remaining_active_orphans": remaining_active,
        "integrity_check_after": integrity_after,
        "status": "PASS" if integrity_after == "ok" else "HOLD",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--expected-target", type=int, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        parser.error("This command requires --apply")
    result = apply_transition(args.db, args.backup, args.expected_target)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
