#!/usr/bin/env python3
"""Backfill missing ``past_cases.data.id`` values with transactional safeguards."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "lease_data.db"
RELATED_TABLES = (
    "screening_records",
    "screening_outcomes",
    "shion_screening_reviews",
    "shion_followup_sessions",
    "judgment_lifecycle_events",
)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _without_id(payload: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(payload)
    copied.pop("id", None)
    return copied


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_count(conn: sqlite3.Connection, table: str) -> int | None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not exists:
        return None
    return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _database_state(conn: sqlite3.Connection) -> dict[str, Any]:
    rows = conn.execute("SELECT id, data FROM past_cases ORDER BY id").fetchall()
    missing = 0
    mismatch = 0
    invalid = 0
    blank_row_ids = 0
    row_ids: list[str] = []
    for raw_id, raw_data in rows:
        row_id = str(raw_id or "").strip()
        row_ids.append(row_id)
        if not row_id:
            blank_row_ids += 1
        try:
            payload = json.loads(raw_data)
        except (TypeError, json.JSONDecodeError):
            invalid += 1
            continue
        if not isinstance(payload, dict):
            invalid += 1
            continue
        inner_id = str(payload.get("id") or "").strip()
        if not inner_id:
            missing += 1
        elif inner_id != row_id:
            mismatch += 1
    duplicate_ids = len(row_ids) - len(set(row_ids))
    integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    return {
        "rows_total": len(rows),
        "missing_inner_id": missing,
        "mismatching_inner_id": mismatch,
        "invalid_json_objects": invalid,
        "blank_row_ids": blank_row_ids,
        "duplicate_row_ids": duplicate_ids,
        "integrity_check": integrity,
        "related_table_counts": {table: _table_count(conn, table) for table in RELATED_TABLES},
    }


def _create_online_backup(source: sqlite3.Connection, backup_path: Path) -> dict[str, Any]:
    if backup_path.exists():
        raise FileExistsError(f"Backup already exists: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(backup_path) as destination:
        source.backup(destination)
    with sqlite3.connect(f"file:{backup_path.resolve().as_posix()}?mode=ro", uri=True) as backup:
        backup.execute("PRAGMA query_only=ON")
        state = _database_state(backup)
    state["path"] = str(backup_path.resolve())
    state["sha256"] = _file_sha256(backup_path)
    state["size_bytes"] = backup_path.stat().st_size
    return state


def backfill_database(db_path: Path, backup_path: Path, expected_missing: int) -> dict[str, Any]:
    resolved = db_path.resolve()
    started_at = dt.datetime.now(dt.timezone.utc)

    with sqlite3.connect(resolved, timeout=5.0) as conn:
        conn.execute("PRAGMA busy_timeout=5000")
        before = _database_state(conn)
        required = {
            "integrity_check": "ok",
            "missing_inner_id": expected_missing,
            "mismatching_inner_id": 0,
            "invalid_json_objects": 0,
            "blank_row_ids": 0,
            "duplicate_row_ids": 0,
        }
        failed = {key: (before[key], expected) for key, expected in required.items() if before[key] != expected}
        if failed:
            raise RuntimeError(f"Preflight guard failed: {failed}")

        backup = _create_online_backup(conn, backup_path)
        for key in ("rows_total", "missing_inner_id", "mismatching_inner_id", "invalid_json_objects"):
            if backup[key] != before[key]:
                raise RuntimeError(f"Backup verification failed for {key}: {backup[key]} != {before[key]}")
        if backup["integrity_check"] != "ok":
            raise RuntimeError("Backup integrity_check failed")

        conn.execute("BEGIN IMMEDIATE")
        try:
            targets = conn.execute("SELECT id, data FROM past_cases ORDER BY id").fetchall()
            updates: list[tuple[str, str]] = []
            original_hashes: dict[str, str] = {}
            for raw_id, raw_data in targets:
                row_id = str(raw_id or "").strip()
                payload = json.loads(raw_data)
                if str(payload.get("id") or "").strip():
                    continue
                original_hashes[row_id] = _canonical_hash(_without_id(payload))
                payload["id"] = row_id
                updates.append((json.dumps(payload, ensure_ascii=False, separators=(",", ":")), row_id))

            if len(updates) != expected_missing:
                raise RuntimeError(f"Target count changed before write: {len(updates)} != {expected_missing}")

            total_changes_before = conn.total_changes
            conn.executemany("UPDATE past_cases SET data=? WHERE id=?", updates)
            changed = conn.total_changes - total_changes_before

            for row_id, raw_data in conn.execute(
                f"SELECT id, data FROM past_cases WHERE id IN ({','.join('?' for _ in updates)})",
                [row_id for _, row_id in updates],
            ):
                payload = json.loads(raw_data)
                if str(payload.get("id") or "").strip() != row_id:
                    raise RuntimeError(f"ID verification failed for {row_id}")
                if _canonical_hash(_without_id(payload)) != original_hashes[row_id]:
                    raise RuntimeError(f"Non-ID payload changed for {row_id}")

            after_in_transaction = _database_state(conn)
            if after_in_transaction["missing_inner_id"] != 0 or after_in_transaction["mismatching_inner_id"] != 0:
                raise RuntimeError("Post-update ID consistency check failed")
            if after_in_transaction["rows_total"] != before["rows_total"]:
                raise RuntimeError("past_cases row count changed")
            if after_in_transaction["related_table_counts"] != before["related_table_counts"]:
                raise RuntimeError("Related table counts changed")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        after = _database_state(conn)

    return {
        "schema_version": 1,
        "operation": "past_cases_json_id_backfill",
        "started_at": started_at.isoformat(timespec="seconds"),
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "database_path": str(resolved),
        "backup": backup,
        "before": before,
        "after": after,
        "updated_rows": len(updates),
        "verified_updated_rows": changed,
        "only_top_level_id_semantics_changed": True,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--expected-missing", type=int, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        parser.error("This command requires --apply")

    result = backfill_database(args.db, args.backup, args.expected_missing)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
