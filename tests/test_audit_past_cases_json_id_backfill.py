from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.audit_past_cases_json_id_backfill import audit_database


def _create_db(path: Path, rows: list[tuple[str, str, dict]]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE past_cases (id TEXT PRIMARY KEY, timestamp TEXT, data TEXT)")
        conn.executemany(
            "INSERT INTO past_cases (id, timestamp, data) VALUES (?, ?, ?)",
            [(case_id, timestamp, json.dumps(payload, ensure_ascii=False)) for case_id, timestamp, payload in rows],
        )


def test_dry_run_simulates_only_missing_id_without_writing(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    _create_db(
        db,
        [
            ("case-1", "2026-08-01", {"id": "case-1", "score": 80}),
            ("case-2", "2026-08-02", {"score": 65, "nested": {"id": "keep-me"}}),
        ],
    )
    before = db.read_bytes()

    result = audit_database(db)

    assert result["summary"]["rows_total"] == 2
    assert result["summary"]["missing_inner_id"] == 1
    assert result["summary"]["simulated_rows"] == 1
    assert result["summary"]["only_id_changed"] == 1
    assert result["summary"]["projected_missing_inner_id"] == 0
    assert result["summary"]["ready_for_backfill"] is True
    assert result["database"]["changed_during_audit"] is False
    assert result["database"]["sha256_before"] == result["database"]["sha256_after"]
    assert db.read_bytes() == before


def test_dry_run_holds_when_nonblank_id_mismatches(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    _create_db(db, [("case-1", "2026-08-01", {"id": "different", "score": 80})])

    result = audit_database(db)

    assert result["summary"]["mismatching_inner_id"] == 1
    assert result["summary"]["ready_for_backfill"] is False


def test_dry_run_accepts_database_that_is_already_consistent(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    _create_db(db, [("case-1", "2026-08-01", {"id": "case-1", "score": 80})])

    result = audit_database(db)

    assert result["summary"]["missing_inner_id"] == 0
    assert result["summary"]["already_consistent"] is True
    assert result["summary"]["ready_for_backfill"] is False
