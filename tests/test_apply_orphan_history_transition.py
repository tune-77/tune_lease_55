from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.apply_orphan_history_transition import apply_transition


CASE_ID = "20260830123456789012_abcd1234"


def _create_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE past_cases (id TEXT PRIMARY KEY);
            CREATE TABLE excluded_grade_cases (id TEXT PRIMARY KEY);
            CREATE TABLE screening_records (
                id INTEGER PRIMARY KEY, case_id TEXT NOT NULL, screened_at TEXT NOT NULL,
                total_score REAL NOT NULL, asset_score REAL NOT NULL, tenant_score REAL,
                q_risk_score REAL, competitor_pressure_score REAL, outcome TEXT,
                input_snapshot TEXT, source TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, record_state TEXT NOT NULL DEFAULT 'active',
                parent_deleted_at TEXT, deletion_event_id TEXT
            );
            CREATE TABLE screening_outcomes (id INTEGER PRIMARY KEY, screening_id INTEGER);
            """
        )
        conn.execute(
            "INSERT INTO screening_records VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (CASE_ID, "2026-08-30T12:00:00Z", 80, 60, None, None, None, None, None, "streamlit", "x", "x", "active", None, None),
        )


def test_apply_transitions_only_state_and_preserves_backup(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    backup = tmp_path / "backup.db"
    _create_db(db)

    result = apply_transition(db, backup, expected_target=1)

    assert result["status"] == "PASS"
    assert result["updated_rows"] == 1
    assert result["non_state_fields_changed"] == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT record_state FROM screening_records").fetchone()[0] == "orphan_history"
    with sqlite3.connect(backup) as conn:
        assert conn.execute("SELECT record_state FROM screening_records").fetchone()[0] == "active"


def test_apply_refuses_unapproved_target_count(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    backup = tmp_path / "backup.db"
    _create_db(db)

    with pytest.raises(RuntimeError, match="Dry-run guard failed"):
        apply_transition(db, backup, expected_target=98)

    assert not backup.exists()
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT record_state FROM screening_records").fetchone()[0] == "active"
