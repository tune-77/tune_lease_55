from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.audit_orphan_history_transition import audit_database


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


def test_dry_run_targets_only_active_parentless_streamlit_rows(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    _create_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO screening_records VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (CASE_ID, "2026-08-30T12:00:00Z", 80, 60, None, None, None, None, None, "streamlit", "x", "x", "active", None, None),
        )
    before = db.read_bytes()

    result = audit_database(db, expected_target=1)

    assert result["summary"]["target_rows"] == 1
    assert result["summary"]["ready_for_apply"] is True
    assert result["summary"]["simulated_changed_columns"] == ["record_state"]
    assert result["metrics_after_active_filter"]["rows"] == 0
    assert db.read_bytes() == before


def test_dry_run_holds_when_expected_count_differs(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    _create_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO screening_records VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (CASE_ID, "2026-08-30T12:00:00Z", 80, 60, None, None, None, None, None, "streamlit", "x", "x", "active", None, None),
        )

    result = audit_database(db, expected_target=98)

    assert result["summary"]["target_count_matches"] is False
    assert result["summary"]["ready_for_apply"] is False


def test_dry_run_accepts_fully_applied_population(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    _create_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO screening_records VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (CASE_ID, "2026-08-30T12:00:00Z", 80, 60, None, None, None, None, None, "streamlit", "x", "x", "orphan_history", None, None),
        )

    result = audit_database(db, expected_target=1)

    assert result["summary"]["target_rows"] == 0
    assert result["summary"]["classified_orphan_history"] == 1
    assert result["summary"]["already_applied"] is True
