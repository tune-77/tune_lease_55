from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import lease_intelligence_tools
from retraining_pipeline import check_retraining_needed
from screening_record_lifecycle import active_screening_predicate
from scripts.export_screening_records import export


def _create_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE screening_records (
                id INTEGER PRIMARY KEY, case_id TEXT NOT NULL, screened_at TEXT NOT NULL,
                total_score REAL NOT NULL, asset_score REAL NOT NULL, tenant_score REAL,
                q_risk_score REAL, competitor_pressure_score REAL, outcome TEXT,
                input_snapshot TEXT, source TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, record_state TEXT NOT NULL DEFAULT 'active',
                parent_deleted_at TEXT, deletion_event_id TEXT
            );
            CREATE TABLE screening_outcomes (
                id INTEGER PRIMARY KEY, delinquent INTEGER, actual_status TEXT
            );
            INSERT INTO screening_outcomes VALUES (1, 1, 'default');
            """
        )
        conn.execute(
            "INSERT INTO screening_records VALUES (1,'active-case','2026-08-31T00:00:00Z',60,40,NULL,NULL,NULL,'contracted',?,'test','x','x','active',NULL,NULL)",
            ('{"company_name":"Active Company"}',),
        )
        conn.execute(
            "INSERT INTO screening_records VALUES (2,'orphan-case','2026-08-31T01:00:00Z',99,90,NULL,NULL,NULL,'contracted',?,'streamlit','x','x','orphan_history',NULL,NULL)",
            ('{"company_name":"Orphan Company"}',),
        )


def test_active_predicate_supports_legacy_schema(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE screening_records (id INTEGER PRIMARY KEY)")
        assert active_screening_predicate(conn) == "1=1"


def test_search_activity_export_and_retraining_ignore_orphan_history(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "cases.db"
    out = tmp_path / "active.csv"
    _create_db(db)
    monkeypatch.setattr(lease_intelligence_tools, "DB_PATH", str(db))

    assert lease_intelligence_tools.search_cases("Active")["count"] == 1
    assert lease_intelligence_tools.search_cases("Orphan")["count"] == 0
    assert lease_intelligence_tools.get_screening_activity("all")["count"] == 1

    assert export(str(db), str(out)) == 1
    with out.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["case_id"] for row in rows] == ["active-case"]

    retraining = check_retraining_needed(
        min_records=2,
        min_delinquent=1,
        db_path=str(db),
    )
    assert retraining["record_count"] == 1
    assert retraining["needed"] is False
