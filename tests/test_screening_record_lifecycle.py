from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from screening_recorder import record_screening_result
from screening_record_lifecycle import mark_screening_parent_deleted


CASE_ID = "20260830123456789012_abcd1234"


def _create_parent_tables(db: Path) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE past_cases (id TEXT PRIMARY KEY, data TEXT)")
        conn.execute("CREATE TABLE excluded_grade_cases (id TEXT PRIMARY KEY, data TEXT)")


def _record(db: Path, case_id: str = CASE_ID):
    return record_screening_result(
        case_id=case_id,
        screened_at="2026-08-30T12:34:57Z",
        total_score=80,
        asset_score=60,
        source="streamlit",
        db_path=str(db),
    )


def test_canonical_case_id_requires_existing_parent(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    _create_parent_tables(db)

    result = _record(db)

    assert result["success"] is False
    assert result["error"] == "parent case not found for canonical case_id"
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM screening_records").fetchone()[0] == 0


def test_existing_parent_allows_record_and_delete_state_is_preserved(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    _create_parent_tables(db)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO past_cases (id, data) VALUES (?, ?)", (CASE_ID, json.dumps({"id": CASE_ID})))

    result = _record(db)
    assert result["success"] is True

    with sqlite3.connect(db) as conn:
        changed, event_id = mark_screening_parent_deleted(conn, [CASE_ID])
        conn.execute("DELETE FROM past_cases WHERE id=?", (CASE_ID,))
        row = conn.execute(
            "SELECT record_state, parent_deleted_at, deletion_event_id FROM screening_records WHERE case_id=?",
            (CASE_ID,),
        ).fetchone()

    assert changed == 1
    assert row[0] == "parent_deleted"
    assert row[1]
    assert row[2] == event_id


def test_data_cases_delete_marks_screening_record_first(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "cases.db"
    _create_parent_tables(db)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO past_cases (id, data) VALUES (?, ?)", (CASE_ID, json.dumps({"id": CASE_ID})))
    assert _record(db)["success"] is True

    import data_cases

    monkeypatch.setattr(data_cases, "DB_PATH", str(db))
    monkeypatch.setattr(data_cases, "refresh_stats_caches", lambda: None)

    assert data_cases.delete_case(CASE_ID) is True
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM past_cases WHERE id=?", (CASE_ID,)).fetchone()[0] == 0
        state = conn.execute(
            "SELECT record_state, parent_deleted_at, deletion_event_id FROM screening_records WHERE case_id=?",
            (CASE_ID,),
        ).fetchone()
        audit = conn.execute(
            "SELECT route, reason, requested_count, matched_count, deleted_count, "
            "affected_screening_count, status FROM case_deletion_events"
        ).fetchone()
        audit_item = conn.execute(
            "SELECT case_id, item_status FROM case_deletion_event_items"
        ).fetchone()
    assert state[0] == "parent_deleted"
    assert state[1]
    assert state[2].startswith("del_")
    assert audit == ("data_cases.delete_case", "manual_delete", 1, 1, 1, 1, "completed")
    assert audit_item == (CASE_ID, "deleted")
