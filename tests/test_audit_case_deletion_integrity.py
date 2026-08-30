from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from case_deletion_audit import begin_case_deletion_event, complete_case_deletion_event
from scripts.audit_case_deletion_integrity import audit_deletion_integrity, render_markdown


def _create_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE past_cases (id TEXT PRIMARY KEY, data TEXT)")
    conn.execute("CREATE TABLE excluded_grade_cases (id TEXT PRIMARY KEY, data TEXT)")
    conn.execute(
        """
        CREATE TABLE screening_records (
            id INTEGER PRIMARY KEY,
            case_id TEXT NOT NULL,
            record_state TEXT NOT NULL DEFAULT 'active',
            parent_deleted_at TEXT,
            deletion_event_id TEXT,
            updated_at TEXT NOT NULL DEFAULT ''
        )
        """
    )
    return conn


def test_integrity_audit_passes_for_completed_event(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    conn = _create_db(db)
    conn.execute("INSERT INTO past_cases VALUES ('case-1', '{}')")
    conn.execute("INSERT INTO screening_records (case_id) VALUES ('case-1')")
    event = begin_case_deletion_event(conn, ["case-1"], route="test", reason="test")
    conn.execute("DELETE FROM past_cases WHERE id='case-1'")
    complete_case_deletion_event(conn, event, ["case-1"])
    conn.commit()
    conn.close()

    result = audit_deletion_integrity(db)

    assert result["status"] == "PASS"
    assert result["summary"]["events"] == 1
    assert result["issues"] == []
    assert "読み取り専用監査" in render_markdown(result)


def test_integrity_audit_detects_count_and_orphan_link_mismatches(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    conn = _create_db(db)
    conn.execute("INSERT INTO past_cases VALUES ('case-1', '{}')")
    event = begin_case_deletion_event(conn, ["case-1"], route="test", reason="test")
    conn.execute("DELETE FROM past_cases WHERE id='case-1'")
    complete_case_deletion_event(conn, event, ["case-1"])
    conn.execute("UPDATE case_deletion_events SET deleted_count=9 WHERE event_id=?", (event.event_id,))
    conn.execute(
        "INSERT INTO screening_records (case_id, record_state, deletion_event_id) VALUES ('orphan', 'parent_deleted', 'del_missing')"
    )
    conn.commit()
    conn.close()

    result = audit_deletion_integrity(db)
    codes = {issue["code"] for issue in result["issues"]}

    assert result["status"] == "FAIL"
    assert "deleted_count_mismatch" in codes
    assert "invalid_completed_event" in codes
    assert "screening_event_without_audit" in codes


def test_integrity_audit_detects_stale_started_event(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    conn = _create_db(db)
    event = begin_case_deletion_event(conn, [], route="test", reason="test")
    conn.execute(
        "UPDATE case_deletion_events SET occurred_at='2026-08-31 00:00:00' WHERE event_id=?",
        (event.event_id,),
    )
    conn.commit()
    conn.close()

    result = audit_deletion_integrity(
        db,
        stale_started_minutes=10,
        now=datetime(2026, 8, 31, 1, 0, tzinfo=timezone.utc),
    )

    assert result["status"] == "FAIL"
    assert {issue["code"] for issue in result["issues"]} == {"stale_started_event"}


def test_daily_pipeline_runs_deletion_integrity_audit_in_strict_mode() -> None:
    pipeline = (Path(__file__).resolve().parents[1] / "scripts" / "run_daily_improvement_core.sh").read_text(
        encoding="utf-8"
    )

    assert 'scripts/audit_case_deletion_integrity.py" --strict' in pipeline
    assert 'log_step "audit_case_deletion_integrity"' in pipeline
    assert 'scripts/notify_case_deletion_integrity_slack.py" --apply' in pipeline
    assert 'log_step "notify_case_deletion_integrity_slack"' in pipeline
    assert "読み取り専用・自動修復なし" in pipeline
