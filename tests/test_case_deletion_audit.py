from __future__ import annotations

import sqlite3

import pytest

from case_deletion_audit import (
    begin_case_deletion_event,
    complete_case_deletion_event,
    list_case_deletion_events,
)


def _create_tables(conn: sqlite3.Connection) -> None:
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


def test_deletion_audit_tracks_parent_and_child_in_one_event() -> None:
    conn = sqlite3.connect(":memory:")
    _create_tables(conn)
    conn.execute("INSERT INTO past_cases VALUES ('case-1', '{}')")
    conn.execute("INSERT INTO screening_records (case_id) VALUES ('case-1')")

    event = begin_case_deletion_event(
        conn,
        ["case-1"],
        route="test.route",
        reason="test_reason",
        metadata={"source": "pytest"},
    )
    conn.execute("DELETE FROM past_cases WHERE id='case-1'")
    status = complete_case_deletion_event(conn, event, ["case-1"])
    conn.commit()

    header = conn.execute(
        "SELECT route, reason, requested_count, matched_count, deleted_count, "
        "affected_screening_count, status, metadata_json FROM case_deletion_events"
    ).fetchone()
    item = conn.execute(
        "SELECT case_id, item_status FROM case_deletion_event_items"
    ).fetchone()
    child = conn.execute(
        "SELECT record_state, deletion_event_id FROM screening_records"
    ).fetchone()

    assert status == "completed"
    assert header == ("test.route", "test_reason", 1, 1, 1, 1, "completed", '{"source": "pytest"}')
    assert item == ("case-1", "deleted")
    assert child == ("parent_deleted", event.event_id)


def test_no_match_attempt_is_audited() -> None:
    conn = sqlite3.connect(":memory:")
    _create_tables(conn)

    event = begin_case_deletion_event(
        conn,
        ["missing-case"],
        route="test.route",
        reason="not_found_test",
    )
    status = complete_case_deletion_event(conn, event, [])

    assert status == "no_match"
    assert conn.execute(
        "SELECT requested_count, matched_count, deleted_count, status FROM case_deletion_events"
    ).fetchone() == (1, 0, 0, "no_match")
    assert conn.execute(
        "SELECT case_id, item_status FROM case_deletion_event_items"
    ).fetchone() == ("missing-case", "not_found")


def test_transaction_rollback_removes_audit_and_restores_child() -> None:
    conn = sqlite3.connect(":memory:")
    _create_tables(conn)
    conn.commit()
    conn.execute("INSERT INTO past_cases VALUES ('case-1', '{}')")
    conn.execute("INSERT INTO screening_records (case_id) VALUES ('case-1')")
    conn.commit()

    with pytest.raises(RuntimeError):
        try:
            begin_case_deletion_event(
                conn,
                ["case-1"],
                route="test.route",
                reason="rollback_test",
            )
            conn.execute("DELETE FROM past_cases WHERE id='case-1'")
            raise RuntimeError("simulated failure")
        except Exception:
            conn.rollback()
            raise

    assert conn.execute("SELECT COUNT(*) FROM past_cases").fetchone()[0] == 1
    assert conn.execute("SELECT record_state FROM screening_records").fetchone()[0] == "active"
    assert conn.execute("SELECT COUNT(*) FROM case_deletion_events").fetchone()[0] == 0


def test_read_only_listing_returns_event_items_and_clamps_limit() -> None:
    conn = sqlite3.connect(":memory:")
    _create_tables(conn)
    conn.execute("INSERT INTO past_cases VALUES ('case-1', '{}')")
    event = begin_case_deletion_event(
        conn,
        ["case-1", "missing-case"],
        route="api.case_delete",
        reason="api_request",
    )
    conn.execute("DELETE FROM past_cases WHERE id='case-1'")
    complete_case_deletion_event(conn, event, ["case-1"])
    conn.commit()

    result = list_case_deletion_events(conn, limit=999, offset=-5)

    assert result["total"] == 1
    assert result["limit"] == 200
    assert result["offset"] == 0
    assert result["filters"] == {"status": "", "date_from": "", "date_to": ""}
    assert result["events"][0]["event_id"] == event.event_id
    assert result["events"][0]["status"] == "completed"
    assert result["events"][0]["items"] == [
        {"case_id": "case-1", "parent_table": "past_cases", "status": "deleted"},
        {"case_id": "missing-case", "parent_table": "past_cases", "status": "not_found"},
    ]


def test_read_only_listing_filters_by_status_and_date() -> None:
    conn = sqlite3.connect(":memory:")
    _create_tables(conn)
    event = begin_case_deletion_event(
        conn,
        ["missing-case"],
        route="api.case_delete",
        reason="api_request",
    )
    complete_case_deletion_event(conn, event, [])
    conn.execute(
        "UPDATE case_deletion_events SET occurred_at='2026-08-31 12:30:00' WHERE event_id=?",
        (event.event_id,),
    )

    matched = list_case_deletion_events(
        conn,
        status="no_match",
        date_from="2026-08-31",
        date_to="2026-08-31",
    )
    excluded = list_case_deletion_events(conn, status="completed")

    assert matched["total"] == 1
    assert matched["filters"] == {
        "status": "no_match",
        "date_from": "2026-08-31",
        "date_to": "2026-08-31",
    }
    assert excluded["total"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"status": "deleted"}, "invalid deletion audit status"),
        ({"date_from": "2026/08/31"}, "Invalid isoformat string"),
        ({"date_from": "2026-09-01", "date_to": "2026-08-31"}, "date_from must not be after date_to"),
    ],
)
def test_read_only_listing_rejects_invalid_filters(kwargs: dict, message: str) -> None:
    conn = sqlite3.connect(":memory:")
    _create_tables(conn)
    from case_deletion_audit import ensure_case_deletion_audit_tables

    ensure_case_deletion_audit_tables(conn)
    with pytest.raises(ValueError, match=message):
        list_case_deletion_events(conn, **kwargs)
