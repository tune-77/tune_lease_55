from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.backfill_past_cases_json_ids import backfill_database


def _create_db(path: Path, rows: list[tuple[str, dict]]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE past_cases (id TEXT PRIMARY KEY, data TEXT)")
        conn.executemany(
            "INSERT INTO past_cases (id, data) VALUES (?, ?)",
            [(case_id, json.dumps(payload, ensure_ascii=False)) for case_id, payload in rows],
        )


def test_backfill_creates_backup_and_changes_only_missing_top_level_id(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    backup = tmp_path / "backup.db"
    _create_db(db, [("case-1", {"score": 70, "nested": {"id": "keep"}}), ("case-2", {"id": "case-2"})])

    result = backfill_database(db, backup, expected_missing=1)

    assert result["status"] == "PASS"
    assert result["updated_rows"] == 1
    assert result["after"]["missing_inner_id"] == 0
    with sqlite3.connect(db) as conn:
        payload = json.loads(conn.execute("SELECT data FROM past_cases WHERE id='case-1'").fetchone()[0])
    assert payload == {"score": 70, "nested": {"id": "keep"}, "id": "case-1"}
    with sqlite3.connect(backup) as conn:
        original = json.loads(conn.execute("SELECT data FROM past_cases WHERE id='case-1'").fetchone()[0])
    assert original == {"score": 70, "nested": {"id": "keep"}}


def test_backfill_refuses_mismatch_before_backup_or_write(tmp_path: Path) -> None:
    db = tmp_path / "cases.db"
    backup = tmp_path / "backup.db"
    _create_db(db, [("case-1", {"id": "wrong", "score": 70})])
    before = db.read_bytes()

    with pytest.raises(RuntimeError, match="Preflight guard failed"):
        backfill_database(db, backup, expected_missing=0)

    assert db.read_bytes() == before
    assert not backup.exists()
