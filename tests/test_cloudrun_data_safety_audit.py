from __future__ import annotations

import importlib.util
import sqlite3

from api.cloudrun_data_safety_audit import audit_cloudrun_data_safety
from api.shion_agent_tools import READ_ONLY_DB_TOOLS
from scripts.promote_cloudrun_return_data import promote_approved_return_data
import data_cases


def test_cloudrun_data_safety_audit_is_green():
    result = audit_cloudrun_data_safety()

    assert result["mode"] == "cloudrun_data_safety_audit"
    assert result["status"] == "ok"
    assert result["issue_count"] == 0
    assert "no_db_write" in result["guardrail"]


def test_cloudrun_data_safety_adk_tool_is_registered():
    names = {tool.__name__ for tool in READ_ONLY_DB_TOOLS}

    assert "audit_cloudrun_data_safety" in names


def test_restore_lease_snapshot_skips_existing_db(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = data_dir / "lease_data.db"
    target.write_bytes(b"local-db")

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("LEASE_DB_FILENAME", "lease_data.db")
    monkeypatch.setenv("CLOUDRUN_DATA_MODE", "production")

    spec = importlib.util.spec_from_file_location(
        "restore_lease_db_snapshot",
        "scripts/restore_lease_db_snapshot.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main()

    assert target.read_bytes() == b"local-db"


def test_score_input_skips_when_return_registered_case_id_exists(tmp_path, monkeypatch):
    return_db = tmp_path / "return.db"
    target_db = tmp_path / "lease_data.db"
    sqlite3.connect(target_db).close()
    monkeypatch.setattr(data_cases, "DB_PATH", str(target_db))

    with sqlite3.connect(return_db) as conn:
        conn.execute(
            """
            CREATE TABLE cloudrun_score_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                event_type TEXT,
                case_id TEXT,
                surface TEXT,
                score REAL,
                hantei TEXT,
                industry_major TEXT,
                industry_sub TEXT,
                inputs_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                return_review_status TEXT,
                return_review_note TEXT,
                return_registered_case_id TEXT DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            INSERT INTO cloudrun_score_inputs (
                event_id, event_type, case_id, surface, score, hantei, industry_major,
                industry_sub, inputs_json, result_json, created_at, return_review_status,
                return_review_note, return_registered_case_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "evt-score-registered",
                "score_full_calculated",
                "case-registered",
                "screening",
                72.5,
                "承認",
                "D 建設業",
                "06 総合工事業",
                '{"company_name":"[REDACTED]","nenshu":200}',
                '{"score":72.5}',
                "2026-07-01T00:01:00Z",
                "approved",
                "already registered",
                "case-existing-1",
            ),
        )

    result = promote_approved_return_data(
        return_db=return_db,
        target_db=target_db,
        backup_dir=tmp_path / "backups",
        apply=True,
        backup=False,
    )

    assert result["summary"] == {"score_input:skipped_already_registered": 1}
    with sqlite3.connect(target_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM cloudrun_return_promotions").fetchone()[0] == 1
        assert conn.execute(
            "SELECT target_id FROM cloudrun_return_promotions"
        ).fetchone()[0] == "case-existing-1"
