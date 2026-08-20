import json
import sqlite3

import pytest

from api.routers import feedback_loop


@pytest.fixture
def sqlite_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lease_data.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    feedback_loop._ensure_shion_screening_reviews_table()
    return db_path


@pytest.fixture
def redirected_logs(tmp_path, monkeypatch):
    usage_log = tmp_path / "judgment_asset_usage_feedback.jsonl"
    drops_log = tmp_path / "judgment_asset_feedback_drops.jsonl"
    monkeypatch.setattr(feedback_loop, "_JUDGMENT_ASSET_USAGE_FEEDBACK_LOG", usage_log)
    monkeypatch.setattr(feedback_loop, "_JUDGMENT_ASSET_FEEDBACK_DROPS_LOG", drops_log)
    return usage_log, drops_log


def _insert_review(db_path, *, case_id="case-1", result_snapshot=None):
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO shion_screening_reviews (case_id, review_text, result_snapshot) VALUES (?, ?, ?)",
            (case_id, "dummy review text", json.dumps(result_snapshot or {}, ensure_ascii=False)),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_feedback_without_knowledge_refs_is_dropped_and_logged(sqlite_db, redirected_logs):
    usage_log, drops_log = redirected_logs
    review_id = _insert_review(sqlite_db, result_snapshot={"knowledge_refs": []})

    feedback_loop._record_judgment_asset_feedback_from_review(review_id, "useful")

    assert _read_jsonl(usage_log) == []
    drops = _read_jsonl(drops_log)
    assert len(drops) == 1
    assert drops[0]["reason"] == "no_matching_refs"
    assert drops[0]["review_id"] == review_id
    assert drops[0]["case_id"] == "case-1"


def test_feedback_with_knowledge_refs_is_recorded_without_drop(sqlite_db, redirected_logs):
    usage_log, drops_log = redirected_logs
    review_id = _insert_review(sqlite_db, result_snapshot={"knowledge_refs": ["rule-abc"]})

    feedback_loop._record_judgment_asset_feedback_from_review(review_id, "useful")

    entries = _read_jsonl(usage_log)
    assert len(entries) == 1
    assert entries[0]["rule_id"] == "rule-abc"
    assert entries[0]["outcome"] == "helped"
    assert _read_jsonl(drops_log) == []


def test_missing_review_is_dropped_and_logged(sqlite_db, redirected_logs):
    _usage_log, drops_log = redirected_logs

    feedback_loop._record_judgment_asset_feedback_from_review(999999, "useful")

    drops = _read_jsonl(drops_log)
    assert len(drops) == 1
    assert drops[0]["reason"] == "review_not_found"
    assert drops[0]["review_id"] == 999999


def test_unmapped_feedback_value_is_dropped_and_logged(redirected_logs):
    _usage_log, drops_log = redirected_logs

    feedback_loop._record_judgment_asset_feedback_from_review(1, "not_a_real_option")

    drops = _read_jsonl(drops_log)
    assert len(drops) == 1
    assert drops[0]["reason"] == "unknown_outcome"
