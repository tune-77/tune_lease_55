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


def _write_active_rules(path, rules):
    path.write_text(json.dumps({"rules": rules}, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def active_rules_file(tmp_path, monkeypatch):
    canonical_path = tmp_path / "canonical_judgment_rules.json"
    _write_active_rules(
        canonical_path,
        [
            {
                "id": "b259411afb954d6d",
                "status": "active",
                "concept": "business_plan_specificity",
                "canonical_statement": "受注根拠と返済原資を確認する。",
            },
        ],
    )
    monkeypatch.setattr(feedback_loop, "_CANONICAL_JUDGMENT_RULES_JSON", canonical_path)
    return canonical_path


def _insert_review(db_path, *, case_id="case-1", review_text="dummy review text"):
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO shion_screening_reviews (case_id, review_text) VALUES (?, ?)",
            (case_id, review_text),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_feedback_without_citation_is_dropped_and_logged(sqlite_db, redirected_logs, active_rules_file):
    usage_log, drops_log = redirected_logs
    review_id = _insert_review(sqlite_db, review_text="出典なしの通常レビュー本文です。")

    feedback_loop._record_judgment_asset_feedback_from_review(review_id, "useful")

    assert _read_jsonl(usage_log) == []
    drops = _read_jsonl(drops_log)
    assert len(drops) == 1
    assert drops[0]["reason"] == "no_matching_refs"
    assert drops[0]["review_id"] == review_id
    assert drops[0]["case_id"] == "case-1"


def test_feedback_with_citation_is_recorded_without_drop(sqlite_db, redirected_logs, active_rules_file):
    usage_log, drops_log = redirected_logs
    review_id = _insert_review(
        sqlite_db,
        review_text="4. 稟議で残すべき一文\n判断資産出典: 正規 JA-cr-b2594 / business_plan_specificity",
    )

    feedback_loop._record_judgment_asset_feedback_from_review(review_id, "useful")

    entries = _read_jsonl(usage_log)
    assert len(entries) == 1
    assert entries[0]["rule_id"] == "b259411afb954d6d"
    assert entries[0]["outcome"] == "helped"
    assert entries[0]["review_id"] == review_id
    assert _read_jsonl(drops_log) == []


def test_ambiguous_citation_prefix_is_dropped_not_misattributed(sqlite_db, redirected_logs, tmp_path, monkeypatch):
    canonical_path = tmp_path / "canonical_judgment_rules.json"
    _write_active_rules(
        canonical_path,
        [
            {
                "id": "b259411afb954d6d",
                "status": "active",
                "concept": "business_plan_specificity",
                "canonical_statement": "受注根拠と返済原資を確認する。",
            },
            {
                "id": "b259400000000000",
                "status": "active",
                "concept": "ambiguous_prefix_twin",
                "canonical_statement": "prefix衝突テスト用。",
            },
        ],
    )
    monkeypatch.setattr(feedback_loop, "_CANONICAL_JUDGMENT_RULES_JSON", canonical_path)

    usage_log, drops_log = redirected_logs
    review_id = _insert_review(
        sqlite_db,
        review_text="判断資産出典: 正規 JA-cr-b2594",
    )
    # b259411afb954d6d と b259400000000000 の両方が "b2594" prefix に一致するため
    # 一意に解決できず、誤った rule_id への誤帰属を避けて drop する。
    feedback_loop._record_judgment_asset_feedback_from_review(review_id, "useful")

    assert _read_jsonl(usage_log) == []
    drops = _read_jsonl(drops_log)
    assert len(drops) == 1
    assert drops[0]["reason"] == "no_matching_refs"


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


def test_drop_is_also_sent_through_cloudrun_writeback(redirected_logs, monkeypatch):
    sent = []
    monkeypatch.setattr(
        feedback_loop,
        "record_cloudrun_input_event",
        lambda **kwargs: sent.append(kwargs),
    )

    feedback_loop._record_judgment_asset_feedback_from_review(1, "not_a_real_option")

    assert len(sent) == 1
    assert sent[0]["event_type"] == "judgment_asset_feedback_drop"
    assert sent[0]["payload"]["reason"] == "unknown_outcome"


def test_usage_log_write_failure_is_logged_as_drop(sqlite_db, redirected_logs, active_rules_file, monkeypatch):
    usage_log, drops_log = redirected_logs
    review_id = _insert_review(
        sqlite_db,
        review_text="判断資産出典: 正規 JA-cr-b2594 / business_plan_specificity",
    )

    class _BoomParent:
        def mkdir(self, *_args, **_kwargs):
            pass

    class _BoomPath:
        parent = _BoomParent()

        def open(self, *_args, **_kwargs):
            raise OSError("disk full")

    monkeypatch.setattr(feedback_loop, "_JUDGMENT_ASSET_USAGE_FEEDBACK_LOG", _BoomPath())

    feedback_loop._record_judgment_asset_feedback_from_review(review_id, "useful")

    drops = _read_jsonl(drops_log)
    assert len(drops) == 1
    assert drops[0]["reason"] == "usage_log_write_error"
    assert not usage_log.exists()
