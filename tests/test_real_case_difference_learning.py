import json
import sqlite3

from scripts.learn_from_case_differences import (
    _anonymous_case_id,
    build_difference_report,
)


def _database():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE screening_records (
            id INTEGER PRIMARY KEY,
            case_id TEXT,
            screened_at TEXT,
            total_score REAL,
            outcome TEXT
        );
        CREATE TABLE past_cases (
            id TEXT PRIMARY KEY,
            industry_sub TEXT,
            data TEXT
        );
        CREATE TABLE judgment_feedback (
            id INTEGER PRIMARY KEY,
            case_id TEXT,
            recorded_at TEXT,
            source TEXT,
            model_decision TEXT,
            human_decision TEXT,
            changed INTEGER,
            reason TEXT,
            score REAL,
            input_snapshot TEXT,
            evidence_snapshot TEXT,
            review_status TEXT
        );
        """
    )
    return conn


def test_report_separates_approved_feedback_from_outcome_signals():
    conn = _database()
    conn.execute(
        """
        INSERT INTO judgment_feedback VALUES
        (1, 'secret-case', '2026-06-13', 'screen', '承認', '条件付', 1,
         '返済原資の確認を追加', 72, ?, '{}', 'approved')
        """,
        (json.dumps({"grade": "4-6", "company_name": "秘密会社"}),),
    )
    conn.execute(
        "INSERT INTO screening_records VALUES (1, 'lost-case', '2026-06-13', 85, 'lost')"
    )
    conn.execute(
        "INSERT INTO past_cases VALUES ('lost-case', '建設業', ?)",
        (
            json.dumps(
                {
                    "company_name": "非公開会社",
                    "inputs": {"grade": "4-6", "company_name": "非公開会社"},
                }
            ),
        ),
    )

    report = build_difference_report(conn, per_signal_limit=5)

    assert report["status"] == "training_ready"
    assert report["summary"]["approved_training_candidates"] == 1
    assert report["summary"]["outcome_signal_counts"]["high_score_lost"] == 1
    assert report["outcome_review_queue"][0]["eligible_for_training"] is False
    serialized = json.dumps(report, ensure_ascii=False)
    assert "秘密会社" not in serialized
    assert "非公開会社" not in serialized
    assert "lost-case" not in serialized


def test_report_requires_human_feedback_when_no_explicit_differences():
    conn = _database()
    conn.execute(
        "INSERT INTO screening_records VALUES (1, 'won-case', '2026-06-13', 40, 'contracted')"
    )

    report = build_difference_report(conn, per_signal_limit=5)

    assert report["status"] == "needs_human_feedback"
    assert report["summary"]["approved_training_candidates"] == 0
    assert report["summary"]["outcome_signal_counts"]["low_score_contracted"] == 1


def test_anonymous_case_id_is_stable_and_does_not_expose_source():
    first = _anonymous_case_id("customer-case-123")
    second = _anonymous_case_id("customer-case-123")

    assert first == second
    assert "customer-case-123" not in first
