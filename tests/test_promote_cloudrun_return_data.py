from __future__ import annotations

import sqlite3

from scripts.promote_cloudrun_return_data import promote_approved_return_data


def _seed_return_db(path):
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE shion_screening_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT,
                company_name TEXT,
                industry_major TEXT,
                industry_sub TEXT,
                sales_dept TEXT,
                score REAL,
                hantei TEXT,
                q_risk REAL,
                umap_anomaly_score REAL,
                memory_refs INTEGER,
                knowledge_refs INTEGER,
                identity_used INTEGER,
                review_text TEXT NOT NULL,
                prompt_text TEXT,
                form_snapshot TEXT,
                result_snapshot TEXT,
                user_feedback TEXT,
                created_at TEXT,
                cloud_review_id TEXT,
                cloud_event_id TEXT,
                return_review_status TEXT,
                return_review_note TEXT,
                return_reviewed_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO shion_screening_reviews (
                case_id, company_name, industry_major, industry_sub, sales_dept,
                score, hantei, q_risk, umap_anomaly_score, memory_refs,
                knowledge_refs, identity_used, review_text, prompt_text,
                form_snapshot, result_snapshot, user_feedback, created_at,
                cloud_review_id, cloud_event_id, return_review_status, return_review_note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "case-1",
                "[REDACTED]",
                "D 建設業",
                "06 総合工事業",
                "東京",
                72.5,
                "条件付き承認",
                41.0,
                9.5,
                2,
                1,
                1,
                "銀行支援と物件保全を確認する。",
                "",
                '{"asset_name":"建機"}',
                '{"score":72.5}',
                "useful",
                "2026-07-01T00:00:00Z",
                "101",
                "evt-review-1",
                "approved",
                "本体DB昇格候補として承認",
            ),
        )
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
                return_review_note TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO cloudrun_score_inputs (
                event_id, event_type, case_id, surface, score, hantei, industry_major,
                industry_sub, inputs_json, result_json, created_at, return_review_status, return_review_note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "evt-score-1",
                "score_full_calculated",
                "case-1",
                "screening",
                72.5,
                "承認",
                "D 建設業",
                "06 総合工事業",
                '{"nenshu":200}',
                '{"score":72.5}',
                "2026-07-01T00:01:00Z",
                "approved",
                "入力はログだけ昇格",
            ),
        )


def test_promote_cloudrun_return_data_dry_run_does_not_write(tmp_path) -> None:
    return_db = tmp_path / "return.db"
    main_db = tmp_path / "main.db"
    _seed_return_db(return_db)

    result = promote_approved_return_data(
        return_db=return_db,
        main_db=main_db,
        backup_dir=tmp_path / "backups",
        apply=False,
        backup=False,
    )

    assert result["summary"] == {
        "shion_review:would_insert": 1,
        "score_input:would_log_only": 1,
    }
    with sqlite3.connect(main_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM shion_screening_reviews").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM cloudrun_return_promotions").fetchone()[0] == 0


def test_promote_cloudrun_return_data_apply_writes_review_and_log(tmp_path) -> None:
    return_db = tmp_path / "return.db"
    main_db = tmp_path / "main.db"
    _seed_return_db(return_db)
    sqlite3.connect(main_db).close()

    result = promote_approved_return_data(
        return_db=return_db,
        main_db=main_db,
        backup_dir=tmp_path / "backups",
        apply=True,
        backup=True,
    )

    assert result["backup_path"]
    assert result["summary"] == {
        "shion_review:inserted": 1,
        "score_input:logged_only": 1,
    }
    with sqlite3.connect(main_db) as conn:
        conn.row_factory = sqlite3.Row
        review = conn.execute("SELECT * FROM shion_screening_reviews").fetchone()
        assert review["hantei"] == "条件付き承認"
        assert "銀行支援" in review["review_text"]
        logs = conn.execute("SELECT * FROM cloudrun_return_promotions ORDER BY id").fetchall()
        assert len(logs) == 2
        assert {row["source_kind"] for row in logs} == {"shion_review", "score_input"}
    with sqlite3.connect(return_db) as conn:
        conn.row_factory = sqlite3.Row
        review = conn.execute("SELECT return_promoted_at, return_promotion_id FROM shion_screening_reviews").fetchone()
        score = conn.execute("SELECT return_promoted_at, return_promotion_id FROM cloudrun_score_inputs").fetchone()
        assert review["return_promoted_at"]
        assert review["return_promotion_id"]
        assert score["return_promoted_at"]
        assert score["return_promotion_id"]


def test_promote_cloudrun_return_data_apply_is_idempotent(tmp_path) -> None:
    return_db = tmp_path / "return.db"
    main_db = tmp_path / "main.db"
    _seed_return_db(return_db)
    sqlite3.connect(main_db).close()

    first = promote_approved_return_data(
        return_db=return_db,
        main_db=main_db,
        backup_dir=tmp_path / "backups",
        apply=True,
        backup=False,
    )
    second = promote_approved_return_data(
        return_db=return_db,
        main_db=main_db,
        backup_dir=tmp_path / "backups",
        apply=True,
        backup=False,
    )

    assert first["summary"] == {
        "shion_review:inserted": 1,
        "score_input:logged_only": 1,
    }
    assert second["summary"] == {}
    with sqlite3.connect(main_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM shion_screening_reviews").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM cloudrun_return_promotions").fetchone()[0] == 2
