import json
import sqlite3

from scripts import build_judgment_asset_response_manifest as manifest


def _active_rules():
    return {
        "b259411afb954d6d": {"id": "b259411afb954d6d", "concept": "business_plan_specificity"},
        "cf61a9701fc8cc42": {"id": "cf61a9701fc8cc42", "concept": "asset_life_and_residual"},
    }


def test_second_review_of_same_case_is_not_falsely_marked_recorded():
    # 同じcase_idに2件レビューがあり、同じルールを引用している。1件目は正常に記録され、
    # 2件目は書き込み失敗（drop）。case_id単位でしか見なければ2件目も「記録済み」に
    # 誤判定されてしまう。review_id単位で見て正しく区別できることを確認する。
    reviews = [
        {
            "id": 10,
            "case_id": "case-multi",
            "review_text": "判断資産出典: 正規 JA-cr-b2594",
            "user_feedback": "useful",
            "created_at": "2026-08-20T00:00:00",
        },
        {
            "id": 11,
            "case_id": "case-multi",
            "review_text": "判断資産出典: 正規 JA-cr-b2594",
            "user_feedback": "useful",
            "created_at": "2026-08-20T00:05:00",
        },
    ]
    usage_rows = [{"rule_id": "b259411afb954d6d", "case_id": "case-multi", "review_id": 10}]
    drop_rows = [{"review_id": 11, "reason": "usage_log_write_error"}]

    payload = manifest.build_manifest(
        target_date="2026-08-20",
        reviews=reviews,
        active_rules=_active_rules(),
        usage_feedback_rows=usage_rows,
        drop_rows=drop_rows,
    )

    entries_by_id = {entry["review_id"]: entry for entry in payload["entries"]}
    assert entries_by_id[10]["status"] == "recorded"
    assert entries_by_id[11]["status"] == "citation_not_recorded"
    assert entries_by_id[11]["drop_reasons"] == ["usage_log_write_error"]


def test_cloud_run_review_id_is_translated_via_cloud_review_id():
    # Cloud Run上ではreview_id=77で記録されたdropが、ローカル同期後は
    # ローカルid(=1)とは別にcloud_review_id="77"として残る。dropログの
    # review_idはCloud Run側の値(77)のままなので、ローカルidではなく
    # cloud_review_id経由で突き合わせないと理由が見えなくなる。
    reviews = [
        {
            "id": 1,
            "cloud_review_id": "77",
            "case_id": "case-cloud",
            "review_text": "判断資産出典: 正規 JA-cr-b2594",
            "user_feedback": "useful",
            "created_at": "2026-08-20T00:00:00",
        }
    ]
    drop_rows = [{"review_id": 77, "reason": "no_matching_refs"}]

    payload = manifest.build_manifest(
        target_date="2026-08-20",
        reviews=reviews,
        active_rules=_active_rules(),
        usage_feedback_rows=[],
        drop_rows=drop_rows,
    )

    entry = payload["entries"][0]
    assert entry["drop_reasons"] == ["no_matching_refs"]


def test_recorded_review_is_status_recorded():
    reviews = [
        {
            "id": 1,
            "case_id": "case-1",
            "review_text": "判断資産出典: 正規 JA-cr-b2594 / business_plan_specificity",
            "user_feedback": "useful",
            "created_at": "2026-08-20T00:00:00",
        }
    ]
    usage_rows = [{"rule_id": "b259411afb954d6d", "case_id": "case-1", "outcome": "helped"}]

    payload = manifest.build_manifest(
        target_date="2026-08-20",
        reviews=reviews,
        active_rules=_active_rules(),
        usage_feedback_rows=usage_rows,
        drop_rows=[],
    )

    assert payload["summary"]["recorded"] == 1
    assert payload["summary"]["citation_not_recorded"] == 0
    entry = payload["entries"][0]
    assert entry["status"] == "recorded"
    assert entry["cited_assets"] == [{"rule_id": "b259411afb954d6d", "concept": "business_plan_specificity"}]
    assert entry["unrecorded_rule_ids"] == []


def test_cited_but_unrecorded_review_is_flagged_with_drop_reason():
    reviews = [
        {
            "id": 2,
            "case_id": "case-2",
            "review_text": "判断資産出典: 正規 JA-cr-cf61a / asset_life_and_residual",
            "user_feedback": "useful",
            "created_at": "2026-08-20T00:01:00",
        }
    ]
    drop_rows = [{"review_id": 2, "reason": "usage_log_write_error"}]

    payload = manifest.build_manifest(
        target_date="2026-08-20",
        reviews=reviews,
        active_rules=_active_rules(),
        usage_feedback_rows=[],
        drop_rows=drop_rows,
    )

    assert payload["summary"]["citation_not_recorded"] == 1
    entry = payload["entries"][0]
    assert entry["status"] == "citation_not_recorded"
    assert entry["unrecorded_rule_ids"] == ["cf61a9701fc8cc42"]
    assert entry["drop_reasons"] == ["usage_log_write_error"]

    markdown = manifest.build_markdown(payload)
    assert "review#2 case=case-2" in markdown
    assert "usage_log_write_error" in markdown


def test_feedback_without_any_citation_is_flagged():
    reviews = [
        {
            "id": 3,
            "case_id": "case-3",
            "review_text": "出典のない通常レビューです。",
            "user_feedback": "needs_fix",
            "created_at": "2026-08-20T00:02:00",
        }
    ]

    payload = manifest.build_manifest(
        target_date="2026-08-20",
        reviews=reviews,
        active_rules=_active_rules(),
        usage_feedback_rows=[],
        drop_rows=[],
    )

    assert payload["summary"]["feedback_without_citation"] == 1
    assert payload["entries"][0]["status"] == "feedback_without_citation"


def test_review_without_feedback_yet_is_not_double_counted():
    reviews = [
        {
            "id": 4,
            "case_id": "case-4",
            "review_text": "判断資産出典: 正規 JA-cr-b2594",
            "user_feedback": "",
            "created_at": "2026-08-20T00:03:00",
        }
    ]

    payload = manifest.build_manifest(
        target_date="2026-08-20",
        reviews=reviews,
        active_rules=_active_rules(),
        usage_feedback_rows=[],
        drop_rows=[],
    )

    assert payload["summary"]["no_feedback_yet"] == 1
    assert payload["summary"]["citation_not_recorded"] == 0
    assert payload["entries"][0]["status"] == "no_feedback_yet"


def test_load_reviews_reads_from_sqlite(tmp_path):
    db_path = tmp_path / "lease_data.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE shion_screening_reviews (
            id INTEGER PRIMARY KEY,
            case_id TEXT,
            review_text TEXT,
            user_feedback TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO shion_screening_reviews (case_id, review_text, user_feedback, created_at) VALUES (?, ?, ?, ?)",
        ("case-1", "判断資産出典: 正規 JA-cr-b2594", "useful", "2026-08-20T00:00:00"),
    )
    conn.commit()
    conn.close()

    rows = manifest.load_reviews(db_path, limit=10)
    assert len(rows) == 1
    assert rows[0]["case_id"] == "case-1"


def test_load_reviews_returns_empty_when_table_missing(tmp_path):
    db_path = tmp_path / "empty.db"
    sqlite3.connect(str(db_path)).close()
    assert manifest.load_reviews(db_path) == []


def test_main_writes_json_and_markdown(tmp_path):
    db_path = tmp_path / "lease_data.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE shion_screening_reviews (
            id INTEGER PRIMARY KEY,
            case_id TEXT,
            review_text TEXT,
            user_feedback TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO shion_screening_reviews (case_id, review_text, user_feedback, created_at) VALUES (?, ?, ?, ?)",
        ("case-1", "判断資産出典: 正規 JA-cr-b2594", "useful", "2026-08-20T00:00:00"),
    )
    conn.commit()
    conn.close()

    canonical_json = tmp_path / "canonical.json"
    canonical_json.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "id": "b259411afb954d6d",
                        "status": "active",
                        "concept": "business_plan_specificity",
                        "canonical_statement": "x",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    usage_jsonl = tmp_path / "usage.jsonl"
    usage_jsonl.write_text(
        json.dumps({"rule_id": "b259411afb954d6d", "case_id": "case-1"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "manifest.json"
    output_md = tmp_path / "manifest.md"

    payload = manifest.build_manifest(
        target_date="2026-08-20",
        reviews=manifest.load_reviews(db_path),
        active_rules=manifest.load_active_rules(path=canonical_json),
        usage_feedback_rows=manifest._read_jsonl(usage_jsonl),
        drop_rows=[],
    )
    manifest._write_json(output_json, payload)
    output_md.write_text(manifest.build_markdown(payload), encoding="utf-8")

    assert json.loads(output_json.read_text(encoding="utf-8"))["summary"]["recorded"] == 1
    assert "Judgment Asset Response Manifest" in output_md.read_text(encoding="utf-8")
