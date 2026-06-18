from __future__ import annotations

import json


def test_get_improvement_log_includes_recursive_summary(tmp_path, monkeypatch):
    import api.main as main

    reports = tmp_path / "reports"
    reports.mkdir()
    report_path = reports / "latest.json"
    recursive_path = reports / "recursive_self_improvement_latest.json"
    report_path.write_text(
        json.dumps(
                {
                    "date": "2026-06-12",
                    "generated_at": "2026-06-12T12:00:00",
                    "status": "COMPLETED",
                    "approved": 1,
                    "auto_fix_candidates": [
                        {
                            "id": "REV-001",
                            "title": "送信ボタンの文言を「保存」に変更する",
                            "auto_fix_policy": {"reason": "小規模・低リスク変更", "risk": "low"},
                        }
                    ],
                    "needs_review": 0,
                    "parked": 0,
                    "rejected": 0,
                    "applied": 1,
                "items": [],
                "obsidian_compliance": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    recursive_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-12T12:30:00",
                "canonical_candidate_count": 3,
                "ranked_queue_count": 1,
                "suppressed_count": 1,
                "measurement_summary": {
                    "pdca_rate": 50.0,
                    "response_changed_rate": 50.0,
                    "repeat_issue_rate": 33.3,
                    "reuse_rate": 66.7,
                    "noise_rate": 33.3,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "_REPO_ROOT", tmp_path)

    result = main.get_improvement_log()

    assert result["status"] == "COMPLETED"
    assert result["recursive_self_improvement"]["canonical_candidate_count"] == 3
    assert result["recursive_self_improvement"]["measurement_summary"]["noise_rate"] == 33.3


def test_normalize_improvement_report_reflects_review_ledger_statuses(monkeypatch):
    import api.main as main

    statuses = {
        "key-approved": "approved",
        "key-rejected": "rejected",
        "key-deferred": "deferred",
        "key-rule": "rule_registered",
        "key-rule-review": "rule_review",
    }
    monkeypatch.setattr(main, "_latest_improvement_statuses", lambda: statuses)

    report = {
        "date": "2026-06-19",
        "generated_at": "2026-06-19T05:00:00",
        "needs_review": [
            {"id": "REV-001", "title": "承認対象", "canonical_key": "key-approved"},
            {"id": "REV-002", "title": "却下対象", "canonical_key": "key-rejected"},
            {"id": "REV-003", "title": "保留対象", "canonical_key": "key-deferred"},
            {"id": "REV-004", "title": "ルール登録対象", "canonical_key": "key-rule"},
            {"id": "REV-005", "title": "ルール要確認対象", "canonical_key": "key-rule-review"},
        ],
    }

    result = main._normalize_improvement_report(report)
    by_id = {item["id"]: item for item in result["items"]}

    assert by_id["REV-001"]["status"] == "APPROVED"
    assert by_id["REV-002"]["status"] == "REJECTED"
    assert by_id["REV-003"]["status"] == "PARKED"
    assert by_id["REV-004"]["status"] == "RULE_REGISTERED"
    assert by_id["REV-005"]["status"] == "RULE_REVIEW"
    assert result["approved"] == 1
    assert result["rejected"] == 1
    assert result["parked"] == 1
