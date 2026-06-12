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
