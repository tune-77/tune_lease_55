from __future__ import annotations


def test_queue_only_treats_policy_allowed_items_as_safe():
    from scripts.build_codex_auto_queue import build_queue

    report = {
        "date": "2026-06-19",
        "needs_review": [
            {
                "id": "REV-001",
                "title": "案件ネットワーク画面の表示不具合修正",
                "reason": "auto_fix_policy: 対象ファイル未特定のため手動確認",
                "auto_fix_policy": {
                    "auto_fix_allowed": False,
                    "reason": "対象ファイル未特定のため手動確認",
                    "risk": "medium",
                    "max_files": 1,
                },
            },
            {
                "id": "REV-002",
                "title": "ボタン文言を修正",
                "target_module": "frontend/src/app/page.tsx",
                "auto_fix_policy": {
                    "auto_fix_allowed": True,
                    "reason": "小規模・低リスク変更",
                    "risk": "low",
                    "max_files": 1,
                    "required_checks": ["py_compile", "targeted_test"],
                },
            },
        ],
    }

    queue = build_queue(report, limit=3)

    assert queue["codex_auto_safe_count"] == 1
    assert queue["codex_auto_maybe_count"] == 1
    assert queue["queued_count"] == 1
    assert queue["items"][0]["id"] == "REV-002"


def test_queue_blocks_policy_allowed_but_non_low_risk_items():
    from scripts.build_codex_auto_queue import build_queue

    report = {
        "needs_review": [
            {
                "id": "REV-003",
                "title": "表示文言を修正",
                "auto_fix_policy": {
                    "auto_fix_allowed": True,
                    "risk": "medium",
                    "max_files": 1,
                },
            }
        ],
    }

    queue = build_queue(report, limit=3)

    assert queue["codex_auto_safe_count"] == 0
    assert queue["codex_auto_maybe_count"] == 1
    assert queue["queued_count"] == 0
