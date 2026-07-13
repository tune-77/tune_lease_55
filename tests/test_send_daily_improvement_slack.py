from scripts.send_daily_improvement_slack import build_message, should_skip


def test_build_message_summarizes_improvement_report():
    report = {
        "applied_count": 1,
        "needs_review_count": 2,
        "failed_count": 0,
        "commit_result": {"message": "コミット対象なし"},
        "needs_review": [
            {
                "id": "REV-001",
                "title": "金額表示を百万単位に統一する",
                "auto_fix_policy": {"risk": "medium"},
            },
            {"id": "REV-002", "title": "内省差分のUser確認依頼を改善する"},
        ],
    }

    payload = build_message(report, report_date="2026-07-14")

    assert "*日次改善レポート*" in payload["text"]
    assert "applied: `1`" in payload["text"]
    assert "needs_review: `2`" in payload["text"]
    assert "REV-001" in payload["text"]
    assert "金額表示を百万単位に統一する" in payload["text"]
    assert "改善状態は変更していません" in payload["text"]


def test_should_skip_same_date_and_hash_unless_forced():
    state = {"last_sent_date": "2026-07-14", "last_report_hash": "abc"}

    assert should_skip(state, report_date="2026-07-14", digest="abc", force=False)
    assert not should_skip(state, report_date="2026-07-14", digest="abc", force=True)
    assert not should_skip(state, report_date="2026-07-15", digest="abc", force=False)
    assert not should_skip(state, report_date="2026-07-14", digest="def", force=False)
