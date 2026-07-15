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
    assert "*Mana判定*" in payload["text"]
    assert "status: `missing`" in payload["text"]
    assert "*審査用語監査*" in payload["text"]
    assert "審査用語監査レポート未生成" in payload["text"]
    assert "*判断資産 実戦検証*" in payload["text"]
    assert "判断資産成長レポート未生成" in payload["text"]
    assert "改善状態は変更していません" in payload["text"]


def test_build_message_includes_mana_report_summary_without_raw_evidence():
    report = {
        "applied_count": 0,
        "needs_review_count": 0,
        "failed_count": 0,
        "needs_review": [],
    }
    mana_report = {
        "status": "hold",
        "inputs": {"candidate_count": 12, "useful_candidate_count": 7},
        "findings": [
            {
                "level": "hold",
                "code": "private_reflection_not_meaningful",
                "message": "Private Reflectionの意味更新が弱い。",
                "evidence": {"raw": "Slackに出してはいけない長い原文"},
            }
        ],
    }

    payload = build_message(report, report_date="2026-07-14", mana_report=mana_report)

    assert "status: `hold`" in payload["text"]
    assert "candidates: `12`" in payload["text"]
    assert "private_reflection_not_meaningful" in payload["text"]
    assert "Slackに出してはいけない長い原文" not in payload["text"]


def test_build_message_includes_screening_terms_summary_without_raw_findings():
    report = {
        "applied_count": 0,
        "needs_review_count": 0,
        "failed_count": 0,
        "needs_review": [],
    }
    terms_report = {
        "status": "warn",
        "counts": {"warn": 2, "review": 9, "ok": 100},
        "findings": [
            {
                "severity": "warn",
                "text": "生の長い指摘本文",
            }
        ],
    }

    payload = build_message(
        report,
        report_date="2026-07-14",
        screening_terms_report=terms_report,
    )

    assert "*審査用語監査*" in payload["text"]
    assert "status: `warn`" in payload["text"]
    assert "warn: `2`" in payload["text"]
    assert "review: `9`" in payload["text"]
    assert "reports/screening_terms_audit_latest.md" in payload["text"]
    assert "生の長い指摘本文" not in payload["text"]


def test_build_message_includes_judgment_asset_field_validation_summary():
    report = {
        "applied_count": 0,
        "needs_review_count": 0,
        "failed_count": 0,
        "needs_review": [],
    }
    growth_report = {
        "latest": {
            "score": 43.2,
            "components": {"field_validation": 18.0},
            "field_feedback": {
                "totals": {"used": 3, "helped": 1, "challenged": 1, "rejected": 0},
                "unused_active_rules": 5,
                "rules": [{"note": "Slackに出さない詳細"}],
            },
        }
    }

    payload = build_message(
        report,
        report_date="2026-07-15",
        judgment_asset_growth_report=growth_report,
    )

    assert "*判断資産 実戦検証*" in payload["text"]
    assert "score: `43.2`" in payload["text"]
    assert "field_validation: `18.0`" in payload["text"]
    assert "used=`3`" in payload["text"]
    assert "helped=`1`" in payload["text"]
    assert "challenged=`1`" in payload["text"]
    assert "unused_active=`5`" in payload["text"]
    assert "reports/judgment_asset_growth_latest.md" in payload["text"]
    assert "Slackに出さない詳細" not in payload["text"]


def test_should_skip_same_date_and_hash_unless_forced():
    state = {"last_sent_date": "2026-07-14", "last_report_hash": "abc"}

    assert should_skip(state, report_date="2026-07-14", digest="abc", force=False)
    assert not should_skip(state, report_date="2026-07-14", digest="abc", force=True)
    assert not should_skip(state, report_date="2026-07-15", digest="abc", force=False)
    assert not should_skip(state, report_date="2026-07-14", digest="def", force=False)
