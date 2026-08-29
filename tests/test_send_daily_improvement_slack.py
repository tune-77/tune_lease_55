from datetime import datetime
import json
import subprocess
import sys

from scripts.send_daily_improvement_slack import (
    build_message,
    _is_plausible_slack_webhook,
    should_skip,
)


def test_system_monitor_section_present_in_message():
    payload = build_message({"applied_count": 0}, report_date="2026-07-14")
    assert "*システム監視*" in payload["text"]


def test_action_ledger_section_missing_report():
    payload = build_message({"applied_count": 0}, report_date="2026-07-14")
    assert "*紫苑の行動ログ*" in payload["text"]
    assert "Agent Action Ledgerレポート未生成" in payload["text"]


def test_action_ledger_section_summarizes_report():
    action_ledger_report = {
        "days": 7,
        "total": 5,
        "pending_approval_count": 2,
        "by_action": {"improvement_classified": 3, "codex_request_drafted": 2},
    }
    payload = build_message(
        {"applied_count": 0},
        report_date="2026-07-14",
        action_ledger_report=action_ledger_report,
    )
    assert "直近7日: `5` 件 / 承認待ち `2` 件" in payload["text"]
    assert "改善候補の分類 3" in payload["text"]
    assert "Codex依頼文の生成 2" in payload["text"]


def test_reflection_journal_section_missing_report():
    payload = build_message({"applied_count": 0}, report_date="2026-07-14")
    assert "*内省モード: 今日の気づき*" in payload["text"]
    assert "レポート未生成" in payload["text"]


def test_reflection_journal_section_summarizes_written_entry():
    reflection_journal_report = {
        "status": "written",
        "theme": "機械学習",
        "angle": "機械学習×システム改善の切り口",
        "problem": "情報が多すぎて整理できていない。",
        "insight": "個人の学習ログが実は一次情報として価値がある。",
        "solution": "週次で学習ログを1本の記事にまとめる。",
    }
    payload = build_message(
        {"applied_count": 0},
        report_date="2026-07-14",
        reflection_journal_report=reflection_journal_report,
    )
    assert "テーマ: 機械学習" in payload["text"]
    assert "切り口: 機械学習×システム改善の切り口" in payload["text"]
    assert "問題: 情報が多すぎて整理できていない。" in payload["text"]
    assert "気づき: 個人の学習ログが実は一次情報として価値がある。" in payload["text"]
    assert "解決: 週次で学習ログを1本の記事にまとめる。" in payload["text"]


def test_reflection_journal_section_reports_skip_reason():
    payload = build_message(
        {"applied_count": 0},
        report_date="2026-07-14",
        reflection_journal_report={"status": "skipped_no_selected_theme"},
    )
    assert "Vaultにテーマ材料なし" in payload["text"]


def test_reflection_journal_section_reports_mana_hold_skip_reason():
    payload = build_message(
        {"applied_count": 0},
        report_date="2026-08-25",
        reflection_journal_report={
            "status": "skipped_mana_hold",
            "mana_action_summary": "Private Reflection を User要求・誤読・次回行動が分かる形で再生成する。",
        },
    )
    assert "Mana判定がhold" in payload["text"]
    assert "Private Reflection を User要求" in payload["text"]


def test_script_dry_run_works_when_executed_by_path(tmp_path):
    # 実運用の reports/latest.json（.gitignore対象で本番実行でのみ生成される）に
    # 依存させず、--report で自己完結した最小レポートを渡す。
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"applied_count": 0}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/send_daily_improvement_slack.py",
            "--report",
            str(report_path),
            "--date",
            "2026-07-25",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "*日次改善レポート*" in result.stdout
    assert "*システム監視*" in result.stdout


def test_invalid_placeholder_webhook_is_skipped_without_send():
    assert not _is_plausible_slack_webhook("https://hooks.slack.com/services/T1234567/B1234567890/xxx")
    assert _is_plausible_slack_webhook(
        "https://hooks.slack.com/services/T1234567/B1234567890/abcdefghijklmnopqrstuvwxyz"
    )


def test_system_monitor_flags_stale_report_and_backlog(tmp_path, monkeypatch):
    import scripts.send_daily_improvement_slack as mod

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "recursive_self_improvement_latest.md").write_text(
        "# Recursive Self-Improvement Report\n\n- Generated at: `2020-01-01T04:00:00`\n",
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    backlog = [
        {"id": str(i), "status": "pending", "promised_at": "2999-01-01T00:00:00"}
        for i in range(25)
    ]
    import json as _json

    (tmp_path / "data" / "shion_pending_tasks.json").write_text(
        _json.dumps(backlog), encoding="utf-8"
    )

    lines = mod._system_monitor_lines(now=datetime(2026, 7, 23, 12, 0, 0))
    text = "\n".join(lines)
    assert "自己改善レポートが" in text and "更新なし" in text
    assert "未完了調査タスクが `25` 件滞留" in text
    assert "異常なし" not in text


def test_system_monitor_flags_consecutive_failure_abort(tmp_path, monkeypatch):
    import scripts.send_daily_improvement_slack as mod

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "codex_queue_result_20260805.json").write_text(
        json.dumps({"guards": {"aborted_by_consecutive_failures": True, "carried_over": ["REV-001", "REV-002"]}}),
        encoding="utf-8",
    )

    lines = mod._system_monitor_lines(now=datetime(2026, 8, 6, 9, 0, 0))
    text = "\n".join(lines)
    assert "連続失敗のため停止しました" in text
    assert "持ち越し 2 件" in text


def test_system_monitor_ignores_non_aborted_queue_result(tmp_path, monkeypatch):
    import scripts.send_daily_improvement_slack as mod

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "codex_queue_result_20260805.json").write_text(
        json.dumps({"guards": {"aborted_by_consecutive_failures": False}}),
        encoding="utf-8",
    )

    lines = mod._system_monitor_lines(now=datetime(2026, 8, 6, 9, 0, 0))
    assert lines == ["• 異常なし"]


def test_system_monitor_healthy(tmp_path, monkeypatch):
    import scripts.send_daily_improvement_slack as mod

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    # ファイル無し（レポートもタスクも無い）→ 異常なし
    assert mod._system_monitor_lines(now=datetime(2026, 7, 23, 12, 0, 0)) == ["• 異常なし"]


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
        "action_summary": "Private Reflection を User要求・誤読・次回行動が分かる形で再生成する。",
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
    assert "next: Private Reflection" in payload["text"]
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


def test_build_message_includes_judgment_asset_field_review_summary():
    report = {
        "applied_count": 0,
        "needs_review_count": 0,
        "failed_count": 0,
        "needs_review": [],
    }
    field_review = {
        "summary": {
            "active_rules": 8,
            "grow": 1,
            "review": 2,
            "sleeping": 4,
            "hold": 1,
        },
        "buckets": {
            "review": [{"statement": "Slackに出さない棚卸し詳細"}],
        },
    }

    payload = build_message(
        report,
        report_date="2026-07-25",
        judgment_asset_field_review=field_review,
    )

    assert "field_review: grow=`1` / review=`2` / sleeping=`4` / hold=`1`" in payload["text"]
    assert "reports/judgment_asset_field_review_latest.md" in payload["text"]
    assert "Slackに出さない棚卸し詳細" not in payload["text"]


def test_should_skip_same_date_and_hash_unless_forced():
    state = {"last_sent_date": "2026-07-14", "last_report_hash": "abc", "digest_version": 2}

    assert should_skip(state, report_date="2026-07-14", digest="abc", force=False)
    assert not should_skip(state, report_date="2026-07-14", digest="abc", force=True)
    assert not should_skip(state, report_date="2026-07-15", digest="abc", force=False)
    assert not should_skip(state, report_date="2026-07-14", digest="def", force=False)


def test_should_skip_treats_legacy_state_without_digest_version_as_already_sent():
    # digest_version 導入前(旧: 生JSON全体からのハッシュ)に書かれた state。
    # 新旧のハッシュ方式は比較不能なため、内容の一致有無に関わらず
    # 同じ日付なら「送信済み」として扱い、移行日の重複送信を防ぐ。
    legacy_state = {"last_sent_date": "2026-07-14", "last_report_hash": "old-scheme-hash"}

    assert should_skip(legacy_state, report_date="2026-07-14", digest="new-scheme-hash", force=False)
    assert not should_skip(legacy_state, report_date="2026-07-14", digest="new-scheme-hash", force=True)
    assert not should_skip(legacy_state, report_date="2026-07-15", digest="new-scheme-hash", force=False)
