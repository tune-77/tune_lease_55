"""Phase 1（紫苑中心・改善ループ移行計画）: トリアージの構造化記録。

- POST /api/improvement/triage は追記形式で記録し、最後のエントリが有効
- 同定は canonical_key + source_event_id + タイトルスナップショットで冗長に持つ
- 対話文脈（P1-3）は記録が無ければ空。未確定は持ち越しのみ（P1-4）
"""

import json

import pytest
from fastapi import HTTPException


@pytest.fixture()
def main_module(tmp_path, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_REPO_ROOT", str(tmp_path))
    # 実行環境の実台帳（~/Library/Logs/tunelease/ledger.jsonl）に依存しないよう固定
    monkeypatch.setattr(main, "_latest_improvement_statuses", lambda: {})
    return main


def test_record_and_get_last_entry_wins(tmp_path, main_module):
    main = main_module
    first = main.record_improvement_triage(
        main.ImprovementTriageRequest(
            canonical_key="misc_abc123",
            decision="later",
            title="対話室の表示改善",
            item_id="REV-220",
            source_event_id="event-1",
            rule_decision="today",
            classified_by="user",
        )
    )
    assert first["ok"] is True
    second = main.record_improvement_triage(
        main.ImprovementTriageRequest(
            canonical_key="misc_abc123",
            decision="today",
            title="対話室の表示改善",
            item_id="REV-220",
            source_event_id="event-1",
            rule_decision="today",
            classified_by="user",
        )
    )
    assert second["record"]["decision"] == "today"

    result = main.get_improvement_triage()
    assert len(result["records"]) == 1
    assert result["records"][0]["decision"] == "today"
    assert result["records"][0]["source_event_id"] == "event-1"
    assert result["records"][0]["title"] == "対話室の表示改善"
    # Codexキュー（idキー）との突き合わせ用にレポートIDも冗長記録される
    assert result["records"][0]["item_id"] == "REV-220"
    assert result["counts"] == {"today": 1}

    # 追記形式でファイルに2行残っている（監査可能）
    raw = (tmp_path / "data" / "shion_improvement_triage.jsonl").read_text(encoding="utf-8")
    assert len([line for line in raw.splitlines() if line.strip()]) == 2


def test_invalid_decision_and_empty_key_rejected(main_module):
    main = main_module
    with pytest.raises(HTTPException) as exc:
        main.record_improvement_triage(
            main.ImprovementTriageRequest(canonical_key="misc_x", decision="someday")
        )
    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        main.record_improvement_triage(
            main.ImprovementTriageRequest(canonical_key="  ", decision="today")
        )
    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        main.record_improvement_triage(
            main.ImprovementTriageRequest(canonical_key="misc_x", decision="today", classified_by="oracle")
        )
    assert exc.value.status_code == 422


def test_triage_context_reflects_records(main_module):
    main = main_module
    main.record_improvement_triage(
        main.ImprovementTriageRequest(
            canonical_key="misc_ctx1",
            decision="today",
            title="表示ラベルの見直し",
            rule_decision="today",
            classified_by="user",
        )
    )
    main.record_improvement_triage(
        main.ImprovementTriageRequest(
            canonical_key="misc_ctx2",
            decision="discard",
            title="古い重複候補",
            rule_decision="discard",
            classified_by="rule",
        )
    )

    context = main._build_dialogue_triage_context()

    assert "改善トリアージ状況" in context
    assert "今日やる1件" in context
    # rule 分類の記録は「提案」であり User 確定の件数には数えない
    assert "捨てる0件" in context
    assert "表示ラベルの見直し" in context
    assert "判断主体: rule" in context
    # P1-4: 未確定は持ち越しのみ・自動昇格なし
    assert "自動昇格・自動破棄はしない" in context


def test_triage_context_separates_llm_proposals(main_module):
    main = main_module
    main.record_improvement_triage(
        main.ImprovementTriageRequest(canonical_key="misc_u1", decision="today", title="確定済み", classified_by="user")
    )
    main.record_improvement_triage(
        main.ImprovementTriageRequest(canonical_key="misc_l1", decision="discard", title="LLM提案", classified_by="llm")
    )

    context = main._build_dialogue_triage_context()

    assert "User確定済みの判断: 今日やる1件" in context
    assert "捨てる0件" in context  # llm 提案は確定件数に入らない
    assert "紫苑（LLM）の未確定提案が1件ある" in context
    assert "キュー・自動承認へ影響しない" in context


def test_triage_context_empty_without_records(main_module):
    assert main_module._build_dialogue_triage_context() == ""


def test_triage_context_excludes_ledger_resolved(main_module, monkeypatch):
    """改善ログ（台帳）との整合性: 台帳で解決済みの候補は持ち越しから外す。

    例: トリアージで「今日やる」とした候補が、既存の改善ログページで
    実装済み(applied)や削除(deleted)になった場合、古い判断を語り続けない。
    """
    main = main_module
    main.record_improvement_triage(
        main.ImprovementTriageRequest(
            canonical_key="misc_done", decision="today", title="もう実装済みの候補"
        )
    )
    main.record_improvement_triage(
        main.ImprovementTriageRequest(
            canonical_key="misc_alive", decision="later", title="まだ生きている候補"
        )
    )
    monkeypatch.setattr(main, "_latest_improvement_statuses", lambda: {"misc_done": "applied"})

    context = main._build_dialogue_triage_context()

    assert "もう実装済みの候補" not in context
    assert "まだ生きている候補" in context
    assert "解決済み（applied/deleted/rejected）になった判断 1 件は持ち越しから除外" in context
    assert "今日やる0件" in context
    assert "後回し1件" in context


def test_approve_requires_today_decision(main_module):
    """P2-3: 実装承認は「今日やる」確定済みの候補のみ。承認で approved_at が付く。"""
    main = main_module
    main.record_improvement_triage(
        main.ImprovementTriageRequest(canonical_key="misc_ap1", decision="today", title="承認対象")
    )
    main.record_improvement_triage(
        main.ImprovementTriageRequest(canonical_key="misc_ap2", decision="later", title="後回し候補")
    )

    result = main.approve_improvement_triage(
        main.ImprovementTriageApproveRequest(canonical_key="misc_ap1")
    )
    assert result["ok"] is True
    assert result["record"]["approved_at"]
    # 承認時に紫苑依頼文の下書きが添付される（生成のみ・実行しない）
    assert result["record"]["codex_request_draft"].startswith("紫苑依頼文:")
    assert "承認対象" in result["record"]["codex_request_draft"]

    current = main.get_improvement_triage()
    by_key = {r["canonical_key"]: r for r in current["records"]}
    assert by_key["misc_ap1"]["approved_at"]
    assert "approved_at" not in by_key["misc_ap2"] or not by_key["misc_ap2"].get("approved_at")

    with pytest.raises(HTTPException) as exc:
        main.approve_improvement_triage(main.ImprovementTriageApproveRequest(canonical_key="misc_ap2"))
    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        main.approve_improvement_triage(main.ImprovementTriageApproveRequest(canonical_key="misc_none"))
    assert exc.value.status_code == 404


def test_triage_context_marks_approved(main_module):
    main = main_module
    main.record_improvement_triage(
        main.ImprovementTriageRequest(canonical_key="misc_apx", decision="today", title="承認済み候補")
    )
    main.approve_improvement_triage(main.ImprovementTriageApproveRequest(canonical_key="misc_apx"))

    context = main._build_dialogue_triage_context()

    assert "今日やる・実装承認済み" in context
    assert "紫苑依頼文は「今日やる・実装承認済み」の候補についてのみ作成する" in context


def test_monitor_report_recorded(tmp_path, main_module):
    """監視先行率KPI: 紫苑の監視報告表示時刻が記録される。"""
    main = main_module
    result = main.record_monitor_report(main.MonitorReportRequest(failed_count=2))

    assert result["ok"] is True
    raw = (tmp_path / "data" / "shion_monitor_report_log.jsonl").read_text(encoding="utf-8")
    row = json.loads(raw.strip().splitlines()[-1])
    assert row["failed_count"] == 2
    assert row["source"] == "dialogue_daily_report"
    assert row["ts"]


def test_triage_file_corruption_tolerated(tmp_path, main_module):
    main = main_module
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "shion_improvement_triage.jsonl").write_text(
        "壊れた行\n" + json.dumps({"canonical_key": "misc_ok", "decision": "later", "decided_at": "2026-07-17T10:00:00"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = main.get_improvement_triage()

    assert len(result["records"]) == 1
    assert result["records"][0]["canonical_key"] == "misc_ok"
