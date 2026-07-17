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
        "key-deleted": "deleted",
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
            {"id": "REV-006", "title": "削除対象", "canonical_key": "key-deleted"},
        ],
    }

    result = main._normalize_improvement_report(report)
    by_id = {item["id"]: item for item in result["items"]}

    assert by_id["REV-001"]["status"] == "APPROVED"
    assert by_id["REV-002"]["status"] == "REJECTED"
    assert by_id["REV-003"]["status"] == "PARKED"
    assert by_id["REV-004"]["status"] == "RULE_REGISTERED"
    assert by_id["REV-005"]["status"] == "RULE_REVIEW"
    assert "REV-006" not in by_id
    assert result["approved"] == 1
    assert result["rejected"] == 1
    assert result["parked"] == 1


def test_cloudrun_improvement_items_include_raw_preview(monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_historical_applied_improvements", lambda: (set(), set()))
    monkeypatch.setattr(
        main,
        "_read_recent_cloudrun_input_events_from_gcs",
        lambda days=45: [
            {
                "event_id": "event-123456789",
                "event_type": "improvement_note",
                "surface": "chat_improvement",
                "ts": "2026-07-15T00:00:00Z",
                "payload": {
                    "title": "AIチャット改善候補",
                    "body": "## AI整理\n- 課題: 回答が途中で切れている\n- 改善案: 回答生成の完了判定を見直す",
                },
            }
        ],
    )

    items = main._cloudrun_improvement_items_from_gcs()

    assert items[0]["source_event_id"] == "event-123456789"
    assert "回答が途中で切れている" in items[0]["raw_preview"]
    assert items[0]["detail"].startswith("## AI整理")


def test_cloudrun_improvement_items_fall_back_to_local_log(tmp_path, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_historical_applied_improvements", lambda: (set(), set()))
    monkeypatch.setattr(main, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(main, "_read_recent_cloudrun_input_events_from_gcs", lambda days=45: [])
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "cloudrun_improvement_log.jsonl").write_text(
        json.dumps(
            {
                "event_id": "local-1",
                "ts": "2026-07-15T01:00:00Z",
                "title": "Cloud Run改善メモ",
                "body": "ローカル同期済みの改善本文",
                "surface": "chat_improvement",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    items = main._cloudrun_improvement_items_from_gcs()

    assert items[0]["source_event_id"] == "local-1"
    assert items[0]["raw_preview"] == "ローカル同期済みの改善本文"


def test_cloudrun_improvement_items_apply_review_control(monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_historical_applied_improvements", lambda: (set(), set()))
    monkeypatch.setattr(
        main,
        "_read_recent_cloudrun_input_events_from_gcs",
        lambda days=45: [
            {
                "event_id": "event-note-123456",
                "event_type": "improvement_note",
                "surface": "chat_improvement",
                "ts": "2026-07-15T00:00:00Z",
                "payload": {
                    "canonical_key": "cloudrun-key-1",
                    "title": "改善ログ操作",
                    "body": "承認ボタン後も消えない",
                },
            },
            {
                "event_id": "event-review-123456",
                "event_type": "improvement_review",
                "surface": "improvement_log",
                "ts": "2026-07-15T00:01:00Z",
                "payload": {
                    "canonical_key": "cloudrun-key-1",
                    "title": "改善ログ操作",
                    "action": "approved",
                },
            },
        ],
    )

    items = main._cloudrun_improvement_items_from_gcs()

    assert len(items) == 1
    assert items[0]["canonical_key"] == "cloudrun-key-1"
    assert items[0]["status"] == "APPROVED"


def test_cloudrun_improvement_canonical_key_stable_against_readable_title(monkeypatch):
    """表示タイトルの整形ロジックが変わっても canonical_key は従来のタイトル式のまま。

    整形前に記録された削除イベント（旧キー）と照合できなくなると、
    削除済みメモが改善リストへ復活してしまう回帰の防止。
    """
    import api.main as main

    monkeypatch.setattr(main, "_historical_applied_improvements", lambda: (set(), set()))
    body = "課題: スコア理由の表示が分かりにくい\n次の行動: 表示ラベルを見直す"
    # 整形導入前のキー: payloadタイトルそのまま（generic）で計算されたもの
    legacy_key = main._improvement_canonical_key("チャット改善メモ", body)

    monkeypatch.setattr(
        main,
        "_read_recent_cloudrun_input_events_from_gcs",
        lambda days=45: [
            {
                "event_id": "event-note-legacy",
                "event_type": "improvement_note",
                "surface": "chat_improvement",
                "ts": "2026-07-15T00:00:00Z",
                "payload": {"title": "チャット改善メモ", "body": body},
            },
            {
                "event_id": "event-delete-legacy",
                "event_type": "improvement_delete",
                "surface": "improvement_log",
                "ts": "2026-07-16T00:00:00Z",
                "payload": {"canonical_key": legacy_key, "status": "deleted"},
            },
        ],
    )

    assert main._cloudrun_improvement_items_from_gcs() == []


def test_cloudrun_improvement_title_falls_back_to_body_excerpt(monkeypatch):
    """タイトルなし・マーカーなしのメモは総称ではなく本文抜粋をタイトルにする。"""
    import api.main as main

    monkeypatch.setattr(main, "_historical_applied_improvements", lambda: (set(), set()))
    monkeypatch.setattr(
        main,
        "_read_recent_cloudrun_input_events_from_gcs",
        lambda days=45: [
            {
                "event_id": "event-note-plain",
                "event_type": "improvement_note",
                "surface": "next_chat_rag",
                "ts": "2026-07-15T00:00:00Z",
                "payload": {"original_text": "審査画面の入力欄が多すぎて迷う"},
            }
        ],
    )

    items = main._cloudrun_improvement_items_from_gcs()

    assert items[0]["title"] == "審査画面の入力欄が多すぎて迷う"


def test_cloudrun_improvement_items_skip_deleted_control(monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_historical_applied_improvements", lambda: (set(), set()))
    monkeypatch.setattr(
        main,
        "_read_recent_cloudrun_input_events_from_gcs",
        lambda days=45: [
            {
                "event_id": "event-note-abcdef",
                "event_type": "improvement_note",
                "surface": "chat_improvement",
                "ts": "2026-07-15T00:00:00Z",
                "payload": {
                    "canonical_key": "cloudrun-key-delete",
                    "title": "削除対象の改善",
                    "body": "一覧から消したい",
                },
            },
            {
                "event_id": "event-delete-abcdef",
                "event_type": "improvement_delete",
                "surface": "improvement_log",
                "ts": "2026-07-15T00:01:00Z",
                "payload": {
                    "canonical_key": "cloudrun-key-delete",
                    "title": "削除対象の改善",
                    "status": "deleted",
                },
            },
        ],
    )

    assert main._cloudrun_improvement_items_from_gcs() == []


def test_cloudrun_improvement_items_skip_regular_lease_chat_noise(monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_historical_applied_improvements", lambda: (set(), set()))
    monkeypatch.setattr(
        main,
        "_read_recent_cloudrun_input_events_from_gcs",
        lambda days=45: [
            {
                "event_id": "event-lease-chat",
                "event_type": "improvement_note",
                "surface": "chat_improvement",
                "ts": "2026-07-17T00:00:00Z",
                "payload": {
                    "title": "トラック２０台リースして欲しい",
                    "body": "**ユーザー要望**\nトラック２０台リースして欲しい\n\n**めぶき返答**\n審査論点を整理します。",
                },
            },
            {
                "event_id": "event-case-memo",
                "event_type": "improvement_note",
                "surface": "chat_improvement",
                "ts": "2026-07-17T00:01:00Z",
                "payload": {
                    "title": "運輸業・郵便業 リース案件審査メモ",
                    "body": "## 抽出された改善候補\n- AIスコアとQ_riskの乖離に関する審査ロジックの改善",
                },
            },
            {
                "event_id": "event-business-question",
                "event_type": "improvement_note",
                "surface": "chat_improvement",
                "ts": "2026-07-17T00:02:00Z",
                "payload": {
                    "title": "条件付き承認を営業担当へ説明するときの追加資料と承認条件は？",
                    "body": "条件付き承認を営業担当へ説明するときの追加資料と承認条件は？",
                },
            },
            {
                "event_id": "event-real-improvement",
                "event_type": "improvement_note",
                "surface": "chat_improvement",
                "ts": "2026-07-17T00:03:00Z",
                "payload": {
                    "title": "チャット改善メモ",
                    "body": "課題: 本物の改善だけ残す。\n次の行動: フィルタを検証する。",
                },
            },
        ],
    )

    items = main._cloudrun_improvement_items_from_gcs()

    assert len(items) == 1
    assert items[0]["source_event_id"] == "event-real-improvement"
