"""紫苑の未完了調査タスク（shion_pending_tasks）の陳腐化・上限・件数定義のテスト。"""

import json
from datetime import datetime, timedelta

import lease_intelligence_pending as pending


def _write(path, tasks):
    path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")


def test_is_pending_open_definition():
    now = datetime(2026, 7, 23, 12, 0, 0)
    fresh = {"status": "pending", "promised_at": (now - timedelta(days=1)).isoformat()}
    stale = {"status": "pending", "promised_at": (now - timedelta(days=30)).isoformat()}
    no_ts = {"status": "pending"}  # promised_at 欠損は「陳腐化していない」扱い

    assert pending.is_pending_open(fresh, now) is True
    assert pending.is_pending_open(stale, now) is False
    assert pending.is_pending_open(no_ts, now) is True
    assert pending.is_pending_open({"status": "done"}, now) is False
    assert pending.is_pending_open({"status": "expired"}, now) is False


def test_reconcile_expires_stale_pending(tmp_path, monkeypatch):
    now = datetime(2026, 7, 23, 12, 0, 0)
    path = tmp_path / "shion_pending_tasks.json"
    _write(path, [
        {"id": "a", "topic": "古い約束", "status": "pending",
         "promised_at": (now - timedelta(days=20)).isoformat()},
        {"id": "b", "topic": "最近の約束", "status": "pending",
         "promised_at": (now - timedelta(days=2)).isoformat()},
        {"id": "c", "topic": "完了済み", "status": "done"},
    ])
    monkeypatch.setattr(pending, "PENDING_PATH", str(path))

    reconciled = pending.reconcile_pending(now=now)

    by_id = {t["id"]: t for t in reconciled}
    assert by_id["a"]["status"] == "expired"
    assert "expired_at" in by_id["a"]
    assert by_id["b"]["status"] == "pending"
    assert by_id["c"]["status"] == "done"

    # 永続化されていること
    saved = {t["id"]: t for t in json.loads(path.read_text(encoding="utf-8"))}
    assert saved["a"]["status"] == "expired"

    # get_pending_tasks は陳腐化後の pending のみ返す
    assert [t["id"] for t in pending.get_pending_tasks()] == ["b"]


def test_reconcile_caps_history(tmp_path, monkeypatch):
    now = datetime(2026, 7, 23, 12, 0, 0)
    path = tmp_path / "shion_pending_tasks.json"
    done = [
        {"id": f"d{i}", "topic": f"完了{i}", "status": "done",
         "done_at": (now - timedelta(days=i)).isoformat()}
        for i in range(pending.MAX_HISTORY + 50)
    ]
    done.append({"id": "live", "topic": "生きてる約束", "status": "pending",
                 "promised_at": now.isoformat()})
    _write(path, done)
    monkeypatch.setattr(pending, "PENDING_PATH", str(path))

    reconciled = pending.reconcile_pending(now=now)

    history = [t for t in reconciled if t.get("status") != "pending"]
    assert len(history) == pending.MAX_HISTORY
    # pending は必ず残る
    assert any(t["id"] == "live" for t in reconciled)
    # 最新の done が残り、最古が捨てられる
    kept_ids = {t["id"] for t in history}
    assert "d0" in kept_ids
    assert f"d{pending.MAX_HISTORY + 49}" not in kept_ids


def test_promise_detection_matches_future_commitments(tmp_path, monkeypatch):
    path = tmp_path / "shion_pending_tasks.json"
    monkeypatch.setattr(pending, "PENDING_PATH", str(path))

    future_replies = [
        "それでは調べてみますね。",
        "こちらで確認します。",
        "詳細を調査いたします。",
        "後で報告します。",
        "結果を改めてご報告します。",
        "検索してみます。",
    ]
    for reply in future_replies:
        ids = pending.extract_and_save_promises("元の質問", reply)
        assert ids, f"未来の約束が検出されなかった: {reply}"


def test_promise_detection_ignores_past_and_requests(tmp_path, monkeypatch):
    """過去形（完了）・ユーザーへの依頼は約束として記録しない（過剰マッチ抑制）。"""
    path = tmp_path / "shion_pending_tasks.json"
    monkeypatch.setattr(pending, "PENDING_PATH", str(path))

    non_promises = [
        "ご確認ください。",              # ユーザーへの依頼
        "確認してください。",            # 依頼
        "さきほど確認しました。",        # 過去（完了）
        "調べてみました。",              # 過去
        "確認できませんでした。",        # 過去否定
        "ご確認いただけますようお願いします。",  # 依頼
        "これは重要な確認事項です。",    # 名詞
    ]
    for reply in non_promises:
        ids = pending.extract_and_save_promises("元の質問", reply)
        assert ids == [], f"約束でない文が記録された: {reply}"
    # ファイルには何も書かれていない
    assert not path.exists() or json.loads(path.read_text(encoding="utf-8")) == []


def test_promise_routes_topic_to_improvement_log(tmp_path, monkeypatch):
    """約束を検出したら topic を改善ログ取り込み口へ起票し、追跡対象にする。"""
    pending_path = tmp_path / "shion_pending_tasks.json"
    intake_path = tmp_path / "chat_quick_fix_intake.jsonl"
    monkeypatch.setattr(pending, "PENDING_PATH", str(pending_path))
    monkeypatch.setattr(pending, "CHAT_INTAKE_PATH", str(intake_path))

    pending.extract_and_save_promises("残価テーブルの根拠を教えて", "承知しました、調べてみます。")

    lines = [l for l in intake_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["title"] == "残価テーブルの根拠を教えて"
    assert rec["source"] == "shion_promise"
    assert rec["target_module"] == ""  # quick_fix にせず needs_review に留める


def test_promise_dispatch_dedupes_same_topic(tmp_path, monkeypatch):
    pending_path = tmp_path / "shion_pending_tasks.json"
    intake_path = tmp_path / "chat_quick_fix_intake.jsonl"
    monkeypatch.setattr(pending, "PENDING_PATH", str(pending_path))
    monkeypatch.setattr(pending, "CHAT_INTAKE_PATH", str(intake_path))

    pending.extract_and_save_promises("同じ質問", "確認します。")
    pending.extract_and_save_promises("同じ質問", "調査いたします。")

    lines = [l for l in intake_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1  # 同一 topic は id で重複排除


def test_non_promise_does_not_route(tmp_path, monkeypatch):
    pending_path = tmp_path / "shion_pending_tasks.json"
    intake_path = tmp_path / "chat_quick_fix_intake.jsonl"
    monkeypatch.setattr(pending, "PENDING_PATH", str(pending_path))
    monkeypatch.setattr(pending, "CHAT_INTAKE_PATH", str(intake_path))

    pending.extract_and_save_promises("これは？", "ご確認ください。")  # 依頼＝約束でない

    assert not intake_path.exists()


def test_reconcile_noop_when_clean(tmp_path, monkeypatch):
    now = datetime(2026, 7, 23, 12, 0, 0)
    path = tmp_path / "shion_pending_tasks.json"
    tasks = [
        {"id": "b", "topic": "最近の約束", "status": "pending",
         "promised_at": (now - timedelta(days=1)).isoformat()},
        {"id": "c", "topic": "完了済み", "status": "done"},
    ]
    _write(path, tasks)
    monkeypatch.setattr(pending, "PENDING_PATH", str(path))
    before = path.stat().st_mtime_ns

    pending.reconcile_pending(now=now)

    # 変更が無ければ書き込まない
    assert path.stat().st_mtime_ns == before
