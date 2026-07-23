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
