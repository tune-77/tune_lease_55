"""チャット起票の即時バックグラウンド実行（execute_chat_quick_fix）。

- 既定（CHAT_QUICK_FIX_IMMEDIATE_EXECUTION 未設定）では常に起票のみで完結し、
  本物の claude --print サブプロセスは絶対に起動しない
  （テスト実行やCIで誤って本番エージェントを起動しないための必須ガード）。
- opt-in 環境変数を設定した場合のみ、codex-safe 条件を満たす候補が
  バックグラウンドで即時実行される。テストでは threading.Thread を
  同期実行スタブに差し替え、run_item も差し替えて実サブプロセスを起動しない。
"""
from __future__ import annotations

import json

import pytest

from scripts import execute_chat_quick_fix as ecf


class _ImmediateThread:
    """threading.Thread の代わりに target を同期的にその場で実行するテスト用スタブ。"""

    def __init__(self, target=None, args=(), kwargs=None, name=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(ecf, "record_status", lambda *a, **k: None)
    monkeypatch.setattr(ecf, "_get_gemini_api_key", lambda root: "")
    monkeypatch.delenv("CHAT_QUICK_FIX_IMMEDIATE_EXECUTION", raising=False)
    monkeypatch.delenv("CODEX_QUEUE_DISABLED", raising=False)
    monkeypatch.delenv("CODEX_QUEUE_DAILY_LIMIT", raising=False)
    return tmp_path


_SAFE_RECORD = {
    "id": "chat_test_safe",
    "title": "FAQページの文言タイポ修正",
    "description": "frontend/src/app/faq/page.tsx の文言タイポを直して",
    "target_module": "frontend/src/app/faq/page.tsx",
}


def test_disabled_by_default_never_touches_run_item(env, monkeypatch):
    """opt-in環境変数が未設定なら、安全な候補でも実サブプロセス経路(run_item)に一切触れない。"""
    calls = []
    monkeypatch.setattr(ecf, "run_item", lambda item, **k: calls.append(item) or {})
    monkeypatch.setattr(ecf.threading, "Thread", _ImmediateThread)

    result = ecf.start_execution(_SAFE_RECORD, root=env)

    assert result["execution"] == "queued_for_batch"
    assert "CHAT_QUICK_FIX_IMMEDIATE_EXECUTION" in result["reason"]
    assert calls == []
    assert not (env / "data" / "chat_quick_fix_executed.json").exists()


def test_risky_request_never_executes_even_when_enabled(env, monkeypatch):
    monkeypatch.setenv("CHAT_QUICK_FIX_IMMEDIATE_EXECUTION", "1")
    calls = []
    monkeypatch.setattr(ecf, "run_item", lambda item, **k: calls.append(item) or {})
    monkeypatch.setattr(ecf.threading, "Thread", _ImmediateThread)

    risky = {
        "id": "chat_test_risky",
        "title": "スコアリングの承認閾値を70に下げて",
        "description": "スコアリングの承認閾値を70に下げて",
        "target_module": None,
    }
    result = ecf.start_execution(risky, root=env)

    assert result["execution"] == "queued_for_batch"
    assert calls == []


def test_enabled_and_safe_runs_in_background_and_marks_executed(env, monkeypatch):
    monkeypatch.setenv("CHAT_QUICK_FIX_IMMEDIATE_EXECUTION", "1")
    monkeypatch.setattr(ecf.threading, "Thread", _ImmediateThread)

    calls = []

    def fake_run_item(item, **k):
        calls.append(item["id"])
        return {
            "id": item["id"], "title": item.get("title"), "exit_code": 0,
            "stdout": "diff applied", "stderr": "", "backend": "claude",
            "started_at": "", "finished_at": "",
        }

    monkeypatch.setattr(ecf, "run_item", fake_run_item)

    result = ecf.start_execution(_SAFE_RECORD, root=env)

    assert result == {"execution": "started", "id": "chat_test_safe"}
    assert calls == ["chat_test_safe"]

    executed = json.loads((env / "data" / "chat_quick_fix_executed.json").read_text(encoding="utf-8"))
    assert executed == ["chat_test_safe"]

    import datetime as dt
    date_tag = dt.date.today().strftime("%Y%m%d")
    result_path = env / "reports" / f"codex_queue_result_{date_tag}_chat.json"
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["results"][0]["id"] == "chat_test_safe"
    assert report["results"][0]["exit_code"] == 0


def test_already_executed_id_is_skipped(env, monkeypatch):
    monkeypatch.setenv("CHAT_QUICK_FIX_IMMEDIATE_EXECUTION", "1")
    monkeypatch.setattr(ecf.threading, "Thread", _ImmediateThread)
    calls = []
    monkeypatch.setattr(ecf, "run_item", lambda item, **k: calls.append(item) or {})

    executed_path = env / "data" / "chat_quick_fix_executed.json"
    executed_path.parent.mkdir(parents=True, exist_ok=True)
    executed_path.write_text(json.dumps(["chat_test_safe"]), encoding="utf-8")

    result = ecf.start_execution(_SAFE_RECORD, root=env)

    assert result == {"execution": "skipped", "reason": "既に実行済みです"}
    assert calls == []


def test_kill_switch_blocks_immediate_execution(env, monkeypatch):
    monkeypatch.setenv("CHAT_QUICK_FIX_IMMEDIATE_EXECUTION", "1")
    monkeypatch.setenv("CODEX_QUEUE_DISABLED", "1")
    monkeypatch.setattr(ecf.threading, "Thread", _ImmediateThread)
    calls = []
    monkeypatch.setattr(ecf, "run_item", lambda item, **k: calls.append(item) or {})

    result = ecf.start_execution(_SAFE_RECORD, root=env)

    assert result["execution"] == "queued_for_batch"
    assert "CODEX_QUEUE_DISABLED" in result["reason"]
    assert calls == []


def test_daily_limit_blocks_immediate_execution(env, monkeypatch):
    monkeypatch.setenv("CHAT_QUICK_FIX_IMMEDIATE_EXECUTION", "1")
    monkeypatch.setenv("CODEX_QUEUE_DAILY_LIMIT", "0")
    monkeypatch.setattr(ecf.threading, "Thread", _ImmediateThread)
    calls = []
    monkeypatch.setattr(ecf, "run_item", lambda item, **k: calls.append(item) or {})

    result = ecf.start_execution(_SAFE_RECORD, root=env)

    assert result["execution"] == "queued_for_batch"
    assert "上限" in result["reason"]
    assert calls == []
