"""weekly_self_management.py のタイムスタンプ抽出ドリフト検知テスト。

ledger.jsonl のタイムスタンプ用フィールド名が変わると、全エントリが
「時刻なし」扱いでスキップされ Weekly Log が毎週0件になっていた実障害が
過去にあった（recorded_at 未読み込み）。同型の再発を検知できるか確認する。
"""
from __future__ import annotations

import json

from scripts import weekly_self_management as weekly


def _write_ledger(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_main_warns_and_exits_1_when_ledger_has_rows_but_no_timestamps(tmp_path, monkeypatch, capsys):
    ledger_path = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger_path,
        [
            {"key": "a", "title": "改善A", "status": "applied"},
            {"key": "b", "title": "改善B", "status": "proposed"},
        ],
    )
    weekly_log_path = tmp_path / "WEEKLY_LOG.md"
    monkeypatch.setattr(weekly, "LEDGER_PATH", ledger_path)
    monkeypatch.setattr(weekly, "WEEKLY_LOG_PATH", weekly_log_path)
    monkeypatch.setattr(weekly, "is_monday", lambda: True)

    exit_code = weekly.main()

    assert exit_code == 1
    assert "ドリフト" in capsys.readouterr().err
    assert not weekly_log_path.exists()


def test_main_returns_0_when_no_entries_in_the_last_week(tmp_path, monkeypatch, capsys):
    """タイムスタンプは有効だが、直近7日にたまたま0件（正常系）は異常と誤検知しない。"""
    ledger_path = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger_path,
        [{"key": "old", "title": "古い改善", "status": "applied", "recorded_at": "2000-01-01T00:00:00+09:00"}],
    )
    weekly_log_path = tmp_path / "WEEKLY_LOG.md"
    monkeypatch.setattr(weekly, "LEDGER_PATH", ledger_path)
    monkeypatch.setattr(weekly, "WEEKLY_LOG_PATH", weekly_log_path)
    monkeypatch.setattr(weekly, "is_monday", lambda: True)

    exit_code = weekly.main()

    assert exit_code == 0
    assert capsys.readouterr().err == ""
    assert "適用済み (applied) | 0 件" in weekly_log_path.read_text(encoding="utf-8")
