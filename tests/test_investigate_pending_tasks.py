"""investigate_pending_tasks.py の下調べ結果が消えていないかのテスト。

search_lease_wiki が例外を握りつぶして常に空を返す（RAG導線が壊れている）と、
未調査タスクが有るのに1件も finding が付かず main() が静かに正常終了し続ける
リスクがある。「本当に約束が無い日」との区別だけ確認する。
"""
from __future__ import annotations

import json
import sys

from scripts import investigate_pending_tasks as inv


def _write(path, tasks):
    path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")


def test_main_warns_and_exits_1_when_findings_all_fail(tmp_path, monkeypatch, capsys):
    """未調査タスクは有るのに1件も finding が付かないのは検索導線が壊れている疑い。"""
    path = tmp_path / "shion_pending_tasks.json"
    _write(
        path,
        [
            {"id": "a", "topic": "残価の根拠は？", "status": "pending", "promised_at": "2999-01-01T00:00:00"},
            {"id": "b", "topic": "耐用年数の考え方", "status": "pending", "promised_at": "2999-01-01T00:00:00"},
        ],
    )
    monkeypatch.setattr(inv.pending_mod, "PENDING_PATH", str(path))
    monkeypatch.setattr(inv, "investigate_topic", lambda topic: "")
    monkeypatch.setattr(sys, "argv", ["investigate_pending_tasks.py"])

    exit_code = inv.main()

    assert exit_code == 1
    assert "RAG導線" in capsys.readouterr().err


def test_main_returns_0_when_no_pending_tasks(tmp_path, monkeypatch, capsys):
    """下調べる約束が1件も無い日は正常（0件を異常と誤検知しない）。"""
    path = tmp_path / "shion_pending_tasks.json"
    _write(path, [])
    monkeypatch.setattr(inv.pending_mod, "PENDING_PATH", str(path))
    monkeypatch.setattr(sys, "argv", ["investigate_pending_tasks.py"])

    exit_code = inv.main()

    assert exit_code == 0
    assert capsys.readouterr().err == ""
