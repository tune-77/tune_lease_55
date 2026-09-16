from __future__ import annotations

from scripts import sync_codex_pr_status as syncer


def test_gh_pr_list_signals_failure_on_nonzero_returncode(monkeypatch):
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "gh: authentication expired"

    monkeypatch.setattr(syncer.subprocess, "run", lambda *a, **k: _Proc())

    rows, ok = syncer.gh_pr_list("merged")

    assert rows == []
    assert ok is False


def test_gh_pr_list_signals_failure_on_exception(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("gh not installed")

    monkeypatch.setattr(syncer.subprocess, "run", _raise)

    rows, ok = syncer.gh_pr_list("merged")

    assert rows == []
    assert ok is False


def test_gh_pr_list_ok_true_on_success(monkeypatch):
    class _Proc:
        returncode = 0
        stdout = "[]"
        stderr = ""

    monkeypatch.setattr(syncer.subprocess, "run", lambda *a, **k: _Proc())

    rows, ok = syncer.gh_pr_list("merged")

    assert rows == []
    assert ok is True


def test_main_returns_nonzero_when_gh_fails(tmp_path, monkeypatch, capsys):
    """gh pr list が壊れているのに『同期対象のPRなし』で無条件成功しないことを
    確認する回帰テスト（sync_memory_from_daily.pyと同型のバグの再発防止）。"""
    monkeypatch.setattr(syncer, "repo_root", lambda: tmp_path)

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "gh: not authenticated"

    monkeypatch.setattr(syncer.subprocess, "run", lambda *a, **k: _Proc())

    exit_code = syncer.main()

    assert exit_code == 1
    assert "同期できていません" in capsys.readouterr().out


def test_main_returns_zero_when_gh_succeeds_with_no_prs(tmp_path, monkeypatch):
    monkeypatch.setattr(syncer, "repo_root", lambda: tmp_path)

    class _Proc:
        returncode = 0
        stdout = "[]"
        stderr = ""

    monkeypatch.setattr(syncer.subprocess, "run", lambda *a, **k: _Proc())

    assert syncer.main() == 0
