from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import run_pipeline_auto_recovery as recovery
from scripts import sync_main_checkout as guard


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str) -> None:
    for name, body in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(body, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


@pytest.fixture()
def repos(tmp_path: Path) -> tuple[Path, Path]:
    """origin（bare）・メインチェックアウト（main）・上流に push する別作業場（up）。main は1コミット遅れ。"""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "master", str(origin)], check=True)
    up = tmp_path / "up"
    subprocess.run(["git", "clone", "-q", str(origin), str(up)], check=True, capture_output=True)
    for repo in (up,):
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "t")
    _commit(up, {"scripts/a.py": "v1\n", "data/rules.json": "{}\n", "data/local.json": "{}\n"}, "init")
    _git(up, "push", "-q", "origin", "master")
    main = tmp_path / "main"
    subprocess.run(["git", "clone", "-q", str(origin), str(main)], check=True, capture_output=True)
    _git(main, "config", "user.email", "t@example.com")
    _git(main, "config", "user.name", "t")
    _commit(up, {"scripts/a.py": "v2\n", "data/rules.json": '{"rule": 2}\n'}, "upstream change")
    _git(up, "push", "-q", "origin", "master")
    return main, up


def test_behind_master_with_unrelated_dirty_data_is_fast_forwarded(repos) -> None:
    main, _ = repos
    (main / "data" / "local.json").write_text('{"working": true}\n', encoding="utf-8")  # 作業中データ
    (main / "data" / "new_untracked.json").write_text("{}\n", encoding="utf-8")

    info = guard.sync(main)

    assert info["status"] == "fast_forwarded" and info["aligned_files"] == []
    assert (main / "scripts" / "a.py").read_text() == "v2\n"
    assert (main / "data" / "local.json").read_text() == '{"working": true}\n'  # 作業中データは保持
    assert (main / "data" / "new_untracked.json").exists()
    assert guard.warning_line(info) is None


def test_overlapping_file_identical_to_upstream_is_aligned_then_ff(repos) -> None:
    main, _ = repos
    (main / "data" / "rules.json").write_text('{"rule": 2}\n', encoding="utf-8")  # 上流と同一（手で同期済み）

    info = guard.sync(main)

    assert info["status"] == "fast_forwarded" and info["aligned_files"] == ["data/rules.json"]
    assert (main / "data" / "rules.json").read_text() == '{"rule": 2}\n'
    assert _git(main, "status", "--porcelain") == ""


def test_overlapping_file_that_differs_blocks_ff_and_changes_nothing(repos) -> None:
    main, _ = repos
    head = _git(main, "rev-parse", "HEAD")
    (main / "data" / "rules.json").write_text('{"rule": "mine"}\n', encoding="utf-8")

    info = guard.sync(main)

    assert info["status"] == "blocked_by_local_changes" and info["conflicting_files"] == ["data/rules.json"]
    assert _git(main, "rev-parse", "HEAD") == head
    assert (main / "data" / "rules.json").read_text() == '{"rule": "mine"}\n'
    assert "自動 ff を見送り" in guard.warning_line(info)


def test_local_commit_on_master_is_not_fast_forwarded(repos) -> None:
    main, _ = repos
    _commit(main, {"notes.md": "local\n"}, "local work")

    info = guard.sync(main)

    assert info["status"] == "diverged" and info["ahead"] == 1 and info["behind"] == 1


def test_other_branch_is_left_alone_and_warned(repos) -> None:
    main, _ = repos
    _git(main, "checkout", "-q", "-b", "feat/pilot")
    (main / "data" / "local.json").write_text("dirty\n", encoding="utf-8")

    info = guard.sync(main)

    assert info["status"] == "not_main_branch" and _git(main, "rev-parse", "--abbrev-ref", "HEAD") == "feat/pilot"
    assert guard.warning_line(info) == (
        "🌿 メインチェックアウトが master 以外: `feat/pilot`（未コミット1件、origin/master より1コミット遅れ）。"
        "定期処理と API はこのブランチのコードで動く"
    )


def test_fetch_failure_changes_nothing_and_warns(repos) -> None:
    main, _ = repos
    _git(main, "remote", "set-url", "origin", str(main.parent / "missing.git"))
    head = _git(main, "rev-parse", "HEAD")

    info = guard.sync(main)

    assert info["status"] == "undetermined" and "git fetch 失敗" in info["error"]
    assert _git(main, "rev-parse", "HEAD") == head
    assert "判定できず" in guard.warning_line(info)


def test_morning_warning_reads_live_branch_and_stored_reason(repos, tmp_path) -> None:
    main, _ = repos
    _git(main, "fetch", "-q", "origin")
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"status": "blocked_by_local_changes", "conflicting_files": ["data/rules.json"]}))
    line = recovery.main_checkout_warning(main, status)
    assert "自動 ff を見送り" in line and line.endswith(": data/rules.json")

    _git(main, "merge", "-q", "--ff-only", "origin/master")
    assert recovery.main_checkout_warning(main, status) is None  # 今は最新なので古い状態ファイルは無視

    _git(main, "checkout", "-q", "-b", "fix/x")
    assert "`fix/x`" in recovery.main_checkout_warning(main, status)


def test_dirty_paths_handles_renames(repos) -> None:
    main, _ = repos
    _git(main, "mv", "scripts/a.py", "scripts/b.py")
    assert set(guard.dirty_paths(main)) == {"scripts/a.py", "scripts/b.py"}
