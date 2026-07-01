"""runtime_paths.ensure_cloudrun_demo_db_seeded の一回限り実行保証のテスト。

過去の実装では毎SQLite接続時にDB復元チェックを実行しており、稼働中の接続と
競合して本番DBの-wal/-shmを削除・上書きしうる自己破壊的な競合状態があった。
プロセス生存中は一度だけ実行されることを保証する。
"""

from __future__ import annotations

import runtime_paths


def test_seed_runs_only_once(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        runtime_paths, "_do_ensure_cloudrun_demo_db_seeded", lambda: calls.append(1)
    )
    monkeypatch.setattr(runtime_paths, "_demo_db_seed_done", False)

    runtime_paths.ensure_cloudrun_demo_db_seeded()
    runtime_paths.ensure_cloudrun_demo_db_seeded()
    runtime_paths.ensure_cloudrun_demo_db_seeded()

    assert len(calls) == 1
