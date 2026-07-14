from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "extract_obsidian_improvements_for_test",
        ROOT / "scripts" / "extract_obsidian_improvements.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_old_raw_failure_is_not_active_candidate() -> None:
    module = _load_module()

    assert module._is_stale_raw_failure(
        "蘭丸の機能で通信エラー障害が発生し、小説の作成ができない状態。",
        "2026-06-13",
        today=dt.date(2026, 7, 15),
    )
    assert module._is_stale_raw_failure(
        "リース審査AIがユーザーのぼやきを検知する機能が停止している、または機能していない。",
        "2026-06-23",
        today=dt.date(2026, 7, 15),
    )


def test_recent_or_recurring_failure_stays_active() -> None:
    module = _load_module()

    assert not module._is_stale_raw_failure(
        "蘭丸の機能で通信エラー障害が発生し、小説の作成ができない状態。",
        "2026-07-14",
        today=dt.date(2026, 7, 15),
    )
    assert not module._is_stale_raw_failure(
        "蘭丸の通信エラーがまだ再発している。",
        "2026-06-13",
        today=dt.date(2026, 7, 15),
    )


def test_non_failure_improvement_is_not_stale() -> None:
    module = _load_module()

    assert not module._is_stale_raw_failure(
        "ホーム画面に改善ログの要点を表示したい。",
        "2026-06-13",
        today=dt.date(2026, 7, 15),
    )
