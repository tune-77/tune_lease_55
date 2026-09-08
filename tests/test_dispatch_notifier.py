"""dispatch_notifier.accumulate_and_maybe_dispatch() の日またぎ積み上げ動作のテスト。"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "auto-improvement-pipeline"
    / "scripts"
    / "dispatch_notifier.py"
)


def load_dispatch_notifier_module():
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("dispatch_notifier_test_target", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_PENDING_CANDIDATES_PATH", tmp_path / "pending_dispatch_candidates.json")
    monkeypatch.setattr(module, "_DISPATCH_QUEUE_PATH", tmp_path / "dispatch_queue.jsonl")
    monkeypatch.setattr(module, "_PENDING_LOG_PATH", tmp_path / "pending_approvals.jsonl")
    monkeypatch.setattr(module, "_get_slack_webhook", lambda: None)


def _small_ui_candidate(id_: str) -> dict:
    return {"id": id_, "title": f"表示ラベルを調整する {id_}", "description": ""}


def _large_candidate(id_: str) -> dict:
    return {"id": id_, "title": f"審査フローを再設計する {id_}", "description": ""}


def test_accumulates_small_ui_candidates_until_threshold(tmp_path, monkeypatch):
    module = load_dispatch_notifier_module()
    _setup(module, tmp_path, monkeypatch)

    for i in range(1, 5):
        result = module.accumulate_and_maybe_dispatch([_small_ui_candidate(f"SU-{i}")], "2026-08-2" + str(i))
        assert result is None
        pending = json.loads(module._PENDING_CANDIDATES_PATH.read_text(encoding="utf-8"))
        assert len(pending) == i
        assert not module._DISPATCH_QUEUE_PATH.exists()


def test_dispatches_when_small_ui_reaches_five_and_clears_backlog(tmp_path, monkeypatch):
    module = load_dispatch_notifier_module()
    _setup(module, tmp_path, monkeypatch)

    for i in range(1, 5):
        assert module.accumulate_and_maybe_dispatch([_small_ui_candidate(f"SU-{i}")], "2026-08-25") is None

    result = module.accumulate_and_maybe_dispatch([_small_ui_candidate("SU-5")], "2026-08-26")

    assert result is not None
    assert len(result["candidates"]) > 0
    queue_lines = module._DISPATCH_QUEUE_PATH.read_text(encoding="utf-8").splitlines()
    assert len(queue_lines) == 1
    pending_after = json.loads(module._PENDING_CANDIDATES_PATH.read_text(encoding="utf-8"))
    assert pending_after == []


def test_single_large_candidate_dispatches_immediately(tmp_path, monkeypatch):
    module = load_dispatch_notifier_module()
    _setup(module, tmp_path, monkeypatch)

    result = module.accumulate_and_maybe_dispatch([_large_candidate("LG-1")], "2026-08-27")

    assert result is not None
    assert module._DISPATCH_QUEUE_PATH.exists()
    pending_after = json.loads(module._PENDING_CANDIDATES_PATH.read_text(encoding="utf-8"))
    assert pending_after == []


def test_duplicate_id_across_calls_is_not_double_counted(tmp_path, monkeypatch):
    module = load_dispatch_notifier_module()
    _setup(module, tmp_path, monkeypatch)

    candidate = _small_ui_candidate("SU-DUP")
    assert module.accumulate_and_maybe_dispatch([candidate], "2026-08-28") is None
    assert module.accumulate_and_maybe_dispatch([candidate], "2026-08-29") is None

    pending = json.loads(module._PENDING_CANDIDATES_PATH.read_text(encoding="utf-8"))
    assert len(pending) == 1
