"""scripts/check_cloudrun_demo_readiness.py の自律パイプライン健全性チェック（backlog §9）。"""

import json
from pathlib import Path

import scripts.check_cloudrun_demo_readiness as mod


def test_check_recent_pipeline_health_fails_on_consecutive_abort(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "codex_queue_result_20260805.json").write_text(
        json.dumps({"guards": {"aborted_by_consecutive_failures": True, "carried_over": ["REV-001"]}}),
        encoding="utf-8",
    )

    checks = mod.CheckRun()
    mod.check_recent_pipeline_health(checks)

    assert checks.failures
    assert "連続失敗のため停止中" in checks.failures[0]


def test_check_recent_pipeline_health_passes_when_healthy(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "codex_queue_result_20260805.json").write_text(
        json.dumps({"guards": {"aborted_by_consecutive_failures": False}}),
        encoding="utf-8",
    )

    checks = mod.CheckRun()
    mod.check_recent_pipeline_health(checks)

    assert not checks.failures
    assert not checks.warnings


def test_check_recent_pipeline_health_skips_when_no_result_files(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()

    checks = mod.CheckRun()
    mod.check_recent_pipeline_health(checks)

    assert not checks.failures
    assert not checks.warnings


def test_check_recent_pipeline_health_warns_on_corrupt_json(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "codex_queue_result_20260805.json").write_text("not json", encoding="utf-8")

    checks = mod.CheckRun()
    mod.check_recent_pipeline_health(checks)

    assert not checks.failures
    assert checks.warnings
