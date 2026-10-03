"""Cloud Run 停止中は Cloud Run 向けの同期を「停止中のためスキップ」にし、朝報では失敗に数えず1行だけ出す。"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import cloudrun_pause
import silent_failure_log

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_switch_reads_config_and_env_overrides(monkeypatch, tmp_path) -> None:
    cfg = tmp_path / "cloudrun_pause.json"
    monkeypatch.setattr(cloudrun_pause, "CONFIG_PATH", cfg)
    monkeypatch.delenv("CLOUDRUN_PAUSED", raising=False)
    assert cloudrun_pause.is_paused() is False  # 設定ファイルが無ければ稼働扱い
    cfg.write_text(json.dumps({"paused": True}), encoding="utf-8")
    assert cloudrun_pause.is_paused() is True
    monkeypatch.setenv("CLOUDRUN_PAUSED", "0")
    assert cloudrun_pause.is_paused() is False


def test_sync_scripts_skip_without_touching_gcs(monkeypatch, tmp_path) -> None:
    log = tmp_path / "sf.jsonl"
    env = {"PATH": "/usr/bin:/bin", "CLOUDRUN_PAUSED": "1", "SILENT_FAILURE_LOG_PATH": str(log), "GCS_BUCKET": "invalid-bucket-for-test"}
    for script in ("sync_cloudrun_inputs_from_gcs.py", "sync_ledger_to_gcs.py", "icloud_to_gcs_sync.py"):
        result = subprocess.run([sys.executable, f"scripts/{script}"], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, (script, result.stderr[-1500:])
        assert "Cloud Run 停止中" in result.stdout
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert {r["kind"] for r in rows} == {"paused_skip"}
    assert len(rows) == 3


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_morning_report_counts_paused_skips_as_one_line(tmp_path) -> None:
    now = dt.datetime(2026, 10, 5, 6, 0, tzinfo=dt.timezone(dt.timedelta(hours=9)))
    ts = (now - dt.timedelta(hours=2)).isoformat()
    path = tmp_path / "sf.jsonl"
    _write(path, [
        {"ts": ts, "component": "backup.sync_ledger_to_gcs.upload", "kind": "paused_skip"},
        {"ts": ts, "component": "backup.sync_cloudrun_inputs.from_gcs", "kind": "paused_skip", "repeat": 1},
    ])
    assert silent_failure_log.morning_report_lines(path, now=now) == ["- Cloud Run 停止中：同期3件スキップ（config/cloudrun_pause.json）"]

    # Cloud Run 向け以外の重要部品の失敗は今まで通り警告に出る
    _write(path, [
        {"ts": ts, "component": "backup.sync_ledger_to_gcs.upload", "kind": "paused_skip"},
        {"ts": ts, "component": "backup.case_data_backup.upload", "kind": "save_failed"},
    ])
    lines = silent_failure_log.morning_report_lines(path, now=now)
    assert lines[0].startswith("- ⚠️ 黙った失敗（直近24h）: 1件・1種")
    assert "case_data_backup" in "\n".join(lines) and "paused_skip" not in "\n".join(lines)
    assert lines[-1] == "- Cloud Run 停止中：同期1件スキップ（config/cloudrun_pause.json）"


def test_chromadb_sync_shell_skips_when_paused(tmp_path) -> None:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "CLOUDRUN_PAUSED": "1",
           "SILENT_FAILURE_LOG_PATH": str(tmp_path / "sf.jsonl"), "PYTHON_BIN": sys.executable}
    result = subprocess.run(["bash", "scripts/sync_chromadb_to_gcs.sh"], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "Cloud Run 停止中" in result.stdout and "gsutil" not in result.stdout
