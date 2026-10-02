from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import secrets
import sqlite3

import pytest

from scripts import backup_case_data as backup

KEY = secrets.token_bytes(32)


def test_judgment_daily_targets_are_also_in_weekly_and_weekly_plot_is_gone():
    assert set(backup.JUDGMENT_DAILY_TARGETS) <= set(backup.DEFAULT_TARGETS)
    assert "data/canonical_judgment_rules.json" in backup.JUDGMENT_DAILY_TARGETS
    assert "data/weekly_plot.json" not in backup.DEFAULT_TARGETS


def test_backup_encrypts_keeps_n_and_restore_verifies(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "data" / "jev_news_label_eval_1.jsonl").write_text('{"secret": "顧客A"}\n', encoding="utf-8")
    (repo / "data" / "jev_asset_label_eval_2.jsonl").write_text("{}\n", encoding="utf-8")
    with sqlite3.connect(repo / "data" / "lease_data.db") as conn:
        conn.execute("create table t (x)")
    monkeypatch.setattr(backup, "REPO_ROOT", repo)
    root = tmp_path / "out"
    (root / "judgment_daily_20261001_233000").mkdir(parents=True)  # 過去の平文フォルダは世代管理で消さない

    for _ in range(3):
        summary = backup.backup_case_data(root, ["data/jev_*label_eval*.jsonl", "data/lease_data.db", "data/none.json", "data/zz_*"], keep=2, prefix="judgment_daily", key=KEY)

    names = sorted(e.destination.rsplit("/", 1)[-1] for e in summary.backed_up)
    assert names == ["jev_asset_label_eval_2.jsonl", "jev_news_label_eval_1.jsonl", "lease_data.db"]
    assert summary.missing == ["data/none.json", "data/zz_*"]
    assert len(list(root.glob("judgment_daily_*.tar.gz.enc"))) == 2
    assert (root / "judgment_daily_20261001_233000").is_dir()
    archive = Path(summary.destination)
    assert "顧客A".encode() not in archive.read_bytes() and b"lease_data" not in archive.read_bytes()

    result = backup.restore_archive(archive, tmp_path / "restored", KEY)
    assert result["ok"] and result["files"] == 3 and result["sqlite_checked"] == 1
    with pytest.raises(ValueError, match="鍵が違う"):
        backup.restore_archive(archive, tmp_path / "wrong", secrets.token_bytes(32))


def test_backup_without_key_fails_and_writes_nothing(tmp_path, monkeypatch):
    def no_key():
        raise backup.BackupKeyError("キーチェーンから鍵を取得できない")

    monkeypatch.setattr(backup, "load_key", no_key)
    with pytest.raises(backup.BackupKeyError):
        backup.backup_case_data(tmp_path / "out", ["data/x.json"], keep=2)
    assert not (tmp_path / "out").exists()


def test_restore_detects_tampered_content(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "data" / "a.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(backup, "REPO_ROOT", repo)
    archive = Path(backup.backup_case_data(tmp_path / "out", ["data/a.json"], keep=2, key=KEY).destination)
    blob = bytearray(archive.read_bytes())
    blob[-1] ^= 1
    archive.write_bytes(bytes(blob))
    with pytest.raises(ValueError):
        backup.restore_archive(archive, tmp_path / "r", KEY)


def test_status_and_morning_line_warn_on_stale_and_failed(tmp_path):
    status = tmp_path / "status.json"
    obsidian = tmp_path / "obsidian"
    (obsidian / "Obsidian Vault_20261001_141311").mkdir(parents=True)
    now = dt.datetime(2026, 10, 2, 7, 0).astimezone()
    summary = backup.BackupSummary(created_at="", destination="d", backed_up=[], missing=[], removed_old=[])
    backup.record_status("judgment_daily", ok=True, summary=summary, path=status)
    data = json.loads(status.read_text())
    data["case_data"] = {"last_success": "2026-09-20T01:30:00+09:00", "ok": True}
    status.write_text(json.dumps(data))

    line = backup.morning_report_line(status_path=status, obsidian_root=obsidian, now=now)
    assert line.startswith("- ⚠️ バックアップ最終成功")
    assert "7日以上成功なし: case_data" in line and "obsidian 10/01 14:13" in line

    backup.record_status("judgment_daily", ok=False, error="OSError: disk", path=status)
    line = backup.morning_report_line(status_path=status, obsidian_root=obsidian, now=now)
    assert "直近失敗: judgment_daily" in line
