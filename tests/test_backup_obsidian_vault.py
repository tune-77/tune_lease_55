from __future__ import annotations

import errno
import importlib.util
import json
import shutil
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backup_obsidian_vault.py"
_SPEC = importlib.util.spec_from_file_location("backup_obsidian_vault", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
backup = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = backup
_SPEC.loader.exec_module(backup)


def test_one_failed_copy_does_not_abort_snapshot(monkeypatch, tmp_path):
    vault = tmp_path / "Vault"
    (vault / "notes").mkdir(parents=True)
    (vault / "notes" / "ok.md").write_text("ok", encoding="utf-8")
    (vault / "無題のファイル 1.base").write_text("x", encoding="utf-8")
    real_copy2 = shutil.copy2

    def flaky_copy2(src, dst, *args, **kwargs):
        if Path(src).name == "無題のファイル 1.base":
            raise OSError(errno.EDEADLK, "Resource deadlock avoided")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(backup.shutil, "copy2", flaky_copy2)

    summary = backup.backup_vault(vault, backup_root=tmp_path / "backups", keep=0)

    assert (summary.destination / "notes" / "ok.md").read_text(encoding="utf-8") == "ok"
    assert [item["path"] for item in summary.failed] == ["無題のファイル 1.base"]
    manifest = json.loads((summary.destination / "backup_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert manifest["copied_count"] == 1
    assert manifest["failed_count"] == 1


def test_complete_snapshot_records_complete_status(tmp_path):
    vault = tmp_path / "Vault"
    vault.mkdir()
    (vault / "a.md").write_text("a", encoding="utf-8")

    summary = backup.backup_vault(vault, backup_root=tmp_path / "backups", keep=0)

    manifest = json.loads((summary.destination / "backup_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["failed_count"] == 0
    assert summary.failed == []


def test_suspicious_file_drop_preserves_older_complete_snapshot(tmp_path):
    vault = tmp_path / "Vault"
    vault.mkdir()
    for index in range(100):
        (vault / f"note-{index}.md").write_text(str(index), encoding="utf-8")

    first = backup.backup_vault(vault, backup_root=tmp_path / "backups", keep=1)
    for index in range(60):
        (vault / f"note-{index}.md").unlink()
    second = backup.backup_vault(vault, backup_root=tmp_path / "backups", keep=1)

    manifest = json.loads((second.destination / "backup_manifest.json").read_text(encoding="utf-8"))
    assert second.suspicious_drop is True
    assert manifest["status"] == "suspicious"
    assert manifest["previous_complete_file_count"] == 100
    assert first.destination.exists()
    assert second.destination.exists()


def test_small_vault_percentage_drop_is_suspicious(tmp_path):
    vault = tmp_path / "Vault"
    vault.mkdir()
    for index in range(40):
        (vault / f"note-{index}.md").write_text(str(index), encoding="utf-8")

    first = backup.backup_vault(vault, backup_root=tmp_path / "backups", keep=1)
    for note in vault.glob("*.md"):
        note.unlink()
    second = backup.backup_vault(vault, backup_root=tmp_path / "backups", keep=1)

    manifest = json.loads((second.destination / "backup_manifest.json").read_text(encoding="utf-8"))
    assert second.suspicious_drop is True
    assert manifest["status"] == "suspicious"
    assert first.destination.exists()


def test_repeated_suspicious_snapshots_are_bounded_while_complete_restore_point_is_pinned(tmp_path):
    vault = tmp_path / "Vault"
    vault.mkdir()
    for index in range(100):
        (vault / f"note-{index}.md").write_text(str(index), encoding="utf-8")

    complete = backup.backup_vault(vault, backup_root=tmp_path / "backups", keep=1)
    for index in range(60):
        (vault / f"note-{index}.md").unlink()
    for _ in range(3):
        latest = backup.backup_vault(vault, backup_root=tmp_path / "backups", keep=1)

    snapshots = list((tmp_path / "backups").glob("Vault_*"))
    assert len(snapshots) == 2
    assert complete.destination.exists()
    assert latest.destination.exists()
