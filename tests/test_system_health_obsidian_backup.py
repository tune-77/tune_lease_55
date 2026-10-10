from __future__ import annotations

import json
from pathlib import Path

from scripts.check_system_health import check_backup_snapshot


def _snapshot(root: Path, manifest: dict) -> Path:
    path = root / "Obsidian Vault_20261010_010000"
    path.mkdir(parents=True)
    (path / "backup_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_backup_health_requires_complete_manifest(tmp_path):
    _snapshot(tmp_path, {"status": "partial", "file_count": 10, "copied_count": 9, "failed_count": 1})

    result = check_backup_snapshot(tmp_path, max_age_hours=24)

    assert result.ok is False
    assert "status=partial" in result.message and "copied=9/10" in result.message


def test_backup_health_accepts_complete_snapshot(tmp_path):
    _snapshot(tmp_path, {"status": "complete", "file_count": 10, "copied_count": 10, "failed_count": 0})

    result = check_backup_snapshot(tmp_path, max_age_hours=24)

    assert result.ok is True
    assert "files=10" in result.message
