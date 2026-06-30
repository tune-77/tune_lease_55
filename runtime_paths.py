"""Shared path helpers for local and Cloud Run execution."""
from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OBSIDIAN_VAULT = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/obsidian-vault"
LEGACY_OBSIDIAN_VAULT = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"


def get_data_dir() -> Path:
    raw = os.environ.get("DATA_DIR") or os.environ.get("LEASE_DATA_DIR")
    return Path(raw) if raw else DEFAULT_DATA_DIR


def get_data_path(*parts: str) -> str:
    return str(get_data_dir().joinpath(*parts))


def get_db_path(name: str = "lease_data.db") -> str:
    raw = os.environ.get("DB_PATH")
    if raw:
        return raw if os.path.isabs(raw) else str(REPO_ROOT / raw)
    return get_data_path(name)


def _sqlite_has_past_cases(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with sqlite3.connect(str(path), timeout=5) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='past_cases'"
            ).fetchone()
            if not row:
                return False
            count_row = conn.execute("SELECT COUNT(*) FROM past_cases").fetchone()
            return bool(count_row and int(count_row[0] or 0) > 0)
    except sqlite3.Error:
        return False


def ensure_cloudrun_demo_db_seeded() -> None:
    """Restore packaged demo DBs if Cloud Run created empty SQLite files first."""
    if os.environ.get("CLOUDRUN_DATA_MODE") != "demo":
        return

    bundle_root = Path(os.environ.get("CLOUDRUN_BUNDLE_DIR") or (REPO_ROOT / ".cloudrun_bundle"))
    bundle_data = bundle_root / "data"
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    for db_name in ("demo.db", "lease_data.db"):
        src = bundle_data / db_name
        dst = data_dir / db_name
        if not src.exists():
            continue
        if _sqlite_has_past_cases(dst):
            continue
        for suffix in ("", "-wal", "-shm"):
            stale = Path(str(dst) + suffix)
            if stale.exists():
                try:
                    stale.unlink()
                except OSError:
                    pass
        shutil.copy2(src, dst)
        print(f"[runtime_paths] restored demo DB from bundle: {dst}")


def get_obsidian_vault_path() -> str:
    raw = os.environ.get("OBSIDIAN_VAULT_PATH") or os.environ.get("OBSIDIAN_VAULT") or ""
    if raw:
        return raw

    if DEFAULT_OBSIDIAN_VAULT.exists():
        return str(DEFAULT_OBSIDIAN_VAULT)
    if LEGACY_OBSIDIAN_VAULT.exists():
        return str(LEGACY_OBSIDIAN_VAULT)

    return str(DEFAULT_OBSIDIAN_VAULT)
