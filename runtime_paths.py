"""Shared path helpers for local and Cloud Run execution."""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OBSIDIAN_VAULT = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"


def get_data_dir() -> Path:
    raw = os.environ.get("DATA_DIR") or os.environ.get("LEASE_DATA_DIR")
    return Path(raw) if raw else DEFAULT_DATA_DIR


def get_data_path(*parts: str) -> str:
    return str(get_data_dir().joinpath(*parts))


def get_db_path(name: str = "lease_data.db") -> str:
    return get_data_path(name)


def get_obsidian_vault_path() -> str:
    raw = os.environ.get("OBSIDIAN_VAULT_PATH") or os.environ.get("OBSIDIAN_VAULT") or ""
    if raw:
        return raw

    return str(DEFAULT_OBSIDIAN_VAULT)
