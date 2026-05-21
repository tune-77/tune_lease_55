#!/usr/bin/env python3
"""Backup an Obsidian vault to timestamped snapshot folders.

The script copies the whole vault directory, including `.obsidian/`,
into a backup root as:

    <backup-root>/<vault-name>_<YYYYmmdd_HHMMSS>/

It is intentionally conservative:
- it never deletes the source vault
- it supports dry-run
- it keeps only the newest N snapshots per vault prefix

Environment variables:
- OBSIDIAN_VAULT: override source vault path
- OBSIDIAN_BACKUP_ROOT: override backup root path
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_VAULT_CANDIDATES = [
    Path(os.environ.get("OBSIDIAN_VAULT", "")).expanduser() if os.environ.get("OBSIDIAN_VAULT") else None,
    Path.home() / "Documents" / "Obsidian Vault",
    Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents",
    Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Obsidian Vault",
]

DEFAULT_BACKUP_ROOT = Path(
    os.environ.get(
        "OBSIDIAN_BACKUP_ROOT",
        str(Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "tune_lease_55_backups" / "obsidian"),
    )
).expanduser()

DEFAULT_EXCLUDES = [
    ".DS_Store",
    "*.tmp",
    "*.swp",
    "*.swo",
    ".obsidian/cache/*",
    ".obsidian/cache/**",
]


@dataclass
class BackupSummary:
    vault: Path
    destination: Path
    file_count: int
    total_bytes: int
    excluded_count: int
    dry_run: bool


def _candidate_vaults() -> list[Path]:
    out: list[Path] = []
    for path in DEFAULT_VAULT_CANDIDATES:
        if path and path.exists() and path.is_dir():
            out.append(path)
    return out


def find_vault(override: str | None = None) -> Path:
    if override:
        path = Path(override).expanduser()
        if path.exists() and path.is_dir():
            return path
        raise FileNotFoundError(f"Obsidian vault not found: {path}")

    candidates = _candidate_vaults()
    if not candidates:
        raise FileNotFoundError(
            "Obsidian vault not found. Set OBSIDIAN_VAULT or pass --vault."
        )
    return candidates[0]


def _matches_any(rel_posix: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(rel_posix, pat) for pat in patterns)


def _iter_vault_files(vault: Path, excludes: list[str]) -> tuple[list[Path], int]:
    files: list[Path] = []
    excluded = 0
    for path in vault.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(vault).as_posix()
        if _matches_any(rel, excludes):
            excluded += 1
            continue
        files.append(path)
    return files, excluded


def _snapshot_name(vault: Path, ts: str) -> str:
    return f"{vault.name}_{ts}"


def _unique_destination(root: Path, base_name: str) -> Path:
    dest = root / base_name
    if not dest.exists():
        return dest
    suffix = 1
    while True:
        candidate = root / f"{base_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _cleanup_old_snapshots(root: Path, vault_name: str, keep: int) -> list[Path]:
    if keep <= 0 or not root.exists():
        return []
    prefix = f"{vault_name}_"
    snapshots = [
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    ]
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    removed: list[Path] = []
    for old in snapshots[keep:]:
        shutil.rmtree(old, ignore_errors=True)
        removed.append(old)
    return removed


def backup_vault(
    vault: Path,
    backup_root: Path = DEFAULT_BACKUP_ROOT,
    keep: int = 10,
    dry_run: bool = False,
    excludes: list[str] | None = None,
) -> BackupSummary:
    excludes = list(excludes or DEFAULT_EXCLUDES)
    files, excluded_count = _iter_vault_files(vault, excludes)
    total_bytes = sum(p.stat().st_size for p in files)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = _unique_destination(backup_root, _snapshot_name(vault, ts))

    if dry_run:
        return BackupSummary(
            vault=vault,
            destination=dest,
            file_count=len(files),
            total_bytes=total_bytes,
            excluded_count=excluded_count,
            dry_run=True,
        )

    backup_root.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=False)

    for src in files:
        rel = src.relative_to(vault)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)

    manifest = {
        "vault": str(vault),
        "destination": str(dest),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "excluded_count": excluded_count,
        "excludes": excludes,
    }
    (dest / "backup_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _cleanup_old_snapshots(backup_root, vault.name, keep)
    return BackupSummary(
        vault=vault,
        destination=dest,
        file_count=len(files),
        total_bytes=total_bytes,
        excluded_count=excluded_count,
        dry_run=False,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a timestamped backup of an Obsidian vault.")
    parser.add_argument("--vault", default=None, help="Source Obsidian vault path. Defaults to auto-detect.")
    parser.add_argument("--backup-root", default=str(DEFAULT_BACKUP_ROOT), help="Backup root directory.")
    parser.add_argument("--keep", type=int, default=10, help="Keep newest N snapshots per vault.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without copying files.")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional glob pattern to exclude. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        vault = find_vault(args.vault)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = backup_vault(
        vault=vault,
        backup_root=Path(args.backup_root).expanduser(),
        keep=max(0, int(args.keep)),
        dry_run=bool(args.dry_run),
        excludes=DEFAULT_EXCLUDES + list(args.exclude or []),
    )

    mode = "DRY-RUN" if summary.dry_run else "BACKED UP"
    size_mb = summary.total_bytes / (1024 * 1024)
    print(
        f"{mode}: {summary.file_count} files, {size_mb:.1f} MB, "
        f"excluded {summary.excluded_count} files"
    )
    print(f"vault: {summary.vault}")
    print(f"destination: {summary.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
