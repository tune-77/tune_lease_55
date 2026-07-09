#!/usr/bin/env python3
"""Backup case and learning data to timestamped snapshot folders."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP_ROOT = Path(
    os.environ.get(
        "CASE_DATA_BACKUP_ROOT",
        Path.home()
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs"
        / "tune_lease_55_backups"
        / "case_data",
    )
)

DEFAULT_TARGETS = [
    "data/lease_data.db",
    "data/screening_db.sqlite",
    "data/users.db",
    "data/novelist_agent.db",
    "data/math_discoveries.db",
    "data/lease_news_metrics.json",
    "data/model_review_state.json",
    "data/training_meta.json",
    "data/coeff_auto.json",
    "data/coeff_overrides.json",
    "data/ensemble_config.json",
    "data/weekly_plot.json",
    # 紫苑記憶システムの派生物。特に revisions（改訂宣言の真実の源）と
    # usage_log（鮮度更新の材料）は失うと再構築できない
    "data/shion_memory_index.json",
    "data/shion_memory_usage_log.jsonl",
    "data/shion_memory_revisions.jsonl",
    "data/shion_memory_health_state.json",
    "data/shion_memory_promotions.jsonl",
]


@dataclass
class BackupEntry:
    source: str
    destination: str
    size_bytes: int
    method: str


@dataclass
class BackupSummary:
    created_at: str
    destination: str
    backed_up: list[BackupEntry]
    missing: list[str]
    removed_old: list[str]


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _snapshot_dir(backup_root: Path, ts: str) -> Path:
    base = backup_root / f"case_data_{ts}"
    if not base.exists():
        return base
    counter = 1
    while True:
        candidate = backup_root / f"case_data_{ts}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def _sqlite_integrity_ok(db_path: Path) -> bool:
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("PRAGMA integrity_check;").fetchone()
    except sqlite3.Error:
        return False
    return bool(row and row[0] == "ok")


def _backup_sqlite(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(src) as source_conn:
        with sqlite3.connect(dst) as dest_conn:
            source_conn.backup(dest_conn)
    if not _sqlite_integrity_ok(dst):
        raise RuntimeError(f"backup integrity check failed: {dst}")


def _backup_file(src: Path, dst: Path) -> str:
    suffix = src.suffix.lower()
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        _backup_sqlite(src, dst)
        return "sqlite_backup_api"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copy2"


def _cleanup_old_snapshots(backup_root: Path, keep: int) -> list[str]:
    snapshots = sorted(
        [path for path in backup_root.glob("case_data_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    removed: list[str] = []
    for old in snapshots[keep:]:
        shutil.rmtree(old)
        removed.append(str(old))
    return removed


def backup_case_data(backup_root: Path, targets: list[str], keep: int) -> BackupSummary:
    ts = _timestamp()
    backup_root.mkdir(parents=True, exist_ok=True)
    dest_root = _snapshot_dir(backup_root, ts)
    dest_root.mkdir(parents=True, exist_ok=False)

    backed_up: list[BackupEntry] = []
    missing: list[str] = []

    for rel_target in targets:
        src = (REPO_ROOT / rel_target).resolve()
        if not src.exists():
            missing.append(rel_target)
            continue
        dst = dest_root / rel_target
        method = _backup_file(src, dst)
        backed_up.append(
            BackupEntry(
                source=str(src),
                destination=str(dst),
                size_bytes=dst.stat().st_size,
                method=method,
            )
        )

    manifest = BackupSummary(
        created_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        destination=str(dest_root),
        backed_up=backed_up,
        missing=missing,
        removed_old=[],
    )
    (dest_root / "backup_manifest.json").write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    removed = _cleanup_old_snapshots(backup_root, keep)
    manifest.removed_old.extend(removed)
    (dest_root / "backup_manifest.json").write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup case and learning data.")
    parser.add_argument("--backup-root", default=str(DEFAULT_BACKUP_ROOT), help="Snapshot root directory.")
    parser.add_argument("--keep", type=int, default=12, help="Number of snapshots to keep.")
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Relative path under repo to back up. Can be repeated. Defaults to core case data.",
    )
    args = parser.parse_args()

    summary = backup_case_data(
        backup_root=Path(args.backup_root).expanduser(),
        targets=args.targets or DEFAULT_TARGETS,
        keep=args.keep,
    )
    total_size = sum(entry.size_bytes for entry in summary.backed_up)
    print(
        f"CASE DATA BACKED UP: {len(summary.backed_up)} files, "
        f"{total_size / 1024 / 1024:.1f} MB"
    )
    if summary.missing:
        print(f"MISSING: {', '.join(summary.missing)}")
    print(f"destination: {summary.destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
