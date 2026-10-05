#!/usr/bin/env python3
"""Backup an iCloud 上の Obsidian Vault to timestamped snapshot folders.

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
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runtime_paths import (  # noqa: E402
    DEFAULT_OBSIDIAN_VAULT,
    ICLOUD_OBSIDIAN_DOCS,
    LEGACY_OBSIDIAN_VAULT,
    OBSIDIAN_VAULT_ENV_VARS,
)


def _env_vault_candidates() -> list[Path]:
    """env 由来の候補を runtime_paths と同じ優先順で返す。

    以前は OBSIDIAN_VAULT しか読んでおらず、OBSIDIAN_VAULT_PATH だけが設定された
    launchd ジョブ（obsidian-reindex 等）と別の Vault をバックアップしうる状態だった。
    """
    out: list[Path] = []
    for name in OBSIDIAN_VAULT_ENV_VARS:
        raw = (os.environ.get(name) or "").strip()
        if raw:
            out.append(Path(raw).expanduser())
    return out


DEFAULT_VAULT_CANDIDATES = [
    *_env_vault_candidates(),
    DEFAULT_OBSIDIAN_VAULT,
    LEGACY_OBSIDIAN_VAULT,
    # Vault そのものが見つからないときの保険として親ディレクトリを走査する。
    ICLOUD_OBSIDIAN_DOCS,
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
    failed: list[dict[str, str]] = field(default_factory=list)
    suspicious_drop: bool = False


# iCloud で「最適化」されて実体がローカルに無いファイル（dataless）。
# launchd 配下ではコピー時に実体化できず EDEADLK（Errno 11）で失敗する。
_SF_DATALESS = getattr(stat, "SF_DATALESS", 0x40000000)
DATALESS_WAIT_SECONDS = 60


def _is_dataless(path: Path) -> bool:
    try:
        return bool(getattr(path.stat(), "st_flags", 0) & _SF_DATALESS)
    except OSError:
        return False


def _materialize_dataless(files: list[Path], wait_seconds: float = DATALESS_WAIT_SECONDS) -> None:
    """dataless ファイルを brctl download で取得し、最大 wait_seconds 待つ（取れなくても続行）。"""
    pending = [p for p in files if _is_dataless(p)]
    if not pending or not shutil.which("brctl"):
        return
    for path in pending:
        try:
            subprocess.run(["brctl", "download", str(path)], capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            pass
    deadline = time.monotonic() + wait_seconds
    while pending and time.monotonic() < deadline:
        time.sleep(2)
        pending = [p for p in pending if _is_dataless(p)]


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
        raise FileNotFoundError(f"iCloud 上の Obsidian Vault が見つかりません: {path}")

    candidates = _candidate_vaults()
    if not candidates:
        raise FileNotFoundError(
            "iCloud 上の Obsidian Vault が見つかりません。"
            f"{OBSIDIAN_VAULT_ENV_VARS[0]} を設定するか --vault を指定してください。"
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


def _cleanup_old_snapshots(
    root: Path,
    vault_name: str,
    keep: int,
    *,
    protected: Path | None = None,
) -> list[Path]:
    if keep <= 0 or not root.exists():
        return []
    prefix = f"{vault_name}_"
    snapshots = [
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    ]
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    retained = set(snapshots[:keep])
    if protected is not None:
        protected_resolved = protected.resolve()
        if any(snapshot.resolve() == protected_resolved for snapshot in snapshots):
            retained.add(next(snapshot for snapshot in snapshots if snapshot.resolve() == protected_resolved))
    removed: list[Path] = []
    for old in snapshots:
        if old in retained:
            continue
        shutil.rmtree(old, ignore_errors=True)
        removed.append(old)
    return removed


def _latest_complete_manifest(root: Path, vault_name: str) -> dict[str, object] | None:
    """直近の完全バックアップmanifestを返す。壊れたmanifestは無視する。"""
    if not root.exists():
        return None
    manifests = sorted(
        root.glob(f"{vault_name}_*/backup_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in manifests:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "complete" and isinstance(payload.get("file_count"), int):
            payload["_snapshot_path"] = str(path.parent)
            return payload
    return None


def _is_suspicious_drop(current_count: int, previous_manifest: dict[str, object] | None) -> bool:
    if not previous_manifest:
        return False
    previous_count = int(previous_manifest["file_count"])
    drop = previous_count - current_count
    if drop <= 0 or previous_count <= 0:
        return False
    return drop >= 50 or (drop / previous_count) >= 0.10


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
    previous_manifest = _latest_complete_manifest(backup_root, vault.name)
    suspicious_drop = _is_suspicious_drop(len(files), previous_manifest)

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
            suspicious_drop=suspicious_drop,
        )

    backup_root.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=False)

    _materialize_dataless(files)
    # 1ファイルの失敗（iCloud 未ダウンロード等）でスナップショット全体を止めない。
    failed: list[dict[str, str]] = []
    for src in files:
        rel = src.relative_to(vault)
        target = dest / rel
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        except OSError as exc:
            target.unlink(missing_ok=True)
            reason = "icloud_dataless" if _is_dataless(src) else "copy_error"
            failed.append({"path": rel.as_posix(), "reason": reason, "error": str(exc)[:200]})

    manifest = {
        "vault": str(vault),
        "destination": str(dest),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "partial" if failed else ("suspicious" if suspicious_drop else "complete"),
        "file_count": len(files),
        "copied_count": len(files) - len(failed),
        "failed_count": len(failed),
        "failed": failed,
        "total_bytes": total_bytes,
        "excluded_count": excluded_count,
        "excludes": excludes,
        "previous_complete_file_count": previous_manifest.get("file_count") if previous_manifest else None,
        "suspicious_drop": suspicious_drop,
    }
    (dest / "backup_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 異常世代も上限内で循環させつつ、直近の正常な復旧点だけは固定して残す。
    protected = None
    if (failed or suspicious_drop) and previous_manifest:
        protected = Path(str(previous_manifest["_snapshot_path"]))
    _cleanup_old_snapshots(backup_root, vault.name, keep, protected=protected)
    return BackupSummary(
        vault=vault,
        destination=dest,
        file_count=len(files),
        total_bytes=total_bytes,
        excluded_count=excluded_count,
        dry_run=False,
        failed=failed,
        suspicious_drop=suspicious_drop,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a timestamped backup of an iCloud 上の Obsidian Vault.")
    parser.add_argument("--vault", default=None, help="iCloud 上の Obsidian Vault パス（省略時は自動検出）")
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
    if summary.failed:
        print(
            f"WARN: {len(summary.failed)} files could not be copied (see backup_manifest.json): "
            + ", ".join(item["path"] for item in summary.failed[:5]),
            file=sys.stderr,
        )
    if summary.suspicious_drop:
        print(
            "ALERT: Vault file count dropped sharply; older complete backups were preserved.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
