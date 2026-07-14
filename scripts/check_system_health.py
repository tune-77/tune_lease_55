#!/usr/bin/env python3
"""Weekly local health checks for the tune_lease_55 automation stack."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAULT = Path(
    os.environ.get(
        "OBSIDIAN_VAULT",
        Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents" / "Obsidian Vault",
    )
)
DEFAULT_BACKUP_ROOT = Path(
    os.environ.get(
        "OBSIDIAN_BACKUP_ROOT",
        Path.home()
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs"
        / "tune_lease_55_backups"
        / "obsidian",
    )
)
DEFAULT_CASE_BACKUP_ROOT = Path(
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
DEFAULT_REPORT_PATH = REPO_ROOT / "logs" / "weekly_health_report.json"


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str


def _now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def _age_hours(path: Path) -> float:
    modified = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=_now().tzinfo)
    return (_now() - modified).total_seconds() / 3600


REINDEX_DONE_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}).*\[reindex\] 完了.*total_in_db=(\d+)")
MAINTENANCE_DONE_RE = re.compile(
    r"実行時刻:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?ChromaDB:\s*success",
    re.DOTALL,
)
MAINTENANCE_DB_COUNT_RE = re.compile(r"LocalVectorDB:\s*(\d+)\s*件")


def check_reindex_log(max_age_hours: int) -> CheckResult:
    log_path = Path.home() / "Library" / "Logs" / "tune_lease_55_obsidian_reindex.out.log"
    if not log_path.exists():
        return CheckResult("obsidian_reindex", False, f"missing log: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed: list[tuple[dt.datetime, int | None]] = []
    for ts_text, total_text in REINDEX_DONE_RE.findall(text):
        parsed.append((dt.datetime.fromisoformat(ts_text).replace(tzinfo=_now().tzinfo), int(total_text)))
    db_counts = MAINTENANCE_DB_COUNT_RE.findall(text)
    maintenance_total = int(db_counts[-1]) if db_counts else None
    for ts_text in MAINTENANCE_DONE_RE.findall(text):
        parsed.append((dt.datetime.strptime(ts_text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_now().tzinfo), maintenance_total))
    if not parsed:
        return CheckResult("obsidian_reindex", False, "no successful completion line found")

    completed_at, total_raw = max(parsed, key=lambda item: item[0])
    total = total_raw if total_raw is not None else 1
    age = (_now() - completed_at).total_seconds() / 3600
    if age > max_age_hours:
        return CheckResult("obsidian_reindex", False, f"last success is stale: {age:.1f}h ago, total_in_db={total}")
    if total <= 0:
        return CheckResult("obsidian_reindex", False, f"last success has invalid total_in_db={total}")
    return CheckResult("obsidian_reindex", True, f"last success {age:.1f}h ago, total_in_db={total}")


def check_backup_snapshot(backup_root: Path, max_age_hours: int) -> CheckResult:
    if not backup_root.exists():
        return CheckResult("obsidian_backup", False, f"missing backup root: {backup_root}")

    snapshots = sorted(backup_root.glob("Obsidian Vault_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not snapshots:
        return CheckResult("obsidian_backup", False, f"no snapshots under {backup_root}")

    latest = snapshots[0]
    age = _age_hours(latest)
    manifest = latest / "backup_manifest.json"
    if age > max_age_hours:
        return CheckResult("obsidian_backup", False, f"latest snapshot is stale: {age:.1f}h ago at {latest}")
    if not manifest.exists():
        return CheckResult("obsidian_backup", False, f"latest snapshot missing manifest: {latest}")
    return CheckResult("obsidian_backup", True, f"latest snapshot {age:.1f}h ago: {latest.name}")


def check_case_data_backup(backup_root: Path, max_age_hours: int) -> CheckResult:
    if not backup_root.exists():
        return CheckResult("case_data_backup", False, f"missing backup root: {backup_root}")

    snapshots = sorted(backup_root.glob("case_data_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not snapshots:
        return CheckResult("case_data_backup", False, f"no snapshots under {backup_root}")

    latest = snapshots[0]
    age = _age_hours(latest)
    manifest = latest / "backup_manifest.json"
    if age > max_age_hours:
        return CheckResult("case_data_backup", False, f"latest snapshot is stale: {age:.1f}h ago at {latest}")
    if not manifest.exists():
        return CheckResult("case_data_backup", False, f"latest snapshot missing manifest: {latest}")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CheckResult("case_data_backup", False, f"manifest is invalid JSON: {exc}")
    files = len(data.get("backed_up") or [])
    if files <= 0:
        return CheckResult("case_data_backup", False, f"manifest has no backed_up files: {latest}")
    return CheckResult("case_data_backup", True, f"latest snapshot {age:.1f}h ago: {latest.name}, files={files}")


def check_news_note(vault: Path, max_age_hours: int) -> CheckResult:
    news_dir = vault / "Projects" / "tune_lease_55" / "News"
    if not news_dir.exists():
        return CheckResult("lease_news", False, f"missing news directory: {news_dir}")

    notes = sorted(
        list(news_dir.glob("*_industry-risk-news-focus.md")) + list(news_dir.glob("*_lease-news.md")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not notes:
        return CheckResult("lease_news", False, f"no industry risk news notes under {news_dir}")

    latest = notes[0]
    age = _age_hours(latest)
    if age > max_age_hours:
        return CheckResult("lease_news", False, f"latest news note is stale: {age:.1f}h ago at {latest.name}")
    if latest.stat().st_size <= 0:
        return CheckResult("lease_news", False, f"latest news note is empty: {latest.name}")
    return CheckResult("lease_news", True, f"latest note {age:.1f}h ago: {latest.name}")


def check_chroma_db(max_age_hours: int) -> CheckResult:
    db_path = REPO_ROOT / "api" / "chroma_db" / "chroma.sqlite3"
    if not db_path.exists():
        return CheckResult("chroma_db", False, f"missing Chroma sqlite: {db_path}")
    if db_path.stat().st_size <= 0:
        return CheckResult("chroma_db", False, "Chroma sqlite is empty")

    age = _age_hours(db_path)
    if age > max_age_hours:
        return CheckResult("chroma_db", False, f"Chroma sqlite is stale: {age:.1f}h ago")
    return CheckResult("chroma_db", True, f"Chroma sqlite updated {age:.1f}h ago")


def check_sqlite_db(path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(f"sqlite:{path.name}", False, f"missing sqlite db: {path}")
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute("PRAGMA integrity_check;").fetchone()
    except sqlite3.Error as exc:
        return CheckResult(f"sqlite:{path.name}", False, f"integrity check failed to run: {exc}")

    status = row[0] if row else ""
    if status != "ok":
        return CheckResult(f"sqlite:{path.name}", False, f"integrity_check={status!r}")
    return CheckResult(f"sqlite:{path.name}", True, "integrity_check=ok")


def run_checks(args: argparse.Namespace) -> list[CheckResult]:
    return [
        check_reindex_log(args.reindex_max_age_hours),
        check_backup_snapshot(Path(args.backup_root).expanduser(), args.backup_max_age_hours),
        check_case_data_backup(Path(args.case_backup_root).expanduser(), args.case_backup_max_age_hours),
        check_news_note(Path(args.vault).expanduser(), args.news_max_age_hours),
        check_chroma_db(args.chroma_max_age_hours),
        check_sqlite_db(REPO_ROOT / "data" / "lease_data.db"),
        check_sqlite_db(REPO_ROOT / "data" / "screening_db.sqlite"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local health checks for tune_lease_55 automation.")
    parser.add_argument("--vault", default=str(DEFAULT_VAULT), help="Obsidian vault path.")
    parser.add_argument("--backup-root", default=str(DEFAULT_BACKUP_ROOT), help="Obsidian backup root path.")
    parser.add_argument("--case-backup-root", default=str(DEFAULT_CASE_BACKUP_ROOT), help="Case data backup root path.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="JSON report output path.")
    parser.add_argument("--reindex-max-age-hours", type=int, default=36)
    parser.add_argument("--backup-max-age-hours", type=int, default=8 * 24)
    parser.add_argument("--case-backup-max-age-hours", type=int, default=8 * 24)
    parser.add_argument("--news-max-age-hours", type=int, default=72)
    parser.add_argument("--chroma-max-age-hours", type=int, default=36)
    args = parser.parse_args()

    results = run_checks(args)
    ok = all(result.ok for result in results)
    report = {
        "checked_at": _now().isoformat(timespec="seconds"),
        "ok": ok,
        "results": [asdict(result) for result in results],
    }

    report_path = Path(args.report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    status = "OK" if ok else "FAIL"
    print(f"HEALTH {status}: {report['checked_at']}")
    for result in results:
        mark = "OK" if result.ok else "FAIL"
        print(f"- {mark} {result.name}: {result.message}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
