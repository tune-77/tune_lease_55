#!/usr/bin/env python3
"""Move inactive Chroma HNSW segment directories out of api/chroma_db.

The script reads active VECTOR segment ids from chroma.sqlite3 and moves every
other top-level UUID directory to /private/tmp by default. It does not delete
data; the manifest records enough information to restore manually if needed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = REPO_ROOT / "api" / "chroma_db"
REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_QUARANTINE_ROOT = Path("/private/tmp/tune_lease_55_chroma_orphans")


def _active_vector_segments(chroma_dir: Path) -> set[str]:
    sqlite_path = chroma_dir / "chroma.sqlite3"
    with sqlite3.connect(sqlite_path) as conn:
        return {
            row[0]
            for row in conn.execute(
                "select id from segments where scope = 'VECTOR'"
            ).fetchall()
        }


def _size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def find_orphans(chroma_dir: Path = CHROMA_DIR) -> dict:
    active = _active_vector_segments(chroma_dir)
    directories = [p for p in chroma_dir.iterdir() if p.is_dir()]
    orphans = [p for p in directories if p.name not in active]
    return {
        "active_vector_segments": sorted(active),
        "directory_count": len(directories),
        "orphans": [
            {
                "name": p.name,
                "path": str(p),
                "size_bytes": _size_bytes(p),
            }
            for p in sorted(orphans)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Quarantine inactive Chroma segment directories.")
    parser.add_argument("--apply", action="store_true", help="move orphan directories")
    parser.add_argument("--quarantine-root", type=Path, default=DEFAULT_QUARANTINE_ROOT)
    args = parser.parse_args()

    scan = find_orphans()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine_dir = args.quarantine_root / timestamp
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "apply" if args.apply else "dry_run",
        "chroma_dir": str(CHROMA_DIR),
        "quarantine_dir": str(quarantine_dir),
        "active_vector_segments": scan["active_vector_segments"],
        "directory_count": scan["directory_count"],
        "orphan_count": len(scan["orphans"]),
        "orphan_bytes": sum(item["size_bytes"] for item in scan["orphans"]),
        "moved": [],
        "orphans": scan["orphans"],
    }

    if args.apply and scan["orphans"]:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for item in scan["orphans"]:
            src = Path(item["path"])
            dst = quarantine_dir / src.name
            if dst.exists():
                raise FileExistsError(f"quarantine destination exists: {dst}")
            shutil.move(str(src), str(dst))
            manifest["moved"].append({"from": str(src), "to": str(dst)})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "chroma_orphan_cleanup_latest.json"
    history_path = REPORTS_DIR / f"chroma_orphan_cleanup_{timestamp}.json"
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    report_path.write_text(text, encoding="utf-8")
    history_path.write_text(text, encoding="utf-8")

    print(f"mode={manifest['mode']}")
    print(f"orphan_count={manifest['orphan_count']}")
    print(f"orphan_mib={manifest['orphan_bytes'] / 1024 / 1024:.2f}")
    print(f"quarantine_dir={quarantine_dir}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
