#!/usr/bin/env python3
"""Record Codex auto-improvement execution status without marking items applied."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


VALID_STATUSES = {
    "blocked_by_quota",
    "pending",
    "running",
    "failed",
    "completed_pending_review",
    "merged",
    "rejected",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"items": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {"items": {}}
    if not isinstance(data.get("items"), dict):
        data["items"] = {}
    return data


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rev_id", help="Improvement id, for example REV-059.")
    parser.add_argument("--status", required=True, choices=sorted(VALID_STATUSES))
    parser.add_argument("--detail", default="")
    parser.add_argument("--source", default="manual")
    parser.add_argument("--status-file", type=Path, default=root / "reports" / "codex_auto_execution_status.json")
    args = parser.parse_args()

    rev_id = args.rev_id.upper()
    data = load_json(args.status_file)
    now = dt.datetime.now().isoformat(timespec="seconds")
    previous = data["items"].get(rev_id, {})
    attempts = int(previous.get("attempts") or 0)
    if args.status in {"blocked_by_quota", "failed", "running"}:
        attempts += 1

    data["items"][rev_id] = {
        "id": rev_id,
        "status": args.status,
        "detail": args.detail,
        "source": args.source,
        "attempts": attempts,
        "created_at": previous.get("created_at") or now,
        "updated_at": now,
    }
    data["updated_at"] = now
    dump_json(args.status_file, data)
    print(f"{rev_id}: {args.status} ({args.status_file})")


if __name__ == "__main__":
    main()
