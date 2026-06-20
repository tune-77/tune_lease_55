#!/usr/bin/env python3
"""Sync merged/closed Codex PR statuses back to codex_auto_execution_status.json.

Uses `gh pr list` to find PRs created by Codex auto-improvement and writes
"merged" or "rejected" back to the status file so build_codex_auto_queue
can skip already-resolved items.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


CODEX_PR_LABEL = "codex-auto"
REV_PATTERN = re.compile(r"REV-\d+", re.IGNORECASE)


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


def gh_pr_list(state: str) -> list[dict[str, Any]]:
    """Run gh pr list and return parsed JSON rows."""
    try:
        proc = subprocess.run(
            [
                "gh", "pr", "list",
                "--state", state,
                "--json", "number,title,mergedAt,closedAt,labels,author",
                "--limit", "100",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            print(f"gh pr list --state {state} failed: {proc.stderr.strip()}", flush=True)
            return []
        return json.loads(proc.stdout or "[]")
    except Exception as exc:
        print(f"gh pr list error: {exc}", flush=True)
        return []


def extract_rev_ids(title: str) -> list[str]:
    return [m.upper() for m in REV_PATTERN.findall(title)]


def is_codex_pr(pr: dict[str, Any]) -> bool:
    """Codex auto queue が作成した PR かどうかを判定する."""
    labels = [lb.get("name", "") for lb in (pr.get("labels") or [])]
    if CODEX_PR_LABEL in labels:
        return True
    title = str(pr.get("title") or "")
    # REV-NNN を含む PR のうち、codex-auto ラベルがなくても
    # execute_codex_queue 由来とみなせるものを拾う
    return bool(REV_PATTERN.search(title))


def main() -> None:
    root = repo_root()
    status_file = root / "reports" / "codex_auto_execution_status.json"
    data = load_json(status_file)
    now = dt.datetime.now().isoformat(timespec="seconds")

    updated = 0

    # merged PRs
    for pr in gh_pr_list("merged"):
        if not is_codex_pr(pr):
            continue
        for rev_id in extract_rev_ids(str(pr.get("title") or "")):
            existing = data["items"].get(rev_id, {})
            if existing.get("status") in ("merged",):
                continue
            data["items"][rev_id] = {
                **existing,
                "id": rev_id,
                "status": "merged",
                "detail": f"PR #{pr.get('number')} merged at {pr.get('mergedAt', '')}",
                "source": "sync_codex_pr_status",
                "attempts": existing.get("attempts", 0),
                "created_at": existing.get("created_at") or now,
                "updated_at": now,
            }
            print(f"  {rev_id}: merged (PR #{pr.get('number')})", flush=True)
            updated += 1

    # closed (not merged) = rejected
    for pr in gh_pr_list("closed"):
        if not is_codex_pr(pr):
            continue
        if pr.get("mergedAt"):
            continue  # merged は上で処理済み
        for rev_id in extract_rev_ids(str(pr.get("title") or "")):
            existing = data["items"].get(rev_id, {})
            if existing.get("status") in ("merged", "rejected"):
                continue
            data["items"][rev_id] = {
                **existing,
                "id": rev_id,
                "status": "rejected",
                "detail": f"PR #{pr.get('number')} closed without merge at {pr.get('closedAt', '')}",
                "source": "sync_codex_pr_status",
                "attempts": existing.get("attempts", 0),
                "created_at": existing.get("created_at") or now,
                "updated_at": now,
            }
            print(f"  {rev_id}: rejected (PR #{pr.get('number')})", flush=True)
            updated += 1

    if updated:
        data["updated_at"] = now
        dump_json(status_file, data)
        print(f"\n{updated} 件のステータスを同期しました: {status_file}", flush=True)
    else:
        print("同期対象の PR なし。", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
