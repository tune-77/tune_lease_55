#!/usr/bin/env python3
"""二重台帳（リポジトリ台帳とランタイム台帳）の整合性チェック。

- リポジトリ台帳: scripts/improvement_ledger.jsonl（CI の ledger-sync が更新）
- ランタイム台帳: ~/Library/Logs/tunelease/ledger.jsonl（UI・パイプラインが更新）

書き手・読み手・キー形式が異なる二本立てのため、乖離が起きても気づけない
（実例: Weekly Log が recorded_at を読まず毎週0件になっていた）。
本スクリプトは直近 N 日の applied REV と状態件数を両台帳で突き合わせ、
乖離があれば警告を出力する。`--sync-repo-applied-to-runtime` 指定時は、
リポジトリ台帳で applied 確定済みの REV をランタイム台帳へ補完する。

使い方:
  python scripts/check_ledger_consistency.py --days 14
  python scripts/check_ledger_consistency.py --days 14 --sync-repo-applied-to-runtime
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_LEDGER = REPO_ROOT / "scripts" / "improvement_ledger.jsonl"
RUNTIME_LEDGER = Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl"


def load_entries(path: Path, since: dt.datetime | None = None) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_str = str(
                row.get("recorded_at")
                or row.get("timestamp")
                or row.get("updated_at")
                or row.get("created_at")
                or ""
            )
            try:
                ts = dt.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            if since is None or ts >= since:
                entries.append(row)
    except OSError:
        return []
    return entries


def load_recent_entries(path: Path, since: dt.datetime) -> list[dict]:
    return load_entries(path, since)


def _status(row: dict) -> str:
    return str(row.get("status") or "").strip().lower()


def rev_id_from_entry(row: dict) -> str:
    rev_id = str(row.get("rev_id") or "").strip()
    if rev_id.startswith("REV-"):
        return rev_id

    for field in ("title", "reason", "key", "canonical_key"):
        text = str(row.get(field) or "")
        match = re.search(r"\bREV-(\d{1,4})\b", text)
        if match:
            return f"REV-{int(match.group(1)):03d}"
    return ""


def applied_entries_by_rev(entries: list[dict]) -> dict[str, dict]:
    by_rev: dict[str, dict] = {}
    for row in entries:
        if _status(row) != "applied":
            continue
        rev_id = rev_id_from_entry(row)
        if rev_id:
            by_rev[rev_id] = row
    return by_rev


def applied_rev_ids(entries: list[dict]) -> set[str]:
    return set(applied_entries_by_rev(entries))


def status_counts(entries: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in entries:
        status = _status(row) or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


def runtime_sync_entry(repo_entry: dict, now: dt.datetime) -> dict:
    rev_id = rev_id_from_entry(repo_entry)
    title = str(repo_entry.get("title") or rev_id)
    canonical_key = str(repo_entry.get("canonical_key") or repo_entry.get("key") or rev_id.lower())
    return {
        "key": str(repo_entry.get("key") or canonical_key or rev_id),
        "rev_id": rev_id,
        "status": "applied",
        "title": title,
        "canonical_key": canonical_key,
        "pr_url": repo_entry.get("pr_url") or "",
        "reason": f"repo ledger applied sync: {repo_entry.get('reason') or 'リポジトリ台帳で applied 確定'}",
        "recorded_at": now.isoformat(),
    }


def append_runtime_entries(entries: list[dict]) -> None:
    if not entries:
        return
    RUNTIME_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME_LEDGER.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def repo_applied_missing_from_runtime(repo_entries: list[dict], runtime_entries_all: list[dict]) -> dict[str, dict]:
    repo_applied = applied_entries_by_rev(repo_entries)
    runtime_applied = applied_entries_by_rev(runtime_entries_all)
    return {
        rev_id: row
        for rev_id, row in repo_applied.items()
        if rev_id not in runtime_applied
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument(
        "--sync-repo-applied-to-runtime",
        action="store_true",
        help="Append repo-ledger applied REV entries missing from the runtime ledger.",
    )
    parser.add_argument(
        "--strict-runtime-only",
        action="store_true",
        help="Treat runtime-only applied REV entries as warnings. By default they are informational because manual triage writes only runtime ledger.",
    )
    args = parser.parse_args()

    since = dt.datetime.now() - dt.timedelta(days=args.days)
    repo_entries = load_recent_entries(REPO_LEDGER, since)
    runtime_entries = load_recent_entries(RUNTIME_LEDGER, since)
    runtime_entries_all = load_entries(RUNTIME_LEDGER)

    print(f"[ledger_consistency] 直近{args.days}日: リポジトリ台帳 {len(repo_entries)} 件 / ランタイム台帳 {len(runtime_entries)} 件")
    print(f"[ledger_consistency] リポジトリ側 状態件数: {status_counts(repo_entries)}")
    print(f"[ledger_consistency] ランタイム側 状態件数: {status_counts(runtime_entries)}")

    repo_applied = applied_rev_ids(repo_entries)
    runtime_applied_recent = applied_rev_ids(runtime_entries)
    runtime_applied_all = applied_rev_ids(runtime_entries_all)
    only_repo = sorted(repo_applied - runtime_applied_all)
    only_runtime = sorted(runtime_applied_recent - repo_applied)
    missing_repo_applied = repo_applied_missing_from_runtime(repo_entries, runtime_entries_all)

    warnings = 0
    if only_repo:
        if args.sync_repo_applied_to_runtime:
            now = dt.datetime.now()
            sync_entries = [runtime_sync_entry(missing_repo_applied[rev_id], now) for rev_id in only_repo if rev_id in missing_repo_applied]
            append_runtime_entries(sync_entries)
            print(f"[ledger_consistency] runtime台帳へ applied を補完: {len(sync_entries)} 件")
        else:
            warnings += 1
            print(f"[ledger_consistency] ⚠️ applied がリポジトリ台帳のみに存在: {', '.join(only_repo[:10])}")
    if only_runtime:
        if args.strict_runtime_only:
            warnings += 1
            print(f"[ledger_consistency] ⚠️ applied がランタイム台帳のみに存在: {', '.join(only_runtime[:10])}")
        else:
            print(f"[ledger_consistency] 情報: applied がランタイム台帳のみに存在（手動トリアージ等）: {', '.join(only_runtime[:10])}")
    if repo_entries and not runtime_entries and RUNTIME_LEDGER.exists():
        warnings += 1
        print("[ledger_consistency] ⚠️ ランタイム台帳に直近エントリがありません（時刻フィールドの読み違い or 書き込み停止の可能性）")
    if runtime_entries and not repo_entries and REPO_LEDGER.exists():
        warnings += 1
        print("[ledger_consistency] ⚠️ リポジトリ台帳に直近エントリがありません（ledger-sync 停止の可能性）")

    if warnings:
        print(f"[ledger_consistency] 乖離 {warnings} 件。台帳の突き合わせ確認を推奨します。")
    else:
        print("[ledger_consistency] 警告対象の乖離なし")
    # 監視用途のため常に0で終了する（パイプラインを止めない）。乖離は出力で伝える
    return 0


if __name__ == "__main__":
    sys.exit(main())
