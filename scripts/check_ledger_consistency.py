#!/usr/bin/env python3
"""二重台帳（リポジトリ台帳とランタイム台帳）の整合性チェック。

- リポジトリ台帳: scripts/improvement_ledger.jsonl（CI の ledger-sync が更新）
- ランタイム台帳: ~/Library/Logs/tunelease/ledger.jsonl（UI・パイプラインが更新）

書き手・読み手・キー形式が異なる二本立てのため、乖離が起きても気づけない
（実例: Weekly Log が recorded_at を読まず毎週0件になっていた）。
本スクリプトは直近 N 日の applied REV と状態件数を両台帳で突き合わせ、
乖離があれば警告を出力する（読み取り専用・自動修正なし）。

使い方:
  python scripts/check_ledger_consistency.py --days 14
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_LEDGER = REPO_ROOT / "scripts" / "improvement_ledger.jsonl"
RUNTIME_LEDGER = Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl"


def load_recent_entries(path: Path, since: dt.datetime) -> list[dict]:
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
            if ts >= since:
                entries.append(row)
    except OSError:
        return []
    return entries


def applied_rev_ids(entries: list[dict]) -> set[str]:
    revs: set[str] = set()
    for row in entries:
        if str(row.get("status") or "").lower() != "applied":
            continue
        rev_id = str(row.get("rev_id") or "")
        if not rev_id:
            title = str(row.get("title") or "")
            if title.startswith("REV-"):
                rev_id = title.split()[0].rstrip(":")
        if rev_id:
            revs.add(rev_id)
    return revs


def status_counts(entries: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in entries:
        status = str(row.get("status") or "unknown").lower()
        counts[status] = counts.get(status, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14)
    args = parser.parse_args()

    since = dt.datetime.now() - dt.timedelta(days=args.days)
    repo_entries = load_recent_entries(REPO_LEDGER, since)
    runtime_entries = load_recent_entries(RUNTIME_LEDGER, since)

    print(f"[ledger_consistency] 直近{args.days}日: リポジトリ台帳 {len(repo_entries)} 件 / ランタイム台帳 {len(runtime_entries)} 件")
    print(f"[ledger_consistency] リポジトリ側 状態件数: {status_counts(repo_entries)}")
    print(f"[ledger_consistency] ランタイム側 状態件数: {status_counts(runtime_entries)}")

    repo_applied = applied_rev_ids(repo_entries)
    runtime_applied = applied_rev_ids(runtime_entries)
    only_repo = sorted(repo_applied - runtime_applied)
    only_runtime = sorted(runtime_applied - repo_applied)

    warnings = 0
    if only_repo:
        warnings += 1
        print(f"[ledger_consistency] ⚠️ applied がリポジトリ台帳のみに存在: {', '.join(only_repo[:10])}")
    if only_runtime:
        warnings += 1
        print(f"[ledger_consistency] ⚠️ applied がランタイム台帳のみに存在: {', '.join(only_runtime[:10])}")
    if repo_entries and not runtime_entries and RUNTIME_LEDGER.exists():
        warnings += 1
        print("[ledger_consistency] ⚠️ ランタイム台帳に直近エントリがありません（時刻フィールドの読み違い or 書き込み停止の可能性）")
    if runtime_entries and not repo_entries and REPO_LEDGER.exists():
        warnings += 1
        print("[ledger_consistency] ⚠️ リポジトリ台帳に直近エントリがありません（ledger-sync 停止の可能性）")

    if warnings:
        print(f"[ledger_consistency] 乖離 {warnings} 件。台帳の突き合わせ確認を推奨します。")
    else:
        print("[ledger_consistency] 乖離なし")
    # 監視用途のため常に0で終了する（パイプラインを止めない）。乖離は出力で伝える
    return 0


if __name__ == "__main__":
    sys.exit(main())
