#!/usr/bin/env python3
"""
scripts/cleanup_rag_unrated_rules.py

台帳（ledger_rules.json）に溜まる RAG-UNRATED 系通知（type=rag_boost_adjust）を
自動整理する。実行可能なルールが通知で埋没するのを防ぐ。

整理ルール:
  1. 同一ノート（path）への重複検出は最新の generated_at だけ残し、古い方をアーカイブ
  2. --max-age-days（既定30日）以上レビューされなかった通知をアーカイブ

アーカイブ先は ledger_rules_archive.json（削除はしない・後から参照可能）。
rag_boost_adjust 以外のルールには一切触れない。

使い方:
  python scripts/cleanup_rag_unrated_rules.py --dry-run
  python scripts/cleanup_rag_unrated_rules.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO_ROOT / "api" / "rule_engine" / "ledger_rules.json"
ARCHIVE_PATH = REPO_ROOT / "api" / "rule_engine" / "ledger_rules_archive.json"

TARGET_TYPE = "rag_boost_adjust"


def _parse_dt(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def plan_cleanup(
    rules: list[dict],
    *,
    now: datetime,
    max_age_days: int,
) -> tuple[list[dict], list[dict]]:
    """整理計画を立てる。(残すルール, アーカイブするルール) を返す。

    アーカイブ対象には parked_reason / parked_at が付与される。
    rag_boost_adjust かつ pending_review のもの以外は無条件で残す。
    """
    cutoff = now - timedelta(days=max(1, max_age_days))

    # path ごとの最新 generated_at を求める（重複集約用）
    latest_by_path: dict[str, datetime] = {}
    for rule in rules:
        if rule.get("type") != TARGET_TYPE or not rule.get("pending_review"):
            continue
        path = str(rule.get("path") or "")
        dt = _parse_dt(rule.get("generated_at"))
        if not path or dt is None:
            continue
        if path not in latest_by_path or dt > latest_by_path[path]:
            latest_by_path[path] = dt

    kept: list[dict] = []
    parked: list[dict] = []
    for rule in rules:
        if rule.get("type") != TARGET_TYPE or not rule.get("pending_review"):
            kept.append(rule)
            continue

        path = str(rule.get("path") or "")
        dt = _parse_dt(rule.get("generated_at"))
        reason = ""
        if path and dt is not None and dt < latest_by_path.get(path, dt):
            reason = "同一ノートのより新しい検出があるため集約"
        elif dt is not None and dt < cutoff:
            reason = f"{max_age_days}日以上レビューされなかったため自動保留（weak penalty通知）"

        if reason:
            archived = dict(rule)
            archived["parked_reason"] = reason
            archived["parked_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            archived["parked_by"] = "cleanup_rag_unrated_rules"
            parked.append(archived)
        else:
            kept.append(rule)
    return kept, parked


def load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_json_list(path: Path, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG-UNRATED 通知の自動整理")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="実行せずに結果を表示")
    mode.add_argument("--apply", action="store_true", help="実際に整理する")
    parser.add_argument("--max-age-days", type=int, default=30, help="この日数を超えた通知を保留（既定30）")
    args = parser.parse_args()

    if not LEDGER_PATH.exists():
        print(f"❌ 台帳ファイルが見つかりません: {LEDGER_PATH}", file=sys.stderr)
        return 1

    rules = load_json_list(LEDGER_PATH)
    now = datetime.now(timezone.utc)
    kept, parked = plan_cleanup(rules, now=now, max_age_days=args.max_age_days)

    label = "DRY-RUN" if args.dry_run else "APPLY"
    print(f"🧹 cleanup_rag_unrated_rules — {label}")
    print(f"   台帳: {len(rules)} 件 → 残す {len(kept)} 件 / アーカイブ {len(parked)} 件")
    for rule in parked:
        print(f"   📦 {rule.get('rev_id')}: {rule.get('parked_reason')}")

    if args.apply and parked:
        archive = load_json_list(ARCHIVE_PATH)
        archive.extend(parked)
        save_json_list(ARCHIVE_PATH, archive)
        save_json_list(LEDGER_PATH, kept)
        print(f"   ✅ アーカイブへ移動しました: {ARCHIVE_PATH}")
    elif args.apply:
        print("   変更はありません")
    return 0


if __name__ == "__main__":
    sys.exit(main())
