"""紫苑記憶索引の鮮度更新（docs/shion_memory_architecture.md Next Step 2 の実装）。

使用ログ（data/shion_memory_usage_log.jsonl、`build_recall_prompt_block` が追記）
から各記憶の last_used_at を求め、長期間使われていない記憶を `stale` に落とす。
使用ログが真実の源なので、索引を再生成しても本スクリプトの再実行で状態を再現できる。

ルール:
- last_used_at = 使用ログ上の最新利用日。
- active かつ作成から --stale-days 超、かつ直近 --stale-days 以内の利用が無い → stale。
- stale でも直近利用があれば active に戻す（鮮度確認済みとみなす）。
- value_memory（Mana・良心などの上位規範）は経年で stale に落とさない。
- 削除はしない（アーキテクチャ方針: 古い記憶は revised / deprecated / stale へ）。

使い方:
    python3 scripts/update_shion_memory_freshness.py            # data/ の索引を更新
    python3 scripts/update_shion_memory_freshness.py --dry-run  # 変更内容の確認のみ
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INDEX = REPO_ROOT / "data" / "shion_memory_index.json"
DEFAULT_USAGE_LOG = REPO_ROOT / "data" / "shion_memory_usage_log.jsonl"
DEFAULT_STALE_DAYS = 45

# 上位規範は経年で鮮度切れ扱いにしない
_NEVER_STALE_TYPES = {"value_memory"}


def load_usage_dates(path: Path) -> dict[str, str]:
    """使用ログを読み、記憶ID → 最新利用日(YYYY-MM-DD) を返す。壊れた行は無視。"""
    latest: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        used_on = str(entry.get("ts") or "")[:10]
        if not used_on:
            continue
        for ref in entry.get("refs") or []:
            rid = str(ref)
            if rid and used_on > latest.get(rid, ""):
                latest[rid] = used_on
    return latest


def _parse_date(value: str) -> date | None:
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def apply_freshness(
    index: dict[str, Any],
    usage_dates: dict[str, str],
    *,
    stale_days: int = DEFAULT_STALE_DAYS,
    today: date | None = None,
) -> dict[str, int]:
    """索引レコードへ last_used_at 反映と stale 昇降格を行い、変更件数を返す。"""
    today = today or date.today()
    cutoff = today - timedelta(days=stale_days)
    summary = {"last_used_updated": 0, "demoted_to_stale": 0, "revived_to_active": 0}

    for record in index.get("records") or []:
        if not isinstance(record, dict):
            continue
        rid = str(record.get("id") or "")
        status = str(record.get("status") or "active")
        memory_type = str(record.get("memory_type") or "")

        used_on = usage_dates.get(rid, "")
        if used_on and used_on != str(record.get("last_used_at") or ""):
            record["last_used_at"] = used_on
            summary["last_used_updated"] += 1

        if status not in {"active", "stale"}:
            continue  # revised / deprecated / private は鮮度で動かさない

        last_used = _parse_date(str(record.get("last_used_at") or ""))
        created = _parse_date(str(record.get("created_at") or ""))
        recently_used = last_used is not None and last_used >= cutoff

        if status == "stale" and recently_used:
            record["status"] = "active"
            summary["revived_to_active"] += 1
        elif (
            status == "active"
            and memory_type not in _NEVER_STALE_TYPES
            and not recently_used
            and created is not None
            and created < cutoff
        ):
            record["status"] = "stale"
            summary["demoted_to_stale"] += 1

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="紫苑記憶索引の last_used_at 更新と stale 降格")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--usage-log", type=Path, default=DEFAULT_USAGE_LOG)
    parser.add_argument("--stale-days", type=int, default=DEFAULT_STALE_DAYS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        index = json.loads(args.index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"索引を読めません: {args.index} ({exc})")
        return 1

    usage_dates = load_usage_dates(args.usage_log)
    summary = apply_freshness(index, usage_dates, stale_days=args.stale_days)

    print(f"usage_log_refs={len(usage_dates)}")
    for key, count in summary.items():
        print(f"{key}={count}")

    if args.dry_run:
        print("dry-run: 索引は書き換えていません")
        return 0

    text = json.dumps(index, ensure_ascii=False, indent=2)
    tmp = args.index.with_suffix(args.index.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(args.index)
    print(f"wrote={args.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
