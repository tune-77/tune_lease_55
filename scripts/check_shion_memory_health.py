#!/usr/bin/env python3
"""紫苑記憶インデックスのヘルスチェック。

夜間パイプラインから呼ばれ、記憶インデックスの「無音の崩壊」を検知する:

- インデックスが存在しない / 読めない / レコード0件 → 異常（exit 1）
- 総レコード数が前回比で急減（既定: 30%超 または 100件超の減少）→ 異常（exit 1）
- 正常時は data/shion_memory_health_state.json に今回の件数を記録する

急減が意図的な場合（大規模な整理など）は --accept-current で現状を新しい
基準として受け入れる。異常時は基準を更新しないので、解消するまで毎晩警告が出る。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = REPO_ROOT / "data" / "shion_memory_index.json"
DEFAULT_STATE = REPO_ROOT / "data" / "shion_memory_health_state.json"


def load_index_summary(index_path: Path) -> dict | None:
    """インデックスから件数サマリを返す。読めない場合は None。"""
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    records = data.get("records")
    if not isinstance(records, list):
        return None
    by_type = Counter(str(r.get("memory_type") or "unknown") for r in records)
    by_status = Counter(str(r.get("status") or "active") for r in records)
    return {
        "total": len(records),
        "by_type": dict(sorted(by_type.items())),
        "by_status": dict(sorted(by_status.items())),
    }


def load_state(state_path: Path) -> dict:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


def save_state(state_path: Path, summary: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {"checked_at": datetime.now().isoformat(timespec="seconds"), **summary},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def check(
    summary: dict | None,
    previous: dict,
    *,
    max_drop_ratio: float = 0.3,
    max_drop_records: int = 100,
) -> tuple[bool, str]:
    """(健全か, メッセージ) を返す。"""
    if summary is None:
        return False, "記憶インデックスが存在しないか読めません"
    total = summary["total"]
    if total == 0:
        return False, "記憶インデックスのレコードが0件です"
    prev_total = int(previous.get("total") or 0)
    if prev_total > 0:
        drop = prev_total - total
        if drop > max_drop_records or (drop / prev_total) > max_drop_ratio:
            return False, (
                f"記憶レコードが急減しています: {prev_total} → {total} 件（-{drop}）。"
                " Vault/ソースの欠落を確認してください。意図的な削減なら"
                " --accept-current で新基準として受け入れられます"
            )
    return True, f"記憶インデックス健全: {total} 件（前回 {prev_total or '記録なし'}）"


def main() -> int:
    parser = argparse.ArgumentParser(description="紫苑記憶インデックスのヘルスチェック")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--max-drop-ratio", type=float, default=0.3)
    parser.add_argument("--max-drop-records", type=int, default=100)
    parser.add_argument(
        "--accept-current",
        action="store_true",
        help="現在の件数を新しい基準として受け入れる（意図的な大規模整理後に使用）",
    )
    args = parser.parse_args()

    summary = load_index_summary(args.index)
    previous = load_state(args.state)

    if args.accept_current:
        if summary is None:
            print("エラー: インデックスが読めないため基準を更新できません", file=sys.stderr)
            return 1
        save_state(args.state, summary)
        print(f"基準を更新しました: {summary['total']} 件")
        return 0

    healthy, message = check(
        summary,
        previous,
        max_drop_ratio=args.max_drop_ratio,
        max_drop_records=args.max_drop_records,
    )
    if healthy:
        assert summary is not None
        print(message)
        print(f"  種別内訳: {summary['by_type']}")
        print(f"  状態内訳: {summary['by_status']}")
        save_state(args.state, summary)
        return 0
    print(f"警告: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
