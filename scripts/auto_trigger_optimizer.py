#!/usr/bin/env python3
"""スコアリング重み最適化の自動トリガースクリプト。

final_status が未設定かつ created_at（timestamp）が30日以上前のケースを
「失注」として自動補完し、auto_optimizer.run_auto_optimization() を呼び出す。

data/lease_data.db を直接更新するため、コミット禁止対象。
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_ts(ts_str: str | None) -> dt.datetime | None:
    """ISO形式タイムスタンプ文字列を datetime に変換する。"""
    if not ts_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(str(ts_str)[:26], fmt)
        except ValueError:
            continue
    try:
        return dt.datetime.fromisoformat(str(ts_str)[:26])
    except ValueError:
        return None


def auto_complete_old_cases(cutoff_days: int = 30) -> tuple[list[str], bool]:
    """final_status が空かつ cutoff_days 以上前のケースを「失注」にする。

    Returns:
        (更新したケース ID のリスト, インポートに成功したか).
        インポート失敗を「対象0件」と区別できないと、依存モジュール破損時も
        無条件成功のまま自動補完が無音停止する。
    """
    sys.path.insert(0, str(repo_root()))
    try:
        from data_cases import load_all_cases, update_case
    except ImportError as exc:
        print(f"[auto_trigger_optimizer] data_cases インポート失敗: {exc}", file=sys.stderr, flush=True)
        return [], False

    cutoff = dt.datetime.now() - dt.timedelta(days=cutoff_days)
    updated_ids: list[str] = []

    cases = load_all_cases()
    for case in cases:
        final_status = str(case.get("final_status") or "").strip()
        if final_status and final_status != "未登録":
            continue  # 既に登録済み

        ts_str = case.get("timestamp") or case.get("created_at") or case.get("registration_date")
        ts = _parse_ts(ts_str)
        if ts is None or ts >= cutoff:
            continue

        case_id = str(case.get("id") or "")
        if not case_id:
            continue

        if update_case(case_id, {"final_status": "失注"}):
            updated_ids.append(case_id)
            print(
                f"  [auto_trigger_optimizer] {case_id}: 失注に自動補完 (timestamp={ts_str})",
                flush=True,
            )

    return updated_ids, True


def run_optimizer() -> tuple[dict[str, Any] | None, bool]:
    """auto_optimizer.run_auto_optimization() を呼び出す。"""
    sys.path.insert(0, str(repo_root()))
    try:
        from auto_optimizer import run_auto_optimization
    except ImportError as exc:
        print(f"[auto_trigger_optimizer] auto_optimizer インポート失敗: {exc}", file=sys.stderr, flush=True)
        return None, False

    result = run_auto_optimization()
    if result is None:
        print("[auto_trigger_optimizer] 最適化条件未達（件数不足またはスキップ）", flush=True)
    else:
        ab = result.get("ab_test_result") or {}
        print(
            f"[auto_trigger_optimizer] 最適化完了: {ab.get('reason', 'N/A')}",
            flush=True,
        )
    return result, True


def main() -> int:
    print("[auto_trigger_optimizer] 30日以上前の未登録ケースを失注に自動補完中...", flush=True)
    updated_ids, complete_ok = auto_complete_old_cases(cutoff_days=30)
    print(
        f"[auto_trigger_optimizer] 自動補完: {len(updated_ids)} 件",
        flush=True,
    )

    print("[auto_trigger_optimizer] スコアリング重み最適化をトリガー中...", flush=True)
    _, optimizer_ok = run_optimizer()

    if not (complete_ok and optimizer_ok):
        print(
            "[auto_trigger_optimizer] 警告: 依存モジュールのインポートに失敗したため、"
            "今回は自動補完/最適化トリガーが正しく実行できていません。",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
