#!/usr/bin/env python3
"""
scripts/pdca_rule_lifecycle.py

PDCAルール（data/pdca_ai_rules.json）のライフサイクルを自動管理する。
プロンプト改善ループの効果測定（data/prompt_feedback_log.jsonl）と接続し、
「測定しているのに手動更新待ち」の状態を解消する。

ポリシー:
  1. 期限が近い（--renew-window 日以内）active ルールは、効果測定が良好なら
     自動で --extend-days 日延長する
     良好の基準: ログ件数 >= --min-samples かつ PDCA反映率 > 0
                 かつ 応答変化率（前回差分率）>= --min-changed-rate
  2. 基準を満たさないルールは触らない（既存の仕組みで自然失効する）
  3. 期限切れから --grace-days 日以上経ったルールは status=inactive にして掃除

制約: ログはルール単位ではなく応答単位のため、効果は全体の集計値で判定する
（個別ルールの効果帰属はできない。判定が粗い分、延長のみで新規追加はしない）。
判断は data/pdca_lifecycle_log.jsonl に記録する。

使い方:
  python scripts/pdca_rule_lifecycle.py --dry-run
  python scripts/pdca_rule_lifecycle.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompt_feedback import load_pdca_rules, save_pdca_rules  # noqa: E402
from prompt_feedback_metrics import DEFAULT_LOG_PATH, build_summary, load_jsonl  # noqa: E402

LIFECYCLE_LOG = REPO_ROOT / "data" / "pdca_lifecycle_log.jsonl"


def _parse_date(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10])
    except ValueError:
        return None


def run_lifecycle(
    *,
    apply: bool,
    renew_window: int,
    extend_days: int,
    grace_days: int,
    min_samples: int,
    min_changed_rate: float,
) -> int:
    data = load_pdca_rules()
    meta = [item for item in (data.get("pdca_rule_meta") or []) if isinstance(item, dict)]
    if not meta:
        print("PDCAルールがありません")
        return 0

    rows = load_jsonl(DEFAULT_LOG_PATH)
    summary = build_summary(rows) if rows else {}
    total = int(summary.get("total") or 0)
    pdca_rate = float(summary.get("pdca_rate") or 0.0)
    changed_rate = float(summary.get("previous_diff_rate") or 0.0)
    effective = total >= min_samples and pdca_rate > 0 and changed_rate >= min_changed_rate

    label = "APPLY" if apply else "DRY-RUN"
    print(f"♻️  pdca_rule_lifecycle — {label}")
    print(f"   効果測定: 件数 {total} / PDCA反映率 {pdca_rate}% / 応答変化率 {changed_rate}%"
          f" → {'良好（延長対象）' if effective else '基準未満（自然失効に任せる）'}")

    today = datetime.now().date()
    decisions: list[dict] = []
    renewed = expired_cleanup = 0
    for item in meta:
        status = str(item.get("status") or "active")
        expires = _parse_date(item.get("expires_at"))
        if expires is None:
            continue
        days_left = (expires.date() - today).days
        rule_text = str(item.get("rule") or "")[:80]

        if status != "inactive" and days_left < -grace_days:
            item["status"] = "inactive"
            item["deactivated_reason"] = f"期限切れから{grace_days}日経過（自動掃除）"
            expired_cleanup += 1
            decisions.append({"rule": rule_text, "action": "deactivate", "days_left": days_left})
            print(f"   🗑️  失効掃除: {rule_text}")
        elif status != "inactive" and 0 <= days_left <= renew_window and effective:
            new_expiry = (datetime.now() + timedelta(days=extend_days)).date().isoformat()
            item["expires_at"] = new_expiry
            item["renewed_at"] = datetime.now().isoformat(timespec="seconds")
            item["renewal_reason"] = (
                f"auto: 効果測定良好（PDCA {pdca_rate}% / 変化率 {changed_rate}%）のため{extend_days}日延長"
            )
            renewed += 1
            decisions.append({"rule": rule_text, "action": "renew", "new_expiry": new_expiry})
            print(f"   🔄 自動延長: {rule_text} → {new_expiry}")

    print(f"   延長 {renewed} 件 / 失効掃除 {expired_cleanup} 件")

    if apply and decisions:
        data["pdca_rule_meta"] = meta
        save_pdca_rules(data)
        LIFECYCLE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LIFECYCLE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "summary": {"total": total, "pdca_rate": pdca_rate, "changed_rate": changed_rate},
                "decisions": decisions,
            }, ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PDCAルールの自動延長・失効掃除")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="実行せずに結果を表示")
    mode.add_argument("--apply", action="store_true", help="実際に更新する")
    parser.add_argument("--renew-window", type=int, default=14, help="期限までこの日数以内なら延長を検討（既定14）")
    parser.add_argument("--extend-days", type=int, default=30, help="延長日数（既定30）")
    parser.add_argument("--grace-days", type=int, default=30, help="期限切れ後この日数で inactive 化（既定30）")
    parser.add_argument("--min-samples", type=int, default=10, help="効果判定に必要な最小ログ件数（既定10）")
    parser.add_argument("--min-changed-rate", type=float, default=20.0, help="延長に必要な応答変化率%%（既定20）")
    args = parser.parse_args()
    return run_lifecycle(
        apply=args.apply,
        renew_window=args.renew_window,
        extend_days=args.extend_days,
        grace_days=args.grace_days,
        min_samples=args.min_samples,
        min_changed_rate=args.min_changed_rate,
    )


if __name__ == "__main__":
    sys.exit(main())
