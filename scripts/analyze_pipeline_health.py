#!/usr/bin/env python3
"""
パイプラインステップの成否ログを集計し、
失敗率の高いステップを改善台帳(ledger_rules.json)に pending_review: true で追記する。

入力: data/pipeline_step_log.jsonl
  各行: {"ts": "2026-06-20T07:00:00", "run_date": "20260620", "step": "extract_obsidian_improvements", "exit_code": 0, "duration_s": 1.2}

出力: api/rule_engine/ledger_rules.json への追記（重複はrev_idで排除）
"""

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "data" / "pipeline_step_log.jsonl"
LEDGER_FILE = PROJECT_ROOT / "api" / "rule_engine" / "ledger_rules.json"

FAILURE_RATE_THRESHOLD = 0.5
MIN_TOTAL_RUNS = 3
LOOKBACK_DAYS = 7

# auto_fix_allowed=true にするための条件
AUTO_FIX_MIN_FAILURE_DAYS = 5   # 7日中5日以上失敗していること
AUTO_FIX_RULE_TYPES = {"patch_json", "config_value"}  # 対応ルール型


def load_recent_logs():
    if not LOG_FILE.exists():
        print("pipeline_step_log.jsonl が存在しません。スキップします。", flush=True)
        return []

    cutoff = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    entries = []
    with LOG_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("run_date", "") >= cutoff:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries


def aggregate(entries):
    counts = defaultdict(lambda: {"good": 0, "bad": 0, "bad_days": set(), "latest_exit_code": None, "latest_ts": ""})
    for e in entries:
        step = e.get("step", "unknown")
        if e.get("exit_code", 1) == 0:
            counts[step]["good"] += 1
        else:
            counts[step]["bad"] += 1
            counts[step]["bad_days"].add(e.get("run_date", ""))
        ts = str(e.get("ts") or "")
        if ts >= counts[step]["latest_ts"]:
            counts[step]["latest_ts"] = ts
            counts[step]["latest_exit_code"] = e.get("exit_code", 1)
    return counts


def has_auto_fix_rule(ledger: list, step: str) -> bool:
    """ledger_rules.json に対象ステップに対応する patch_json / config_value 型ルールが存在するか。"""
    for entry in ledger:
        if entry.get("type") not in AUTO_FIX_RULE_TYPES:
            continue
        if step in str(entry.get("description") or ""):
            return True
    return False


def load_ledger():
    if not LEDGER_FILE.exists():
        return []
    with LEDGER_FILE.open() as f:
        return json.load(f)


def max_rev_number(ledger):
    max_num = 0
    for entry in ledger:
        m = re.match(r"REV-(\d+)", entry.get("rev_id", ""))
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num


def already_exists(ledger, step):
    for entry in ledger:
        if step in entry.get("description", "") and entry.get("status") not in {"resolved", "stale_resolved"}:
            return True
    return False


def resolve_recovered_entries(ledger: list, counts: dict, now_iso: str) -> int:
    """最新実行が成功している過去のパイプライン障害検出を解決済みにする。"""
    resolved = 0
    for entry in ledger:
        if entry.get("source") != "analyze_pipeline_health":
            continue
        if entry.get("status") in {"resolved", "stale_resolved"}:
            continue
        description = str(entry.get("description") or "")
        for step, c in counts.items():
            if step not in description:
                continue
            if c.get("latest_exit_code") == 0:
                entry["status"] = "stale_resolved"
                entry["pending_review"] = False
                entry["resolved_at"] = now_iso
                entry["resolution_reason"] = "直近の同ステップ実行が成功しているため、過去検出を解決済みに更新"
                resolved += 1
            break
    return resolved


def main():
    entries = load_recent_logs()
    if not entries:
        return

    counts = aggregate(entries)

    penalty_steps = []
    for step, c in counts.items():
        total = c["good"] + c["bad"]
        if total < MIN_TOTAL_RUNS:
            continue
        rate = c["bad"] / total
        if rate >= FAILURE_RATE_THRESHOLD:
            penalty_steps.append((step, c["bad"], total, rate, len(c["bad_days"])))

    if not penalty_steps:
        print(f"直近{LOOKBACK_DAYS}日間で失敗率閾値({FAILURE_RATE_THRESHOLD*100:.0f}%)超のステップはありません。", flush=True)
        return

    ledger = load_ledger()
    base_rev = max_rev_number(ledger)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    resolved = resolve_recovered_entries(ledger, counts, now_iso)

    added = 0
    for step, bad, total, rate, bad_days in sorted(penalty_steps, key=lambda x: -x[3]):
        if counts[step].get("latest_exit_code") == 0:
            print(f"スキップ（直近成功）: {step}", flush=True)
            continue
        if already_exists(ledger, step):
            print(f"スキップ（既存）: {step}", flush=True)
            continue

        base_rev += 1
        rev_id = f"REV-{base_rev:03d}a"
        pct = int(rate * 100)
        description = f"[パイプライン自動検出] {step} が過去{LOOKBACK_DAYS}日で失敗率{pct}%（{bad}/{total}件, {bad_days}日失敗）"

        # 7日中5日以上失敗 かつ patch_json/config_value 型ルールが存在する場合のみ自動修正許可
        can_auto_fix = (
            bad_days >= AUTO_FIX_MIN_FAILURE_DAYS
            and has_auto_fix_rule(ledger, step)
        )

        new_entry = {
            "rev_id": rev_id,
            "type": "patch_json" if can_auto_fix else "manual",
            "pending_review": True,
            "category": "pipeline_fix",
            "description": description,
            "status": "pending_review",
            "source": "analyze_pipeline_health",
            "detected_at": now_iso,
            "affected_files": [],
            "risk": "low" if can_auto_fix else "medium",
            "auto_fix_allowed": can_auto_fix,
        }
        ledger.append(new_entry)
        added += 1
        print(f"追記: {rev_id} — {description}", flush=True)

    if added > 0 or resolved > 0:
        with LEDGER_FILE.open("w") as f:
            json.dump(ledger, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"\n台帳更新: 追記 {added} 件 / 解決 {resolved} 件: {LEDGER_FILE}", flush=True)
    else:
        print("追記なし（すべて既存エントリと重複）。", flush=True)

    # サマリー出力
    print("\n=== パイプラインヘルス サマリー（直近7日） ===", flush=True)
    for step, c in sorted(counts.items()):
        total = c["good"] + c["bad"]
        rate = c["bad"] / total if total else 0
        bad_days = len(c["bad_days"])
        active_failure = c.get("latest_exit_code") != 0
        flag = " ⚠️" if active_failure and rate >= FAILURE_RATE_THRESHOLD and total >= MIN_TOTAL_RUNS else ""
        print(
            f"  {step}: 成功{c['good']}/失敗{c['bad']}({bad_days}日) (失敗率{rate*100:.0f}%){flag}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main() or 0)
