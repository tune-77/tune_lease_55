#!/usr/bin/env python3
"""スクリーニングレポートの品質フィードバックを集計して改善台帳に追記するスクリプト。

入力: data/report_quality_log.jsonl
  各行: {"ts": "...", "report_id": "uuid", "rating": "good"|"bad", "surface": "...", "comment": "..."}

直近30日の bad 率が 50% 以上かつ最低10件のサンプルがある場合に
ledger_rules.json へ type="manual", pending_review=true で追記する。
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_QUALITY_LOG = PROJECT_ROOT / "data" / "report_quality_log.jsonl"
LEDGER_FILE = PROJECT_ROOT / "api" / "rule_engine" / "ledger_rules.json"

LOOKBACK_DAYS = 30
BAD_RATE_THRESHOLD = 0.50
MIN_SAMPLES = 10
REV_ID = "REPORT-QUALITY-LOOP"


def load_entries(lookback_days: int = LOOKBACK_DAYS) -> list[dict]:
    if not REPORT_QUALITY_LOG.exists():
        print("[analyze_report_quality] report_quality_log.jsonl が存在しません。スキップします。", flush=True)
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    entries = []
    try:
        lines = REPORT_QUALITY_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts_str = str(entry.get("ts") or "")
        try:
            ts = datetime.fromisoformat(ts_str).astimezone(timezone.utc)
        except Exception:
            continue
        if ts >= cutoff:
            entries.append(entry)
    return entries


def load_ledger() -> list[dict]:
    if not LEDGER_FILE.exists():
        return []
    try:
        return json.loads(LEDGER_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_ledger(rules: list[dict]) -> None:
    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_FILE.write_text(
        json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def already_exists(ledger: list[dict]) -> bool:
    for entry in ledger:
        if entry.get("rev_id") == REV_ID:
            return True
        if (
            entry.get("category") == "report_quality"
            and entry.get("source") == "analyze_report_quality"
        ):
            return True
    return False


def max_rev_number(ledger: list[dict]) -> int:
    max_num = 0
    for entry in ledger:
        m = re.match(r"REV-(\d+)", str(entry.get("rev_id") or ""))
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num


def main() -> None:
    entries = load_entries()
    if not entries:
        print(f"[analyze_report_quality] 直近{LOOKBACK_DAYS}日のサンプルなし。", flush=True)
        return

    good = sum(1 for e in entries if e.get("rating") == "good")
    bad = sum(1 for e in entries if e.get("rating") == "bad")
    total = good + bad
    print(f"[analyze_report_quality] サンプル: {total}件 (good={good}, bad={bad})", flush=True)

    if total < MIN_SAMPLES:
        print(
            f"[analyze_report_quality] サンプル不足（{total}/{MIN_SAMPLES}件）。スキップ。",
            flush=True,
        )
        return

    bad_rate = bad / total
    if bad_rate < BAD_RATE_THRESHOLD:
        print(
            f"[analyze_report_quality] bad 率 {bad_rate:.0%} < 閾値 {BAD_RATE_THRESHOLD:.0%}。追記不要。",
            flush=True,
        )
        return

    ledger = load_ledger()
    if already_exists(ledger):
        print("[analyze_report_quality] 既存エントリあり。スキップ。", flush=True)
        return

    base_rev = max_rev_number(ledger)
    base_rev += 1
    rev_id = f"REV-{base_rev:03d}q"
    now_iso = datetime.now(timezone.utc).isoformat()
    desc = (
        f"[レポート品質] bad 率 {bad_rate:.0%}（{bad}/{total}件, 直近{LOOKBACK_DAYS}日）—"
        " report_generator.py または審査コメント生成ロジックの改善が必要"
    )
    new_entry = {
        "rev_id": rev_id,
        "type": "manual",
        "pending_review": True,
        "category": "report_quality",
        "description": desc,
        "status": "pending_review",
        "source": "analyze_report_quality",
        "detected_at": now_iso,
        "bad_count": bad,
        "good_count": good,
        "sample_count": total,
        "bad_rate": round(bad_rate, 3),
        "affected_files": ["report_generator.py"],
        "risk": "low",
        "auto_fix_allowed": False,
    }
    ledger.append(new_entry)
    save_ledger(ledger)
    print(f"[analyze_report_quality] 追記: {rev_id} — {desc}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
