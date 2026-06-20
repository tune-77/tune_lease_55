#!/usr/bin/env python3
"""ウィザード入力ログを分析して、空欄率が高いフィールドを改善台帳に追記するスクリプト。

入力: data/wizard_input_log.jsonl
  各行: {"ts": "...", "total_fields": N, "empty_count": N, "empty_fields": [...], "surface": "wizard_calculate"}

フィールド別の空欄率が 50% 以上かつ最低 10 件のサンプルがある場合に
ledger_rules.json へ type="manual", pending_review=true で追記する。
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WIZARD_LOG = PROJECT_ROOT / "data" / "wizard_input_log.jsonl"
LEDGER_FILE = PROJECT_ROOT / "api" / "rule_engine" / "ledger_rules.json"

LOOKBACK_DAYS = 14
EMPTY_RATE_THRESHOLD = 0.5
MIN_SAMPLES = 10


def load_entries(lookback_days: int = LOOKBACK_DAYS) -> list[dict]:
    if not WIZARD_LOG.exists():
        print("[analyze_wizard_inputs] wizard_input_log.jsonl が存在しません。スキップします。", flush=True)
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    entries = []
    try:
        lines = WIZARD_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
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
        ts_str = entry.get("ts") or ""
        try:
            ts = datetime.fromisoformat(ts_str).astimezone(timezone.utc)
        except Exception:
            continue
        if ts >= cutoff:
            entries.append(entry)
    return entries


def aggregate_empty_rates(entries: list[dict]) -> dict[str, tuple[int, int]]:
    """フィールド名 → (空欄回数, 総サンプル数) を返す。"""
    empty_counts: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for entry in entries:
        empty_fields = entry.get("empty_fields") or []
        total = entry.get("total_fields") or len(empty_fields)
        for field in empty_fields:
            empty_counts[field] += 1
        for field in empty_fields:
            totals[field] += 1
        # total_fields から tracked フィールド数を推定
        # 全フィールドの total を加算（tracked フィールドベース）
        tracked = entry.get("total_fields", 0)
        if tracked:
            for f in []:  # 既知のフィールドリストがなければスキップ
                totals[f] += 1
    # 各エントリから全追跡フィールドのトータルを再計算
    all_fields: set[str] = set()
    for entry in entries:
        all_fields.update(entry.get("empty_fields") or [])
    # エントリ数を total として使う
    n = len(entries)
    result: dict[str, tuple[int, int]] = {}
    for field in all_fields:
        result[field] = (empty_counts[field], n)
    return result


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


def already_exists(ledger: list[dict], field: str) -> bool:
    for entry in ledger:
        if (
            field in str(entry.get("description") or "")
            and entry.get("category") == "wizard_field_empty"
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
        print(f"[analyze_wizard_inputs] 直近{LOOKBACK_DAYS}日のサンプルなし。", flush=True)
        return

    rate_map = aggregate_empty_rates(entries)
    n = len(entries)
    print(f"[analyze_wizard_inputs] サンプル数: {n}件", flush=True)

    high_empty = {
        field: (empty, total)
        for field, (empty, total) in rate_map.items()
        if total >= MIN_SAMPLES and (empty / total) >= EMPTY_RATE_THRESHOLD
    }

    if not high_empty:
        print(
            f"[analyze_wizard_inputs] 空欄率{int(EMPTY_RATE_THRESHOLD * 100)}%超のフィールドなし（最低{MIN_SAMPLES}件）。",
            flush=True,
        )
        return

    ledger = load_ledger()
    base_rev = max_rev_number(ledger)
    now_iso = datetime.now(timezone.utc).isoformat()
    added = 0

    for field, (empty, total) in sorted(high_empty.items(), key=lambda x: -x[1][0] / max(x[1][1], 1)):
        if already_exists(ledger, field):
            print(f"[analyze_wizard_inputs] スキップ（既存）: {field}", flush=True)
            continue
        base_rev += 1
        rev_id = f"REV-{base_rev:03d}w"
        rate_pct = int(empty / total * 100)
        desc = (
            f"[ウィザード入力] フィールド '{field}' の空欄率が{rate_pct}%"
            f"（{empty}/{total}件, 直近{LOOKBACK_DAYS}日）"
        )
        new_entry = {
            "rev_id": rev_id,
            "type": "manual",
            "pending_review": True,
            "category": "wizard_field_empty",
            "description": desc,
            "status": "pending_review",
            "source": "analyze_wizard_inputs",
            "detected_at": now_iso,
            "field_name": field,
            "empty_count": empty,
            "sample_count": total,
            "empty_rate": round(empty / total, 3),
            "affected_files": ["frontend/src/app/"],
            "risk": "low",
            "auto_fix_allowed": False,
        }
        ledger.append(new_entry)
        added += 1
        print(f"[analyze_wizard_inputs] 追記: {rev_id} — {desc}", flush=True)

    if added > 0:
        save_ledger(ledger)
        print(f"[analyze_wizard_inputs] {added} 件を台帳に追記しました。", flush=True)
    else:
        print("[analyze_wizard_inputs] 新規候補なし。", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
