#!/usr/bin/env python3
"""Normalize lost_reason values in past_cases.data while preserving raw values."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "lease_data.db"


CANONICAL_REASONS = {
    "",
    "他社競合（レート）",
    "他社競合（その他）",
    "調達方法変更",
    "設備見合わせ",
    "物件不適",
    "業績不振",
    "その他（不成約）",
    "理由未入力",
}

FUNDING_NOT_COMPETITOR = {"現金", "銀行借入対応", "融資対応", "自己資金"}


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().replace(" ", "").replace("　", "")


def normalize_reason(raw_value: object, final_status: str = "") -> str:
    raw = _clean(raw_value)
    status = _clean(final_status)

    if raw in {"", "0", "０", "none", "None", "null", "NULL", "未入力", "不明"}:
        return "理由未入力" if status == "失注" else ""

    if "業績不振" in raw:
        return "業績不振"

    if "中古物件" in raw or "物件不適" in raw or "不可" in raw:
        return "物件不適"

    if "他社" in raw or "他行" in raw or "他リース" in raw:
        if "レート" in raw or "金利" in raw or "%" in raw or "利率" in raw:
            return "他社競合（レート）"
        return "他社競合（その他）"

    if "金利" in raw or "レート" in raw:
        return "他社競合（レート）"

    if (
        "見合" in raw
        or "見させ" in raw
        or "見出せ" in raw
        or "延期" in raw
        or "延用" in raw
        or "投資" in raw
    ):
        return "設備見合わせ"

    if (
        "方法" in raw
        or "自己資金" in raw
        or "自己貴金" in raw
        or "自己責金" in raw
        or "自己金" in raw
        or "白己" in raw
        or "己資" in raw
        or "現金" in raw
        or "融資対応" in raw
    ):
        return "調達方法変更"

    if raw in CANONICAL_REASONS:
        return raw

    return "その他（不成約）"


def normalize_competitor(data: dict) -> tuple[str, str]:
    raw_competitor = _clean(data.get("competitor"))
    raw_name = _clean(data.get("competitor_name"))
    reason = _clean(data.get("lost_reason"))

    if raw_competitor == "競合あり":
        competitor = "競合あり"
    elif raw_competitor == "競合なし":
        competitor = "競合なし"
    elif reason.startswith("他社競合"):
        competitor = "競合あり"
    elif raw_name and raw_name not in {"0", "０"} and raw_name not in FUNDING_NOT_COMPETITOR:
        competitor = "競合あり"
    else:
        competitor = "競合なし"

    competitor_name = "" if raw_name in {"", "0", "０"} else raw_name
    if competitor == "競合なし" and competitor_name in FUNDING_NOT_COMPETITOR:
        competitor_name = ""
    return competitor, competitor_name


def normalize_db(db_path: Path, dry_run: bool = False) -> tuple[int, Counter, Counter]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, final_status, data FROM past_cases ORDER BY id").fetchall()

    updates: list[tuple[str, str]] = []
    before: Counter = Counter()
    after: Counter = Counter()

    for case_id, final_status, data_text in rows:
        try:
            data = json.loads(data_text or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue

        raw_reason = data.get("lost_reason")
        raw_clean = _clean(raw_reason)
        normalized = normalize_reason(raw_reason, final_status or data.get("final_status") or "")
        before[raw_clean] += 1
        after[normalized] += 1

        if data.get("lost_reason_raw") is None and raw_clean not in {"", normalized}:
            data["lost_reason_raw"] = raw_reason
        data["lost_reason"] = normalized

        competitor, competitor_name = normalize_competitor(data)
        if data.get("competitor_raw") is None and _clean(data.get("competitor")) not in {
            "",
            competitor,
        }:
            data["competitor_raw"] = data.get("competitor")
        if data.get("competitor_name_raw") is None and _clean(data.get("competitor_name")) not in {
            "",
            competitor_name,
        }:
            data["competitor_name_raw"] = data.get("competitor_name")
        data["competitor"] = competitor
        data["competitor_name"] = competitor_name

        updates.append((json.dumps(data, ensure_ascii=False), case_id))

    if updates and not dry_run:
        with conn:
            conn.executemany("UPDATE past_cases SET data = ? WHERE id = ?", updates)
    conn.close()
    return len(updates), before, after


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    count, before, after = normalize_db(args.db, dry_run=args.dry_run)
    mode = "DRY RUN" if args.dry_run else "UPDATED"
    print(f"{mode}: {count} rows scanned")
    print("\nAfter:")
    for reason, n in after.most_common():
        label = reason or "(blank)"
        print(f"{label}\t{n}")
    print("\nBefore top:")
    for reason, n in before.most_common(args.top):
        label = reason or "(blank)"
        print(f"{label}\t{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
