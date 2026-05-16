#!/usr/bin/env python3
"""Fill blank company_name values in past_cases with industry-aware aliases."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "lease_data.db"


DOG_BREEDS = [
    "チワワ", "柴犬", "秋田犬", "豆柴", "コーギー", "ダックス", "ポメラニアン",
    "パグ", "ビーグル", "マルチーズ", "シーズー", "トイプードル", "ボーダーコリー",
    "ラブラドール", "ゴールデン", "シェルティ", "フレンチブル", "ミニピン",
    "サモエド", "ハスキー", "ドーベルマン", "シュナウザー", "グレートピレニーズ",
    "ジャックラッセル",
]


def classify(industry_major: str, industry_sub: str) -> tuple[str, str]:
    sub = industry_sub or ""
    text = f"{industry_major} {industry_sub}"
    if "道路貨物" in sub or "運送" in sub:
        return "transport", "運送"
    if "食料品製造" in sub:
        return "manufacturing", "食品製造"
    if "金属製品" in sub:
        return "manufacturing", "金属製作所"
    if "機械器具" in sub or "生産用機械" in sub:
        return "manufacturing", "機械製作所"
    if "総合工事" in sub:
        return "construction", "総合建設"
    if "職別工事" in sub:
        return "construction", "工務店"
    if "職業紹介" in sub or "労働者派遣" in sub:
        return "staffing", "人材サービス"
    if "医療業" in sub or "病院" in sub or "診療所" in sub:
        return "medical", "クリニック"
    if "介護" in sub or "福祉" in sub:
        return "welfare", "介護サービス"
    if "自動車整備" in sub:
        return "auto", "自動車整備"
    if "小売" in sub:
        return "retail", "小売商店"
    if "卸売" in sub:
        return "wholesale", "卸売商事"
    if "宿泊" in sub or "ホテル" in sub or "旅館" in sub:
        return "hotel", "ホテル"
    if "物品賃貸" in sub or "リース" in sub or "レンタル" in sub:
        return "lease", "リース"
    if "運輸" in text:
        return "transport", "運送"
    if "製造" in text:
        return "manufacturing", "製作所"
    if "建設" in text or "工事" in text:
        return "construction", "建設"
    if "小売" in text:
        return "retail", "小売商店"
    if "卸売" in text:
        return "wholesale", "卸売商事"
    return "default", "商事"


def build_alias(kind: str, suffix: str, ordinal: int) -> str:
    prefix = DOG_BREEDS[(ordinal - 1) % len(DOG_BREEDS)]
    cycle = (ordinal - 1) // len(DOG_BREEDS) + 1
    return f"{prefix}{suffix}{cycle:03d}"


def is_blank(value: object) -> bool:
    if value is None:
        return True
    normalized = str(value).strip().lower()
    return normalized in {"", "0", "０", "none", "null", "不明", "未入力"}


def company_key(case_id: str, data: dict) -> str:
    company_no = str(data.get("company_no") or "").strip()
    if not is_blank(company_no):
        return f"company_no:{company_no}"
    return f"case_id:{case_id}"


def fill_names(
    db_path: Path,
    dry_run: bool = False,
    force_all: bool = False,
) -> list[tuple[str, str, str, str]]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, industry_sub, data FROM past_cases ORDER BY timestamp, id"
    ).fetchall()

    counters: defaultdict[str, int] = defaultdict(int)
    parsed_rows: list[tuple[str, str, dict]] = []
    industry_by_key: defaultdict[str, Counter] = defaultdict(Counter)
    used_names: set[str] = set()
    existing_name_by_key: dict[str, str] = {}
    alias_by_key: dict[str, str] = {}
    updates: list[tuple[str, str, str, str]] = []

    for case_id, industry_sub_col, data_text in rows:
        try:
            data = json.loads(data_text or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        parsed_rows.append((case_id, industry_sub_col, data))
        key = company_key(case_id, data)
        industry_major = str(data.get("industry_major") or "")
        industry_sub = str(data.get("industry_sub") or industry_sub_col or "")
        if not is_blank(industry_sub):
            industry_by_key[key][(industry_major, industry_sub)] += 1

        current_name = str(data.get("company_name") or "").strip()
        if not force_all and not is_blank(current_name):
            key = company_key(case_id, data)
            used_names.add(current_name)
            existing_name_by_key.setdefault(key, current_name)

    for case_id, industry_sub_col, data in parsed_rows:
        current_name = data.get("company_name")
        if not force_all and not is_blank(current_name):
            continue

        key = company_key(case_id, data)
        if not force_all and key in existing_name_by_key:
            alias = existing_name_by_key[key]
        elif key in alias_by_key:
            alias = alias_by_key[key]
        else:
            if industry_by_key[key]:
                industry_major, industry_sub = industry_by_key[key].most_common(1)[0][0]
            else:
                industry_major = str(data.get("industry_major") or "")
                industry_sub = str(data.get("industry_sub") or industry_sub_col or "")
            kind, suffix = classify(industry_major, industry_sub)
            while True:
                counters[kind] += 1
                alias = build_alias(kind, suffix, counters[kind])
                if alias not in used_names:
                    used_names.add(alias)
                    alias_by_key[key] = alias
                    break

        industry_sub = str(data.get("industry_sub") or industry_sub_col or "")
        data["company_name"] = alias
        updates.append((json.dumps(data, ensure_ascii=False), case_id, alias, industry_sub))

    if not dry_run and updates:
        with conn:
            conn.executemany(
                "UPDATE past_cases SET data = ? WHERE id = ?",
                [(data_text, case_id) for data_text, case_id, _, _ in updates],
            )
    conn.close()
    return updates


def _is_valid_json_object(data_text: str) -> bool:
    try:
        return isinstance(json.loads(data_text), dict)
    except json.JSONDecodeError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="replace every company_name, not only blank-like values",
    )
    parser.add_argument("--sample", type=int, default=20)
    args = parser.parse_args()

    updates = fill_names(args.db, dry_run=args.dry_run, force_all=args.force_all)
    mode = "DRY RUN" if args.dry_run else "UPDATED"
    target = "company_name values" if args.force_all else "blank-like company_name values"
    print(f"{mode}: {len(updates)} {target}")
    for _, case_id, alias, industry_sub in updates[: args.sample]:
        print(f"{case_id}\t{industry_sub}\t{alias}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
