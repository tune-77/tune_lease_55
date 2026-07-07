#!/usr/bin/env python3
"""Replace demo DB company names with dog-breed based aliases."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "demo.db"

DOG_BREEDS = [
    "ブルドッグ",
    "柴犬",
    "ダックス",
    "コーギー",
    "ハスキー",
    "ポメラニアン",
    "ビーグル",
    "ラブラドール",
    "シェパード",
    "秋田犬",
    "チワワ",
    "プードル",
    "ボーダーコリー",
    "ゴールデン",
    "パグ",
    "ドーベルマン",
    "マルチーズ",
    "シーズー",
    "サモエド",
    "テリア",
    "ボクサー",
    "グレートピレニーズ",
    "キャバリア",
    "ミニチュアシュナウザー",
    "バセットハウンド",
    "ジャックラッセル",
    "イタグレ",
    "紀州犬",
    "甲斐犬",
    "ボルゾイ",
]

INDUSTRY_SUFFIXES = [
    (("06", "建設", "工事"), ["建設", "工務店", "土木", "設備工業"]),
    (("09", "食料", "食品"), ["食品", "フーズ", "製菓", "ミート"]),
    (("24", "金属", "製造", "機械"), ["工業", "製作所", "精機", "メタル"]),
    (("44", "運送", "貨物", "道路"), ["運輸", "物流", "急便", "ロジスティクス"]),
    (("76", "飲食"), ["食堂", "ダイニング", "カフェ", "キッチン"]),
    (("83", "医療", "病院", "診療"), ["医療", "クリニック", "メディカル", "ヘルスケア"]),
    (("サービス",), ["サービス", "商事", "企画", "サポート"]),
]

DEFAULT_SUFFIXES = ["商事", "産業", "リース", "総合企画"]


def suffixes_for_industry(industry: str) -> list[str]:
    text = str(industry or "")
    for triggers, suffixes in INDUSTRY_SUFFIXES:
        if any(trigger in text for trigger in triggers):
            return suffixes
    return DEFAULT_SUFFIXES


def alias_for_index(index: int, industry: str) -> str:
    breed = DOG_BREEDS[index % len(DOG_BREEDS)]
    suffixes = suffixes_for_industry(industry)
    suffix = suffixes[(index // len(DOG_BREEDS)) % len(suffixes)]
    generation = index // (len(DOG_BREEDS) * len(suffixes))
    if generation:
        return f"{breed}{suffix}{generation + 1}号"
    return f"{breed}{suffix}"


def load_past_case_names(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    rows = conn.execute("SELECT id, industry_sub, data FROM past_cases ORDER BY timestamp, id").fetchall()
    result: list[tuple[str, str, str]] = []
    for case_id, industry, raw in rows:
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            data = {}
        name = str(data.get("company_name") or "").strip()
        if name:
            result.append((str(case_id), str(industry or ""), name))
    return result


def replace_names_in_json(value: str, mapping: dict[str, str]) -> tuple[str, bool]:
    try:
        data = json.loads(value or "{}")
    except json.JSONDecodeError:
        return value, False
    changed = False

    def walk(obj):
        nonlocal changed
        if isinstance(obj, dict):
            for key, child in list(obj.items()):
                if key == "company_name" and isinstance(child, str) and child in mapping:
                    obj[key] = mapping[child]
                    changed = True
                else:
                    walk(child)
        elif isinstance(obj, list):
            for child in obj:
                walk(child)

    walk(data)
    return json.dumps(data, ensure_ascii=False), changed


def dogify(db_path: Path, *, dry_run: bool = False) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        case_names = load_past_case_names(conn)
        mapping: dict[str, str] = {}
        for index, (_case_id, industry, original_name) in enumerate(case_names):
            mapping.setdefault(original_name, alias_for_index(index, industry))

        next_index = len(mapping)
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_history'").fetchone():
            for row in conn.execute("SELECT DISTINCT company_name FROM conversation_history WHERE company_name IS NOT NULL AND company_name != ''"):
                original = str(row["company_name"] or "").strip()
                if original and original not in mapping:
                    mapping[original] = alias_for_index(next_index, "サービス業全般")
                    next_index += 1

        past_updates: list[tuple[str, str]] = []
        for row in conn.execute("SELECT id, data FROM past_cases ORDER BY timestamp, id"):
            new_data, changed = replace_names_in_json(row["data"], mapping)
            if changed:
                past_updates.append((new_data, row["id"]))

        conversation_updates: list[tuple[str, int]] = []
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_history'").fetchone():
            for row in conn.execute("SELECT id, company_name FROM conversation_history WHERE company_name IS NOT NULL"):
                original = str(row["company_name"] or "")
                if original in mapping:
                    conversation_updates.append((mapping[original], int(row["id"])))

        if not dry_run:
            conn.executemany("UPDATE past_cases SET data = ? WHERE id = ?", past_updates)
            conn.executemany("UPDATE conversation_history SET company_name = ? WHERE id = ?", conversation_updates)
            conn.commit()

        return {
            "unique_names": len(mapping),
            "past_cases_updated": len(past_updates),
            "conversation_history_updated": len(conversation_updates),
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = dogify(args.db, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
