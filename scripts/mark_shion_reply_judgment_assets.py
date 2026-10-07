#!/usr/bin/env python3
"""判断資産に混入した紫苑自身の返答に出所（content_source=shion）を付けて除外する（REV-498）。

対話室で「保存して」と頼まれた紫苑の返答（「承知いたしました。### 判断資産：…」「① 推論：…」
「```markdown」など）が、ユーザーが教えた知識と同じ扱いで正規判断資産になり、方針らしさ採点や
回答に使われていた。削除はしない。バックアップを取ったうえで次だけを行う。

- 出所を付ける: content_source="shion"
- 使わない: status を "excluded_shion_reply" にし、元の status を status_before_exclusion に残す
  （回答・記憶索引・方針らしさ採点はどれも status == "active" だけを使う）

判定は対話由来（evidence_paths が manual://）のルールだけを見る。紫苑の返答に特有の書き方
（承知いたしました・Markdown 見出し・コードブロック・「① 推論」など）があれば紫苑の返答とする。
です・ます調が多いだけ／「ですね」程度のものは判定が微妙として一覧に出すだけで変更しない。

既定は確認のみ（--apply で書き込む）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_JSON = REPO_ROOT / "data" / "canonical_judgment_rules.json"
DEFAULT_BACKUP_DIR = REPO_ROOT / "data" / "backups" / "judgment_asset_content_source"
LOCK_PATH = REPO_ROOT / "data" / ".judgment_asset_promotion.lock"
EXCLUDED_STATUS = "excluded_shion_reply"

_SHION_REPLY = re.compile(
    r"承知(いた)?しました|かしこまりました|(^|\s)#{1,4} |```|[①②③] 推論|以下の項目を|コピー＆ペースト"
    r"|お役に立て|いかがでしょう|させていただ"
)
_SOFT = re.compile(r"ですね[。！]|ですよ[。！]|ましょう[。！]|示唆")
_POLITE = re.compile(r"(です|ます|ません|でしょう|ください)[。！]")


def _texts(rule: dict[str, Any]) -> str:
    return " ".join([str(rule.get("canonical_statement") or ""), *map(str, rule.get("sample_claims") or [])])


def _from_dialogue(rule: dict[str, Any]) -> bool:
    return any(str(path).startswith("manual://") for path in rule.get("evidence_paths") or [])


def _polite_ratio(text: str) -> float:
    sentences = [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]
    return sum(1 for s in sentences if _POLITE.search(s)) / max(1, len(sentences))


def classify(rule: dict[str, Any]) -> str:
    """shion（紫苑の返答）/ uncertain（判定が微妙）/ user のどれか。"""
    if rule.get("content_source") == "shion":
        return "shion"
    if not _from_dialogue(rule):
        return "user"
    text = _texts(rule)
    if _SHION_REPLY.search(text):
        return "shion"
    statement = str(rule.get("canonical_statement") or "")
    if _SOFT.search(statement) or _polite_ratio(statement) >= 0.5:
        return "uncertain"
    return "user"


def plan(rules: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"shion": [], "uncertain": []}
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        kind = classify(rule)
        if kind in result:
            result[kind].append(rule)
    return result


def apply_marks(rules: list[dict[str, Any]], *, now: str) -> int:
    changed = 0
    for rule in rules:
        if not isinstance(rule, dict) or classify(rule) != "shion":
            continue
        if rule.get("content_source") == "shion" and rule.get("status") == EXCLUDED_STATUS:
            continue
        rule["content_source"] = "shion"
        if rule.get("status") != EXCLUDED_STATUS:
            rule["status_before_exclusion"] = str(rule.get("status") or "")
            rule["status"] = EXCLUDED_STATUS
        rule["excluded_reason"] = "紫苑自身の返答がユーザーの教えた知識として登録されていた（REV-498）"
        rule["excluded_at"] = now
        changed += 1
    return changed


def _summary_line(rule: dict[str, Any]) -> str:
    statement = " ".join(str(rule.get("canonical_statement") or "").split())
    return f"- `{rule.get('id')}` [{rule.get('status')}] {statement[:80]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--apply", action="store_true", help="バックアップを取って出所と除外を書き込む")
    args = parser.parse_args()

    from filelock import FileLock

    with FileLock(str(LOCK_PATH), timeout=30):
        store = json.loads(args.canonical.read_text(encoding="utf-8"))
        rules = store.get("rules") or []
        found = plan(rules)
        print(f"紫苑の返答と判定: {len(found['shion'])} 件")
        for rule in found["shion"]:
            print(_summary_line(rule))
        print(f"判定が微妙（変更しない・ユーザー確認待ち）: {len(found['uncertain'])} 件")
        for rule in found["uncertain"]:
            print(_summary_line(rule))
        if not args.apply:
            print("確認のみ（--apply で書き込み）")
            return 0
        now = dt.datetime.now().isoformat(timespec="seconds")
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"canonical_judgment_rules_{now.replace(':', '')}.json"
        shutil.copy2(args.canonical, backup)
        changed = apply_marks(rules, now=now)
        store["summary"] = {
            **(store.get("summary") or {}),
            "active_rules": sum(1 for rule in rules if isinstance(rule, dict) and rule.get("status") == "active"),
        }
        tmp = args.canonical.with_suffix(args.canonical.suffix + ".tmp")
        tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(args.canonical)
        print(f"書き込み: {changed} 件に出所=shion と除外を付けた（バックアップ: {backup}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
