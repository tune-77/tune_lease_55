#!/usr/bin/env python3
"""
scripts/auto_approve_safe_recipes.py

「今回の修正案」（data/recipes/pending/）のうち、安全と判定済みのものを
自動で承認済み（approved/）へ移動する。人間は discuss / review 判定の
レシピだけレビューすればよくなる。

自動承認の条件（すべて満たす場合のみ）:
  - shion_recommendation == "auto"（紫苑が自動修正可と判定）
  - risk_level が "low"（未設定は自動承認しない）
  - 変更対象ファイルがすべて保護対象外（batch_apply と同じ判定を import）

移動の記録は data/recipes/auto_approved_log.jsonl に残す。
UI の「適用待ちへ送る」ボタンと同じ操作をしているだけで、
実際のコード適用はこれまで通り別処理が行う。

使い方:
  python scripts/auto_approve_safe_recipes.py --dry-run
  python scripts/auto_approve_safe_recipes.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.rule_engine.batch_apply import _is_protected  # noqa: E402

RECIPES_ROOT = REPO_ROOT / "data" / "recipes"
AUDIT_LOG = RECIPES_ROOT / "auto_approved_log.jsonl"


def is_auto_approvable(recipe: dict) -> tuple[bool, str]:
    """自動承認できるか判定し、(可否, 理由) を返す。"""
    if recipe.get("shion_recommendation") != "auto":
        return False, f"紫苑判定が auto ではない: {recipe.get('shion_recommendation') or '未設定'}"
    risk = recipe.get("risk_level")
    if risk != "low":
        return False, f"リスクが low ではない: {risk or '未設定'}"
    paths = [str(f.get("path") or "") for f in recipe.get("files") or []]
    if not paths:
        return False, "変更対象ファイルが空"
    for path in paths:
        if _is_protected(path):
            return False, f"保護対象ファイルを含む: {path}"
    return True, "紫苑auto・低リスク・保護対象外"


def main() -> int:
    parser = argparse.ArgumentParser(description="安全な修正案の自動承認")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="実行せずに結果を表示")
    mode.add_argument("--apply", action="store_true", help="実際に承認済みへ移動する")
    args = parser.parse_args()

    pending_dir = RECIPES_ROOT / "pending"
    approved_dir = RECIPES_ROOT / "approved"
    if not pending_dir.exists():
        print(f"承認待ちレシピがありません: {pending_dir}")
        return 0

    label = "DRY-RUN" if args.dry_run else "APPLY"
    print(f"🤖 auto_approve_safe_recipes — {label}")

    approved_count = skipped_count = 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for recipe_file in sorted(pending_dir.glob("*.json")):
        try:
            recipe = json.loads(recipe_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"   ⚠️  {recipe_file.name} 読み込み失敗: {exc}")
            skipped_count += 1
            continue

        ok, reason = is_auto_approvable(recipe)
        rev = recipe.get("rev") or recipe.get("id") or recipe_file.stem
        if not ok:
            print(f"   ⏭️  {rev} 人間レビューへ: {reason}")
            skipped_count += 1
            continue

        print(f"   ✅ {rev} 自動承認: {reason}")
        approved_count += 1
        if args.apply:
            approved_dir.mkdir(parents=True, exist_ok=True)
            recipe_file.rename(approved_dir / recipe_file.name)
            with open(AUDIT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": now,
                    "id": recipe.get("id") or recipe_file.stem,
                    "rev": recipe.get("rev", ""),
                    "title": recipe.get("title", ""),
                    "reason": reason,
                }, ensure_ascii=False) + "\n")

    print(f"   自動承認 {approved_count} 件 / 人間レビュー待ち {skipped_count} 件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
