#!/usr/bin/env python3
"""
レシピ対話型レビュースクリプト。

data/recipes/pending/ にある未承認レシピを 1 件ずつ表示し、
承認 / 却下 / スキップ / 終了 を対話的に選択できる。

使い方:
  python3 scripts/review_recipes.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PENDING_DIR = PROJECT_ROOT / "data" / "recipes" / "pending"
APPROVED_DIR = PROJECT_ROOT / "data" / "recipes" / "approved"
REJECTED_DIR = PROJECT_ROOT / "data" / "recipes" / "rejected"

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"


def _colored(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def _show_diff_preview(recipe: dict) -> None:
    for file_entry in recipe.get("files", []):
        rel_path = file_entry["path"]
        abs_path = PROJECT_ROOT / rel_path
        print(f"\n  {_colored('ファイル:', BOLD)} {rel_path}")

        if not abs_path.exists():
            print(f"  {_colored('⚠ ファイルが存在しません', YELLOW)}")
            continue

        try:
            content = abs_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  {_colored(f'読み込みエラー: {e}', RED)}")
            continue

        for i, change in enumerate(file_entry.get("changes", []), 1):
            find_str = change.get("find", "")
            replace_str = change.get("replace", "")
            occurrence = change.get("occurrence")

            occ_label = f" (出現 #{occurrence})" if occurrence else " (全て)"
            print(f"\n  変更 #{i}{occ_label}")

            if find_str in content:
                # 変更前後をプレビュー表示
                find_lines = find_str.splitlines() or [""]
                replace_lines = replace_str.splitlines() or [""]
                print(f"  {_colored('変更前:', RED)}")
                for line in find_lines[:5]:
                    print(f"    {_colored('- ' + line, RED)}")
                if len(find_lines) > 5:
                    print(f"    {_colored(f'... ({len(find_lines) - 5} 行省略)', DIM)}")
                print(f"  {_colored('変更後:', GREEN)}")
                for line in replace_lines[:5]:
                    print(f"    {_colored('+ ' + line, GREEN)}")
                if len(replace_lines) > 5:
                    print(f"    {_colored(f'... ({len(replace_lines) - 5} 行省略)', DIM)}")
            else:
                print(f"  {_colored('⚠ find 文字列がファイル内に見つかりません', YELLOW)}")
                print(f"    find: {find_str!r:.80}")


def _show_recipe(recipe: dict) -> None:
    rev = recipe.get("rev", "?")
    title = recipe.get("title", "?")
    rtype = recipe.get("type", "?")
    safety = recipe.get("safety", "none")
    max_lines = recipe.get("max_lines_changed", 50)
    generated_at = recipe.get("generated_at", "")

    print()
    print("=" * 60)
    print(f"  {_colored(rev, BOLD + CYAN)}  {title}")
    print(f"  type={rtype}  safety={safety}  max_lines={max_lines}")
    if generated_at:
        print(f"  生成日時: {generated_at}")
    print("=" * 60)

    _show_diff_preview(recipe)
    print()


def _prompt() -> str:
    try:
        return input(_colored("[a]pprove / [r]eject / [s]kip / [q]uit > ", BOLD)).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def main() -> None:
    APPROVED_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    pending = sorted(PENDING_DIR.glob("*.json"))
    if not pending:
        print("[review_recipes] 承認待ちレシピはありません")
        return

    print(f"[review_recipes] 承認待ちレシピ {len(pending)} 件")

    approved_count = 0
    rejected_count = 0
    skipped_count = 0

    for recipe_path in pending:
        try:
            with recipe_path.open(encoding="utf-8") as f:
                recipe = json.load(f)
        except Exception as e:
            print(f"[review_recipes] 読み込みエラー ({recipe_path.name}): {e}")
            skipped_count += 1
            continue

        _show_recipe(recipe)

        while True:
            action = _prompt()
            if action in ("a", "r", "s", "q"):
                break
            print("  a / r / s / q のいずれかを入力してください")

        if action == "q":
            print("[review_recipes] 終了します")
            break
        elif action == "a":
            recipe["approved_at"] = datetime.now(timezone.utc).isoformat()
            dest = APPROVED_DIR / recipe_path.name
            with dest.open("w", encoding="utf-8") as f:
                json.dump(recipe, f, ensure_ascii=False, indent=2)
            recipe_path.unlink()
            print(f"  {_colored('✓ 承認しました', GREEN)}: {dest.name}")
            approved_count += 1
        elif action == "r":
            shutil.move(str(recipe_path), REJECTED_DIR / recipe_path.name)
            print(f"  {_colored('✗ 却下しました', RED)}: {recipe_path.name}")
            rejected_count += 1
        else:  # skip
            print(f"  {_colored('- スキップ', YELLOW)}: {recipe_path.name}")
            skipped_count += 1

    print()
    print(
        f"[review_recipes] 結果 — "
        f"承認: {approved_count}件 / 却下: {rejected_count}件 / スキップ: {skipped_count}件"
    )


if __name__ == "__main__":
    main()
