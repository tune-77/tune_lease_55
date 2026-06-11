#!/usr/bin/env python3
"""
レシピ適用スクリプト。

data/recipes/approved/ のレシピを機械的に実行してブランチ作成・PR 作成を行う。

安全装置:
  - frontend/src/ 以外のパスは即座にスキップ
  - api/, scoring, models/, data/ を含むパスもスキップ
  - git が clean でない場合は全処理を中断
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPROVED_DIR = PROJECT_ROOT / "data" / "recipes" / "approved"
APPLIED_DIR = PROJECT_ROOT / "data" / "recipes" / "applied"
LEDGER_PATH = Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl"

SAFE_PATH_PREFIX = "frontend/src/"
BLOCKED_PATH_KEYWORDS = {"api/", "scoring", "models/", "data/"}


def _is_safe_path(path: str) -> bool:
    if not path.startswith(SAFE_PATH_PREFIX):
        return False
    for kw in BLOCKED_PATH_KEYWORDS:
        if kw in path:
            return False
    return True


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or PROJECT_ROOT, check=check)


def _check_git_clean() -> None:
    result = _run(["git", "status", "--porcelain"])
    if result.stdout.strip():
        raise RuntimeError(
            "git ワーキングツリーに未コミットの変更があります。apply_recipe を中断します。\n"
            + result.stdout
        )


def _count_changed_lines(original: str, modified: str) -> int:
    orig_lines = original.splitlines()
    mod_lines = modified.splitlines()
    changed = sum(1 for a, b in zip(orig_lines, mod_lines) if a != b)
    changed += abs(len(orig_lines) - len(mod_lines))
    return changed


def _apply_find_replace(content: str, change: dict) -> str:
    find = change["find"]
    replace = change["replace"]
    occurrence = change.get("occurrence")

    if find not in content:
        raise ValueError(f"find 文字列が見つかりません: {find!r}")

    if occurrence is None:
        return content.replace(find, replace)

    # 指定された番号の一致のみ置換
    count = 0
    idx = 0
    result = []
    while True:
        pos = content.find(find, idx)
        if pos == -1:
            result.append(content[idx:])
            break
        count += 1
        if count == occurrence:
            result.append(content[idx:pos])
            result.append(replace)
            result.append(content[pos + len(find):])
            break
        result.append(content[idx:pos + len(find)])
        idx = pos + len(find)
    else:
        pass

    joined = "".join(result)
    if count < occurrence:
        raise ValueError(f"occurrence={occurrence} ですが一致が {count} 件しかありません")
    return joined


def _run_safety_check(safety: str) -> tuple[bool, str]:
    if safety == "none":
        return True, ""
    if safety == "tsc":
        result = _run(
            ["npx", "tsc", "--noEmit"],
            cwd=PROJECT_ROOT / "frontend",
            check=False,
        )
        if result.returncode != 0:
            return False, result.stdout + result.stderr
        return True, ""
    if safety == "lint":
        # eslint --fix 禁止: チェックのみ
        result = _run(
            ["npm", "run", "lint"],
            cwd=PROJECT_ROOT / "frontend",
            check=False,
        )
        if result.returncode != 0:
            return False, result.stdout + result.stderr
        return True, ""
    return True, ""


def _get_pr_number(output: str) -> int | None:
    m = re.search(r"/pull/(\d+)", output)
    if m:
        return int(m.group(1))
    return None


def _append_ledger(rev: str, title: str, pr_number: int | None, pr_url: str) -> None:
    entry = {
        "rev_id": rev,
        "status": "applied",
        "title": title,
        "pr_url": pr_url,
        "pr_number": pr_number,
        "reason": "レシピ自動適用",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _process_recipe(recipe_path: Path) -> tuple[str, str]:
    """Returns (status, message): status is 'applied' | 'skipped' | 'error'"""
    try:
        with recipe_path.open(encoding="utf-8") as f:
            recipe = json.load(f)
    except Exception as e:
        return "error", f"JSON 読み込み失敗: {e}"

    rev = recipe.get("rev", recipe_path.stem)
    title = recipe.get("title", "")
    files_spec: list[dict] = recipe.get("files", [])
    safety = recipe.get("safety", "none")
    max_lines = recipe.get("max_lines_changed", 50)

    # パス安全チェック
    for file_entry in files_spec:
        path = file_entry.get("path", "")
        if not _is_safe_path(path):
            return "skipped", f"安全でないパス: {path}"

    # 各ファイルの変更を実行
    original_contents: dict[str, str] = {}
    modified_contents: dict[str, str] = {}
    total_changed_lines = 0

    for file_entry in files_spec:
        rel_path = file_entry["path"]
        abs_path = PROJECT_ROOT / rel_path

        if not abs_path.exists():
            return "skipped", f"ファイルが存在しません: {rel_path}"

        original = abs_path.read_text(encoding="utf-8")
        original_contents[rel_path] = original
        content = original

        for change in file_entry.get("changes", []):
            try:
                content = _apply_find_replace(content, change)
            except ValueError as e:
                return "skipped", str(e)

        changed_lines = _count_changed_lines(original, content)
        total_changed_lines += changed_lines
        modified_contents[rel_path] = content

    if total_changed_lines > max_lines:
        return "skipped", f"変更行数 {total_changed_lines} が上限 {max_lines} を超えています"

    # ファイルに書き込む（ロールバック用にオリジナルを保持済み）
    for rel_path, content in modified_contents.items():
        abs_path = PROJECT_ROOT / rel_path
        abs_path.write_text(content, encoding="utf-8")

    # safety チェック
    ok, safety_msg = _run_safety_check(safety)
    if not ok:
        # ロールバック
        for rel_path, original in original_contents.items():
            abs_path = PROJECT_ROOT / rel_path
            abs_path.write_text(original, encoding="utf-8")
        return "skipped", f"safety チェック失敗 ({safety}): {safety_msg[:200]}"

    # ブランチ作成
    branch = f"auto-recipe/{rev}"
    try:
        _run(["git", "checkout", "-b", branch])
    except subprocess.CalledProcessError as e:
        # ロールバック
        for rel_path, original in original_contents.items():
            (PROJECT_ROOT / rel_path).write_text(original, encoding="utf-8")
        return "error", f"ブランチ作成失敗: {e.stderr}"

    # git add & commit
    changed_files = list(modified_contents.keys())
    try:
        _run(["git", "add"] + changed_files)
        commit_msg = f"[{rev}] レシピ自動適用: {title}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
        _run(["git", "commit", "-m", commit_msg])
    except subprocess.CalledProcessError as e:
        # ブランチ・変更を元に戻す
        _run(["git", "checkout", "master"], check=False)
        _run(["git", "branch", "-D", branch], check=False)
        for rel_path, original in original_contents.items():
            (PROJECT_ROOT / rel_path).write_text(original, encoding="utf-8")
        return "error", f"コミット失敗: {e.stderr}"

    # push
    try:
        _run(["git", "push", "origin", branch])
    except subprocess.CalledProcessError as e:
        _run(["git", "checkout", "master"], check=False)
        return "error", f"push 失敗: {e.stderr}"

    # PR 作成
    files_summary = ", ".join(changed_files)
    pr_body = (
        f"レシピベース自動適用\n\n"
        f"対象ファイル: {files_summary}\n\n"
        f"🤖 Generated with [Claude Code](https://claude.com/claude-code)"
    )
    try:
        pr_result = _run([
            "gh", "pr", "create",
            "--title", f"[{rev}] {title}",
            "--body", pr_body,
            "--base", "master",
        ])
        pr_url = pr_result.stdout.strip()
        pr_number = _get_pr_number(pr_url)
    except subprocess.CalledProcessError as e:
        pr_url = ""
        pr_number = None
        print(f"  警告: PR 作成失敗: {e.stderr}", file=sys.stderr)

    # master に戻る
    _run(["git", "checkout", "master"], check=False)

    # レシピを applied/ に移動
    APPLIED_DIR.mkdir(parents=True, exist_ok=True)
    applied_path = APPLIED_DIR / recipe_path.name

    recipe["applied_at"] = datetime.now(timezone.utc).isoformat()
    if pr_number:
        recipe["pr_number"] = pr_number
    with applied_path.open("w", encoding="utf-8") as f:
        json.dump(recipe, f, ensure_ascii=False, indent=2)
    recipe_path.unlink()

    # ledger に追記
    try:
        _append_ledger(rev, title, pr_number, pr_url)
    except Exception as e:
        print(f"  警告: ledger 追記失敗: {e}", file=sys.stderr)

    return "applied", pr_url


def main() -> None:
    applied_count = 0
    skipped_count = 0
    error_count = 0

    try:
        _check_git_clean()
    except RuntimeError as e:
        print(f"[apply_recipe] {e}", file=sys.stderr)
        sys.exit(1)

    approved_recipes = sorted(APPROVED_DIR.glob("*.json"))
    if not approved_recipes:
        print("[apply_recipe] 承認済みレシピがありません")
        return

    print(f"[apply_recipe] 承認済みレシピ {len(approved_recipes)} 件を処理します")

    for recipe_path in approved_recipes:
        rev = recipe_path.stem
        print(f"\n[apply_recipe] 処理中: {rev}")
        status, message = _process_recipe(recipe_path)

        if status == "applied":
            applied_count += 1
            print(f"  ✓ 適用完了: {message}")
        elif status == "skipped":
            skipped_count += 1
            print(f"  - スキップ: {message}")
        else:
            error_count += 1
            print(f"  ✗ エラー: {message}", file=sys.stderr)

    print(
        f"\n[apply_recipe] 完了 — 適用: {applied_count}件 / "
        f"スキップ: {skipped_count}件 / エラー: {error_count}件"
    )


if __name__ == "__main__":
    main()
