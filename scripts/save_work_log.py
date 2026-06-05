#!/usr/bin/env python3
"""Codexスタイルの作業ログを memory/ と Obsidian に保存するスクリプト。

使い方:
  python scripts/save_work_log.py \
    --title "UMAPタイムアウト問題" \
    --what "scoring_core.pyのUMAP呼び出しをモジュールレベルキャッシュに変更" \
    --why-hard "FastAPIスレッドプール枯渇・複数層にまたがる原因" \
    --next-time "scoring_core.pyを触るときはUMAP存在を先に確認" \
    --lesson "UMAPモデルは毎リクエストロードしない → モジュールレベルキャッシュ必須" \
    --pr 282
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

# プロジェクトルートとmobile_appを検索パスに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mobile_app"))
sys.path.insert(0, str(ROOT))  # obsidian_query.py はルート直下

MEMORY_DIR = Path.home() / ".claude" / "projects" / "-Users-kobayashiisaoryou-clawd-tune-lease-55" / "memory"
CLAUDE_MD = ROOT / "CLAUDE.md"


def _build_frontmatter(tags: list[str]) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    tag_str = ", ".join(tags)
    return f"---\ndate: {now}\ntype: work_log\ntags: [{tag_str}]\n---\n\n"


def _build_body(title: str, what: str, why_hard: str, next_time: str, lesson: str, pr: str | None) -> str:
    pr_suffix = f"（PR #{pr}）" if pr else ""
    lines = [
        f"## 作業: {title}{pr_suffix}\n",
        "### 何をしたか",
        what,
        "",
        "### なぜ大変だったか",
        why_hard,
        "",
        "### 次回どう切り分けるか",
        next_time,
    ]
    if lesson:
        lines += ["", "### 教訓", lesson]
    return "\n".join(lines) + "\n"


def save_to_memory(title: str, what: str, why_hard: str, next_time: str, lesson: str, pr: str | None, tags: list[str]) -> Path:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MEMORY_DIR / f"work_log_{ts}.md"
    slug = title[:40].replace(" ", "_").replace("/", "-")
    content = f"""---
name: work_log_{ts}
description: 作業ログ: {title}
metadata:
  type: project
---

"""
    content += _build_body(title, what, why_hard, next_time, lesson, pr)
    path.write_text(content, encoding="utf-8")
    return path


def save_to_obsidian(title: str, what: str, why_hard: str, next_time: str, lesson: str, pr: str | None, tags: list[str]) -> dict:
    try:
        from obsidian_bridge import append_work_log
        return append_work_log(title=title, what=what, why_hard=why_hard, next_time=next_time, lesson=lesson, pr=pr, tags=tags)
    except Exception as e:
        return {"status": "skipped", "reason": f"obsidian_bridge error: {type(e).__name__}: {e}"}


def append_lesson_to_claude_md(lesson: str) -> None:
    """CLAUDE.mdのやらかしパターンセクションにlessonを追記する。"""
    if not CLAUDE_MD.exists():
        return
    text = CLAUDE_MD.read_text(encoding="utf-8")
    marker = "```\n\n---"
    closing_fence = "```"
    # やらかしパターンの closing ``` を探して直前に追記
    pattern_start = text.find("### 6. やらかしパターン")
    if pattern_start == -1:
        return
    # closing fence の位置を探す（パターンセクション内）
    fence_pos = text.find(closing_fence, pattern_start)
    # 最初の ``` はopening、次の ``` がclosing
    open_fence = text.find("```", pattern_start)
    if open_fence == -1:
        return
    close_fence = text.find("```", open_fence + 3)
    if close_fence == -1:
        return
    insert_pos = close_fence
    new_line = f"✗ {lesson.strip()}\n"
    new_text = text[:insert_pos] + new_line + text[insert_pos:]
    CLAUDE_MD.write_text(new_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="作業ログを保存する")
    parser.add_argument("--title", required=True, help="作業タイトル")
    parser.add_argument("--what", required=True, help="何をしたか")
    parser.add_argument("--why-hard", default="", help="なぜ大変だったか")
    parser.add_argument("--next-time", default="", help="次回どう切り分けるか")
    parser.add_argument("--lesson", default="", help="教訓（CLAUDE.mdのやらかしパターンに追記）")
    parser.add_argument("--pr", default=None, help="PR番号")
    parser.add_argument("--tags", default="作業ログ", help="カンマ区切りのタグ")
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    if "作業ログ" not in tags:
        tags.insert(0, "作業ログ")

    mem_path = save_to_memory(args.title, args.what, args.why_hard, args.next_time, args.lesson, args.pr, tags)
    print(f"[memory] saved: {mem_path}")

    obsidian_result = save_to_obsidian(args.title, args.what, args.why_hard, args.next_time, args.lesson, args.pr, tags)
    print(f"[obsidian] {obsidian_result}")

    if args.lesson:
        append_lesson_to_claude_md(args.lesson)
        print(f"[claude.md] lesson appended")


if __name__ == "__main__":
    main()
