#!/usr/bin/env python3
"""
Claude / Dispatch 会話ログ → Obsidian Vault エクスポート

~/.claude/projects/ 配下の JSONL セッションファイルを読み込み、
Obsidian Vault の「Claude会話記録/」フォルダに日付別 Markdown として保存する。

使用方法:
    python3 scripts/export_claude_to_obsidian.py [--dry-run] [--days N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

VAULT_PATH = Path.home() / "Documents" / "Obsidian Vault"
OUTPUT_SUBDIR = "Claude会話記録"
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"

# セッションを探すプロジェクトディレクトリ（優先順）
PROJECT_DIRS = [
    "-Users-kobayashiisaoryou",                          # Dispatch/Cowork
    "-Users-kobayashiisaoryou-clawd-tune-lease-55",      # tune_lease_55
]

ROLE_LABELS = {
    "user": "🧑 Tune",
    "assistant": "🤖 Claude",
}


def extract_text(content) -> str:
    """content フィールドからテキストを抽出する。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        return extract_text(content.get("content", ""))
    return ""


def load_sessions(days: int) -> dict[str, list[dict]]:
    """指定日数分のセッションを読み込み、日付→メッセージリストで返す。"""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    by_date: dict[str, list[dict]] = defaultdict(list)

    for proj_dir in PROJECT_DIRS:
        proj_path = CLAUDE_PROJECTS / proj_dir
        if not proj_path.exists():
            continue
        for jsonl_file in sorted(proj_path.glob("*.jsonl")):
            if jsonl_file.stat().st_mtime < cutoff.timestamp():
                continue
            try:
                with open(jsonl_file, encoding="utf-8") as f:
                    lines = [json.loads(l) for l in f if l.strip()]
            except Exception:
                continue

            for entry in lines:
                if entry.get("type") not in ("user", "assistant"):
                    continue
                ts_raw = entry.get("timestamp", "")
                if not ts_raw:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except Exception:
                    continue
                if ts < cutoff:
                    continue

                msg = entry.get("message", {})
                if isinstance(msg, dict):
                    text = extract_text(msg.get("content", ""))
                else:
                    text = extract_text(msg)

                # ツール呼び出し・システムメッセージは除外
                if not text or text.startswith("<task-notification") or text.startswith("<system-reminder"):
                    continue
                # 短すぎるものは除外
                if len(text) < 3:
                    continue

                date_str = ts.astimezone().strftime("%Y-%m-%d")
                time_str = ts.astimezone().strftime("%H:%M")
                by_date[date_str].append({
                    "role": entry["type"],
                    "time": time_str,
                    "text": text,
                    "session": jsonl_file.stem,
                    "project": proj_dir,
                })

    return by_date


def render_md(date_str: str, messages: list[dict]) -> str:
    """日付と会話リストから Markdown を生成する。"""
    lines = [
        f"---",
        f"date: {date_str}",
        f"tags: [Claude会話記録, claude, dispatch]",
        f"total_messages: {len(messages)}",
        f"---",
        f"",
        f"# {date_str} Claude 会話記録",
        f"",
        f"**総メッセージ数**: {len(messages)}",
        f"",
        f"---",
        f"",
    ]
    for msg in messages:
        label = ROLE_LABELS.get(msg["role"], msg["role"])
        lines.append(f"**{label}** `{msg['time']}`")
        lines.append("")
        # 長すぎるメッセージは短縮
        text = msg["text"]
        if len(text) > 2000:
            text = text[:2000] + "\n…（省略）"
        lines.append(text)
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--days", type=int, default=7, help="何日分を対象にするか（デフォルト7）")
    args = parser.parse_args()

    by_date = load_sessions(args.days)
    if not by_date:
        print("対象セッションなし")
        return

    out_dir = VAULT_PATH / OUTPUT_SUBDIR
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for date_str in sorted(by_date.keys()):
        msgs = sorted(by_date[date_str], key=lambda m: m["time"])
        md = render_md(date_str, msgs)
        out_path = out_dir / f"{date_str}_Claude会話セッション.md"
        print(f"  {date_str}: {len(msgs)}件 → {out_path}")
        if not args.dry_run:
            out_path.write_text(md, encoding="utf-8")

    if args.dry_run:
        print("[dry-run] 書き込みなし")
    else:
        print("完了")


if __name__ == "__main__":
    main()
