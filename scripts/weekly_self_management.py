"""
週次セルフマネジメント: 毎週月曜に実行し、直近7日分の ledger.jsonl を集計して
CLAUDE.md の末尾に ## Weekly Log セクションを追記する。
月曜以外の曜日では即 return する。
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LEDGER_PATH = Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl"
CLAUDE_MD_PATH = PROJECT_ROOT / "CLAUDE.md"

JST = timezone(timedelta(hours=9))


def is_monday() -> bool:
    return datetime.now(JST).weekday() == 0


def load_ledger_entries(since: datetime) -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    entries = []
    try:
        for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts_str = entry.get("timestamp") or entry.get("updated_at") or entry.get("created_at") or ""
                if not ts_str:
                    continue
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=JST)
                if ts >= since:
                    entries.append(entry)
            except Exception:
                continue
    except Exception:
        pass
    return entries


def summarize_entries(entries: list[dict]) -> dict:
    applied = []
    proposed = []
    rejected = []
    parked = []

    seen_keys: dict[str, dict] = {}
    for e in entries:
        key = e.get("key") or e.get("title") or e.get("id") or ""
        if key:
            seen_keys[key] = e

    for e in seen_keys.values():
        status = e.get("status", "")
        title = e.get("title") or e.get("key") or e.get("id") or "不明"
        if status == "applied":
            applied.append(title)
        elif status in ("proposed", "needs_review"):
            proposed.append(title)
        elif status == "rejected":
            rejected.append(title)
        elif status == "parked":
            parked.append(title)

    return {
        "applied": applied,
        "proposed": proposed,
        "rejected": rejected,
        "parked": parked,
    }


def format_list(items: list[str], max_items: int = 10) -> str:
    if not items:
        return "_なし_"
    lines = [f"- {item}" for item in items[:max_items]]
    if len(items) > max_items:
        lines.append(f"- …他 {len(items) - max_items} 件")
    return "\n".join(lines)


def build_weekly_log(summary: dict, week_start: datetime, week_end: datetime) -> str:
    now_str = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    week_label = f"{week_start.strftime('%Y-%m-%d')} 〜 {week_end.strftime('%Y-%m-%d')}"

    applied_count = len(summary["applied"])
    proposed_count = len(summary["proposed"])
    rejected_count = len(summary["rejected"])
    parked_count = len(summary["parked"])

    return f"""
### {week_label}（生成: {now_str}）

| 状態 | 件数 |
|------|------|
| 適用済み (applied) | {applied_count} 件 |
| 提案・要レビュー | {proposed_count} 件 |
| 却下 (rejected) | {rejected_count} 件 |
| 保留 (parked) | {parked_count} 件 |

**適用されたREV:**
{format_list(summary["applied"])}

**新規提案REV（要レビュー含む）:**
{format_list(summary["proposed"])}
"""


def append_weekly_log(log_text: str) -> None:
    if not CLAUDE_MD_PATH.exists():
        print(f"[weekly_self_management] CLAUDE.md が見つかりません: {CLAUDE_MD_PATH}")
        return

    content = CLAUDE_MD_PATH.read_text(encoding="utf-8")

    weekly_log_header = "## Weekly Log"
    if weekly_log_header not in content:
        content = content.rstrip() + f"\n\n{weekly_log_header}\n"

    content = content.rstrip() + "\n" + log_text + "\n"
    CLAUDE_MD_PATH.write_text(content, encoding="utf-8")
    print(f"[weekly_self_management] Weekly Log を CLAUDE.md に追記しました")


def main() -> None:
    if not is_monday():
        print(f"[weekly_self_management] 月曜日以外のため実行をスキップします（曜日: {datetime.now(JST).strftime('%A')}）")
        sys.exit(0)

    now = datetime.now(JST)
    week_end = now
    week_start = now - timedelta(days=7)

    print(f"[weekly_self_management] 集計期間: {week_start.strftime('%Y-%m-%d')} 〜 {week_end.strftime('%Y-%m-%d')}")

    entries = load_ledger_entries(week_start)
    print(f"[weekly_self_management] 対象エントリ数: {len(entries)} 件")

    summary = summarize_entries(entries)
    log_text = build_weekly_log(summary, week_start, week_end)
    append_weekly_log(log_text)


if __name__ == "__main__":
    main()
