#!/usr/bin/env python3
"""Agent Action Ledger（backlog §9.2）の日次サマリレポートを生成する。

data/shion_action_ledger.jsonl を集計し、直近 N 日分の行動件数・
risk_level 分布・承認待ち件数を reports/agent_action_ledger_latest.md へ出力する。
監査用の要約であり、実行権限や承認の代替にはならない。
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api.shion_action_ledger import VALID_ACTIONS, read_actions  # noqa: E402

REPORTS_DIR = _REPO_ROOT / "reports"
DEFAULT_MD = REPORTS_DIR / "agent_action_ledger_latest.md"


def _within_days(entry: dict[str, Any], since: dt.datetime) -> bool:
    ts = str(entry.get("timestamp") or "")
    try:
        parsed = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed >= since


def build_summary(entries: list[dict[str, Any]], days: int) -> dict[str, Any]:
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max(1, days))
    recent = [e for e in entries if _within_days(e, since)]

    by_action = {action: 0 for action in sorted(VALID_ACTIONS)}
    by_risk = {"low": 0, "medium": 0, "high": 0}
    pending_approval = []
    for entry in recent:
        action = str(entry.get("action") or "")
        if action in by_action:
            by_action[action] += 1
        risk = str(entry.get("risk_level") or "")
        if risk in by_risk:
            by_risk[risk] += 1
        if entry.get("requires_user_approval") and entry.get("user_approved") is not True:
            pending_approval.append(entry)

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "days": days,
        "total": len(recent),
        "by_action": by_action,
        "by_risk": by_risk,
        "pending_approval_count": len(pending_approval),
        "pending_approval": pending_approval[-20:],
        "recent": recent[-50:],
    }


def write_report(summary: dict[str, Any], md_path: Path) -> None:
    lines = [
        "# 紫苑 Agent Action Ledger サマリ",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- 期間: 直近{summary['days']}日",
        f"- 総行動数: {summary['total']}",
        f"- 承認待ち: {summary['pending_approval_count']}件",
        "",
        "## アクション種別",
    ]
    for action, count in summary["by_action"].items():
        lines.append(f"- {action}: {count}")
    lines.extend(["", "## リスクレベル分布"])
    for risk, count in summary["by_risk"].items():
        lines.append(f"- {risk}: {count}")

    lines.extend(["", "## 承認待ち一覧"])
    if summary["pending_approval"]:
        for entry in summary["pending_approval"]:
            lines.append(
                f"- {entry.get('timestamp')} [{entry.get('action')}] "
                f"{entry.get('summary')}（target: {entry.get('target') or '-'}）"
            )
    else:
        lines.append("- なし")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    entries = read_actions()
    summary = build_summary(entries, args.days)
    write_report(summary, args.md)
    print(f"agent_action_ledger total={summary['total']} pending_approval={summary['pending_approval_count']}")
    print(args.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
