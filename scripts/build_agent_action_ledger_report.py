#!/usr/bin/env python3
"""Agent Action Ledger（backlog §9.2）の日次サマリレポートを生成する。

data/shion_action_ledger.jsonl を集計し、直近 N 日分の行動件数・
risk_level 分布・承認待ち件数を reports/agent_action_ledger_latest.md へ出力する。
監査用の要約であり、実行権限や承認の代替にはならない。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api.shion_action_ledger import read_actions, summarize  # noqa: E402

REPORTS_DIR = _REPO_ROOT / "reports"
DEFAULT_MD = REPORTS_DIR / "agent_action_ledger_latest.md"
DEFAULT_JSON = REPORTS_DIR / "agent_action_ledger_latest.json"


def build_summary(entries: list[dict[str, Any]], days: int) -> dict[str, Any]:
    """api.shion_action_ledger.summarize の薄いエイリアス（API側と集計ロジックを共有）。"""
    return summarize(entries, days)


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
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    entries = read_actions()
    summary = build_summary(entries, args.days)
    write_report(summary, args.md)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"agent_action_ledger total={summary['total']} pending_approval={summary['pending_approval_count']}")
    print(args.md)
    print(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
