#!/usr/bin/env python3
"""未完了の調査約束(pending)を read-only 検索で自動下調べし、finding を付与する。

ハッカソン安全運用（読む・報告するのみ／実装・git・デプロイ・外部接続追加はしない）に
従い、既存の read-only ナレッジ検索（search_lease_wiki）だけを使う。LLM 生成は行わない。
各 pending 約束に「関連ナレッジの要約」を finding として付け、次の対話や日次レポートで
紫苑が結果を報告できるようにする（＝約束を「自分で調べる」導線）。

使い方:
  python scripts/investigate_pending_tasks.py            # 最大5件を下調べ
  python scripts/investigate_pending_tasks.py --limit 3
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import lease_intelligence_pending as pending_mod


def investigate_topic(topic: str) -> str:
    """topic を read-only 検索し、短い finding 文を返す（見つからなければ空）。"""
    topic = (topic or "").strip()
    if not topic:
        return ""
    try:
        from lease_intelligence_tools import search_lease_wiki

        result = search_lease_wiki(topic, limit=2)
    except Exception:
        return ""
    results = result.get("results") if isinstance(result, dict) else None
    if not results:
        return ""
    parts: list[str] = []
    for r in results[:2]:
        if not isinstance(r, dict):
            continue
        fname = str(r.get("file") or "").strip()
        snippet = str(r.get("snippet") or "").replace("\n", " ").strip()[:160]
        if fname and snippet:
            parts.append(f"{fname}: {snippet}")
        elif snippet:
            parts.append(snippet)
    return " / ".join(parts)[:400]


def investigate_pending(limit: int = 5, now: _dt.datetime | None = None) -> list[dict]:
    """未調査の open pending 約束を下調べして finding を付与する。処理したタスクを返す。"""
    now = now or _dt.datetime.now()
    open_tasks = [t for t in pending_mod.get_pending_tasks() if not t.get("finding")]
    processed: list[dict] = []
    for task in open_tasks[: max(0, int(limit))]:
        finding = investigate_topic(str(task.get("topic") or ""))
        if finding and pending_mod.attach_finding(str(task.get("id") or ""), finding, now=now):
            task["finding"] = finding
            processed.append(task)
    return processed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5, help="1回に下調べする最大件数")
    args = parser.parse_args()

    processed = investigate_pending(limit=args.limit)
    print(f"investigate_pending: {len(processed)} 件の約束を下調べしました")
    for task in processed:
        print(f"  - {str(task.get('topic'))[:50]} → {str(task.get('finding'))[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
