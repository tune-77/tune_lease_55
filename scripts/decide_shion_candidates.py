#!/usr/bin/env python3
"""SHION対話発の改善候補（dispatch_queue.jsonl・source=shion）に台帳決定を記録するCLI。

背景: lease_intelligence_pending.py の save_countermeasures_to_dispatch() は
候補ごとに乱数ID（SHION-XXXXXX）を振るため、台帳(ledger.jsonl)への決定記録は
タイトル文字列のcanonical_key（pipeline_ledger.compute_key(title, "")）で行う。
これまでは対象タイトルを手で書き写して1件ずつコマンドを打つ必要があったため、
このスクリプトで「一覧表示 → 番号指定でまとめて記録」を1コマンドにまとめる。

使い方:
  python3 scripts/decide_shion_candidates.py                # 未決着の候補を一覧表示
  python3 scripts/decide_shion_candidates.py --reject 1,3
  python3 scripts/decide_shion_candidates.py --park 2 --reason "様子見"
  python3 scripts/decide_shion_candidates.py --list-all      # 決着済みも含めて表示
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent.parent / ".agents" / "skills" / "auto-improvement-pipeline"
sys.path.insert(0, str(_PIPELINE_DIR))
sys.path.insert(0, str(_PIPELINE_DIR / "scripts"))
try:
    import pipeline_ledger as ledger  # type: ignore
except ImportError:
    ledger = None  # type: ignore

DISPATCH_QUEUE_PATH = Path.home() / "Library" / "Logs" / "tunelease" / "dispatch_queue.jsonl"

# pipeline_ledger.is_processed() が理解する決定ステータスのみ許可する。
# "suppressed" はクールダウン抑制の一時的な観測結果であり決定ではないため対象外
# （README_ledger.md / cleanup_improvement_reviews.py 参照）。
_VALID_STATUSES = ("rejected", "parked", "applied", "needs_review", "deleted")


def load_shion_candidates(path: Path | None = None) -> list[dict]:
    """dispatch_queue.jsonl から source=shion の候補だけをフラットなリストで返す。
    重複タイトルは最初の1件のみ残す（同じ提案が複数日にまたがって出ていても1回で決定できる）。
    """
    if path is None:
        path = DISPATCH_QUEUE_PATH
    if not path.exists():
        return []
    seen_titles: set[str] = set()
    candidates: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("source") != "shion":
            continue
        for c in entry.get("candidates", []):
            title = str(c.get("title", "")).strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            candidates.append({
                "id": c.get("id", ""),
                "title": title,
                "reason": c.get("reason", ""),
                "date": entry.get("date", ""),
            })
    return candidates


def annotate_status(candidates: list[dict]) -> list[dict]:
    """各候補に、台帳上の現在の決定状況（未決着なら空文字）を付与する。"""
    for c in candidates:
        if ledger is None:
            c["ledger_status"] = ""
            continue
        key = ledger.compute_key(c["title"], "")
        c["key"] = key
        processed, status = ledger.is_processed(key)
        c["ledger_status"] = status if processed else ""
    return candidates


def _print_list(candidates: list[dict], list_all: bool) -> list[dict]:
    shown = candidates if list_all else [c for c in candidates if not c["ledger_status"]]
    if not shown:
        print("未決着のSHION提案はありません。" if not list_all else "候補がありません。")
        return shown
    for i, c in enumerate(shown, start=1):
        status_note = f"  [{c['ledger_status']}]" if c["ledger_status"] else ""
        print(f"{i}. ({c['date']}) {c['title']}{status_note}")
    if not list_all:
        print("\n--reject/--park/--applied/--needs-review/--deleted に上の番号をカンマ区切りで指定してください。")
    return shown


def _parse_indices(spec: str, max_index: int) -> list[int]:
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if not (1 <= n <= max_index):
            raise SystemExit(f"番号が範囲外です: {n}（1〜{max_index}）")
        indices.append(n)
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reject", help="番号をカンマ区切りで指定（例: 1,3）")
    parser.add_argument("--park", help="番号をカンマ区切りで指定")
    parser.add_argument("--applied", help="番号をカンマ区切りで指定")
    parser.add_argument("--needs-review", help="番号をカンマ区切りで指定")
    parser.add_argument("--deleted", help="番号をカンマ区切りで指定")
    parser.add_argument("--reason", default="", help="記録するreason（省略可）")
    parser.add_argument("--list-all", action="store_true", help="決着済みの候補も表示する")
    args = parser.parse_args()

    if ledger is None:
        raise SystemExit("pipeline_ledger を読み込めませんでした（.agents/skills/auto-improvement-pipeline 配下を確認してください）")

    candidates = annotate_status(load_shion_candidates())
    shown = _print_list(candidates, args.list_all)

    decisions = {
        "rejected": args.reject,
        "parked": args.park,
        "applied": args.applied,
        "needs_review": args.needs_review,
        "deleted": args.deleted,
    }
    if not any(decisions.values()):
        return

    if not shown:
        raise SystemExit("表示対象の候補が無いため番号指定できません。")

    for status, spec in decisions.items():
        if not spec:
            continue
        for idx in _parse_indices(spec, len(shown)):
            c = shown[idx - 1]
            key = c.get("key") or ledger.compute_key(c["title"], "")
            ledger.record(key, status, c["title"], reason=args.reason)
            print(f"記録しました: [{status}] {c['title']}")


if __name__ == "__main__":
    main()
