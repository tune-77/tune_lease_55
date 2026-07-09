#!/usr/bin/env python3
"""昇格候補キューの承認分を長期記憶ソースへ追記する。

build_shion_memory_promotion_queue.py が出したキューから、ユーザーが承認した
候補だけを knowledge_base/shion_promoted_memories.md へ bullet として追記する。
このファイルは build_shion_memory_index.py のソースなので、次回の
インデックス再構築（夜間 or デプロイ時）で記憶として想起可能になる。

適用済みは data/shion_memory_promotions.jsonl に記録し、二重昇格を防ぐ。

使い方:
    python3 scripts/apply_shion_memory_promotions.py --ids promo_xxx,promo_yyy
    python3 scripts/apply_shion_memory_promotions.py --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = REPO_ROOT / "reports" / "shion_memory_promotion_queue_latest.json"
DEFAULT_TARGET = REPO_ROOT / "knowledge_base" / "shion_promoted_memories.md"
DEFAULT_APPLIED_LOG = REPO_ROOT / "data" / "shion_memory_promotions.jsonl"

_HEADER = """# 紫苑 昇格済み長期記憶

会話から承認を経て昇格した長期記憶。build_shion_memory_index.py が
このファイルの bullet を記憶レコードとして取り込む。
編集する場合は bullet（`- `）単位で。削除ではなく revise
（scripts/revise_shion_memory.py）を優先すること。
"""


def _load_applied_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("candidate_id"):
            ids.add(str(row["candidate_id"]))
    return ids


def apply_promotions(
    queue_path: Path,
    target_path: Path,
    applied_log_path: Path,
    *,
    ids: set[str] | None,
    apply_all: bool,
    dry_run: bool,
) -> int:
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"エラー: キューを読めません: {exc}", file=sys.stderr)
        return 1
    candidates = queue.get("candidates") if isinstance(queue, dict) else None
    if not isinstance(candidates, list) or not candidates:
        print("キューに候補がありません")
        return 0

    applied_ids = _load_applied_ids(applied_log_path)
    selected = []
    for c in candidates:
        cid = str(c.get("candidate_id") or "")
        if not cid or cid in applied_ids:
            continue
        if apply_all or (ids and cid in ids):
            selected.append(c)

    if not selected:
        print("承認対象がありません（既に適用済みか、ID不一致）")
        return 0

    today = datetime.now().date().isoformat()
    bullets = [
        f"- {str(c.get('proposed_content') or '').strip()}"
        f"（昇格 {today} / {c.get('kind')} / {c.get('candidate_id')}）"
        for c in selected
    ]

    if dry_run:
        print(f"[dry-run] {target_path} へ {len(selected)} 件追記予定:")
        for b in bullets:
            print(" ", b[:120])
        return 0

    target_path.parent.mkdir(parents=True, exist_ok=True)
    current = target_path.read_text(encoding="utf-8") if target_path.exists() else _HEADER
    target_path.write_text(current.rstrip() + "\n\n" + "\n".join(bullets) + "\n", encoding="utf-8")

    applied_log_path.parent.mkdir(parents=True, exist_ok=True)
    with applied_log_path.open("a", encoding="utf-8") as fh:
        for c in selected:
            fh.write(
                json.dumps(
                    {
                        "candidate_id": c.get("candidate_id"),
                        "kind": c.get("kind"),
                        "content": c.get("proposed_content"),
                        "applied_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"{len(selected)} 件を {target_path} へ追記しました（次回インデックス再構築で記憶化）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="記憶昇格候補の承認・適用")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--applied-log", type=Path, default=DEFAULT_APPLIED_LOG)
    parser.add_argument("--ids", default="", help="承認する candidate_id（カンマ区切り）")
    parser.add_argument("--all", action="store_true", help="キューの全候補を承認する")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ids = {token.strip() for token in args.ids.split(",") if token.strip()} or None
    if not ids and not args.all:
        print("エラー: --ids か --all を指定してください", file=sys.stderr)
        return 1
    return apply_promotions(
        args.queue,
        args.target,
        args.applied_log,
        ids=ids,
        apply_all=args.all,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
