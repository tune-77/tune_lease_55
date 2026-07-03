"""紫苑記憶の改訂履歴（docs/shion_memory_architecture.md Next Step 3 の実装）。

矛盾した・古くなった記憶は削除せず、改訂宣言を追記式ファイル
（data/shion_memory_revisions.jsonl）へ記録し、索引へ適用する:

- 旧記憶: `status=revised`（旧結論として参照可能のまま、想起優先度は下がる）
- 後継記憶: `supersedes=[旧ID]`。--new-content なら新レコードを作成、
  --new-id なら既存レコードへ紐付ける。

改訂宣言が真実の源なので、索引を再生成しても `build_shion_memory_index.py`
が本モジュール経由で再適用する。

使い方:
    python3 scripts/revise_shion_memory.py --old-id mem_xxxx \
        --new-content "コンテナの法定耐用年数は7年（2026-07改訂）" \
        --reason "制度改定で年数が変わった"
    python3 scripts/revise_shion_memory.py --list   # 改訂履歴の確認
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.shion_memory_taxonomy import make_memory_record

DEFAULT_INDEX = REPO_ROOT / "data" / "shion_memory_index.json"
DEFAULT_REVISIONS = REPO_ROOT / "data" / "shion_memory_revisions.jsonl"


def load_revisions(path: Path = DEFAULT_REVISIONS) -> list[dict[str, Any]]:
    """改訂宣言を読み込む。壊れた行は無視する。"""
    revisions: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("old_id"):
            revisions.append(entry)
    return revisions


def apply_revisions(index: dict[str, Any], revisions: list[dict[str, Any]]) -> dict[str, int]:
    """改訂宣言を索引レコードへ適用し、変更件数を返す。冪等。"""
    records = [r for r in index.get("records") or [] if isinstance(r, dict)]
    by_id = {str(r.get("id") or ""): r for r in records}
    summary = {"revised": 0, "superseded_created": 0, "superseded_linked": 0}

    for revision in revisions:
        old_id = str(revision.get("old_id") or "")
        old_record = by_id.get(old_id)
        if old_record is None:
            continue
        if str(old_record.get("status") or "active") not in {"revised", "deprecated", "private"}:
            old_record["status"] = "revised"
            summary["revised"] += 1

        new_id = str(revision.get("new_id") or "")
        new_content = str(revision.get("new_content") or "").strip()
        if new_id and new_id in by_id:
            successor = by_id[new_id]
            supersedes = [str(s) for s in successor.get("supersedes") or []]
            if old_id not in supersedes:
                supersedes.append(old_id)
                successor["supersedes"] = supersedes
                summary["superseded_linked"] += 1
        elif new_content:
            successor_record = make_memory_record(
                new_content,
                source="revision",
                source_path=str(revision.get("source_path") or old_record.get("source_path") or ""),
                memory_type=old_record.get("memory_type") or None,
                confidence=float(revision.get("confidence") or 0.8),
            ).to_dict()
            successor_record["supersedes"] = [old_id]
            successor_id = str(successor_record["id"])
            if successor_id not in by_id:
                records.append(successor_record)
                by_id[successor_id] = successor_record
                summary["superseded_created"] += 1
            else:
                existing = by_id[successor_id]
                supersedes = [str(s) for s in existing.get("supersedes") or []]
                if old_id not in supersedes:
                    supersedes.append(old_id)
                    existing["supersedes"] = supersedes
                    summary["superseded_linked"] += 1

    index["records"] = records
    return summary


def append_revision(
    *,
    old_id: str,
    reason: str,
    new_content: str = "",
    new_id: str = "",
    path: Path = DEFAULT_REVISIONS,
) -> dict[str, Any]:
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "old_id": old_id,
        "reason": reason,
    }
    if new_content:
        entry["new_content"] = new_content
    if new_id:
        entry["new_id"] = new_id
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="紫苑記憶の改訂（revised化と supersedes 紐付け）")
    parser.add_argument("--old-id", help="改訂する旧記憶のID (mem_...)")
    parser.add_argument("--reason", default="", help="改訂理由")
    parser.add_argument("--new-content", default="", help="後継記憶の本文（新規レコード作成）")
    parser.add_argument("--new-id", default="", help="後継として紐付ける既存記憶ID")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--revisions", type=Path, default=DEFAULT_REVISIONS)
    parser.add_argument("--list", action="store_true", help="改訂履歴を表示して終了")
    args = parser.parse_args()

    if args.list:
        for entry in load_revisions(args.revisions):
            print(json.dumps(entry, ensure_ascii=False))
        return 0

    if not args.old_id:
        parser.error("--old-id が必要です（--list で履歴確認）")
    if args.new_content and args.new_id:
        parser.error("--new-content と --new-id は同時に指定できません")

    try:
        index = json.loads(args.index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"索引を読めません: {args.index} ({exc})")
        return 1

    known_ids = {str(r.get("id") or "") for r in index.get("records") or [] if isinstance(r, dict)}
    if args.old_id not in known_ids:
        print(f"旧記憶IDが索引にありません: {args.old_id}")
        return 1
    if args.new_id and args.new_id not in known_ids:
        print(f"後継記憶IDが索引にありません: {args.new_id}")
        return 1

    append_revision(
        old_id=args.old_id,
        reason=args.reason,
        new_content=args.new_content,
        new_id=args.new_id,
        path=args.revisions,
    )
    summary = apply_revisions(index, load_revisions(args.revisions))
    for key, value in summary.items():
        print(f"{key}={value}")

    text = json.dumps(index, ensure_ascii=False, indent=2)
    tmp = args.index.with_suffix(args.index.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(args.index)
    print(f"wrote={args.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
