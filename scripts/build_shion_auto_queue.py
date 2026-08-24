#!/usr/bin/env python3
"""紫苑対話発の承認済み提案（decide_shion_candidates.py --applied）から
Codex自動実行キュー（scripts/execute_codex_queue.py が読む形式）を生成する。

背景: dispatch_queue.jsonl（source=shion）の候補を人間が --applied にしても、
台帳(ledger.jsonl)に記録が残るだけで実行には繋がっていなかった。
このスクリプトは、その「承認済みだが未実行」の候補を、レポート発の改善候補と
同じ安全判定（build_codex_auto_queue.py の is_blocked/is_codex_safe）に通した上で
専用のキューファイルに書き出す。安全判定ロジックは重複実装せず再利用する。

実行は scripts/execute_codex_queue.py が既存のキルスイッチ・連続失敗停止ガード込みで行う。
日次実行件数は呼び出し側（run_daily_improvement_core.sh）が
`--output reports/codex_queue_result_shion_{date}.json` と
`CODEX_QUEUE_DAILY_LIMIT=1` を指定することで、レポート発キューの日次上限3件とは
独立した専用1件/日の枠として扱う。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_codex_auto_queue import (  # noqa: E402
    is_blocked,
    is_codex_safe,
    queue_item,
    refresh_auto_fix_policy,
    repo_root,
)
from decide_shion_candidates import annotate_status, load_shion_candidates  # noqa: E402

STATE_FILE_NAME = "shion_auto_queue_state.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"queued_ids": []}
    try:
        data = load_json(path)
    except Exception:
        return {"queued_ids": []}
    if not isinstance(data.get("queued_ids"), list):
        data["queued_ids"] = []
    return data


def applied_candidates(root: Path) -> list[dict[str, Any]]:
    """decide_shion_candidates.py --applied で承認済みの候補一覧を返す。"""
    candidates = annotate_status(load_shion_candidates())
    return [c for c in candidates if c.get("ledger_status") == "applied"]


def build_queue(
    candidates: list[dict[str, Any]],
    already_queued_ids: set[str],
    limit: int,
    root: Path,
) -> dict[str, Any]:
    pending = [c for c in candidates if str(c.get("id") or "") not in already_queued_ids]

    safe: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    maybe: list[dict[str, Any]] = []

    for candidate in pending:
        item = {
            "id": candidate.get("id"),
            "title": candidate.get("title", ""),
            "reason": candidate.get("reason", ""),
            "user_approved": True,
        }
        item = refresh_auto_fix_policy(item, root)
        is_manual, reason = is_blocked(item)
        if is_manual:
            blocked.append({"id": item.get("id"), "title": item.get("title"), "reason": reason})
        elif is_codex_safe(item):
            safe.append(item)
        else:
            maybe.append({"id": item.get("id"), "title": item.get("title")})

    queued = safe[:limit]

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "limit": limit,
        "pending_count": len(pending),
        "shion_auto_safe_count": len(safe),
        "shion_auto_maybe_count": len(maybe),
        "manual_or_blocked_count": len(blocked),
        "queued_count": len(queued),
        "status": "READY" if queued else "EMPTY",
        "items": [queue_item(item) for item in queued],
        "skipped_safe_ids": [item.get("id") for item in safe[limit:]],
        "manual_or_blocked": blocked,
        "maybe": maybe,
    }


def _log_shion_request_drafted(root: Path, queue: dict[str, Any], output_path: Path) -> None:
    """Agent Action Ledger への記録。失敗しても処理は止めない。"""
    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from api.shion_action_ledger import log_action

        log_action(
            "codex_request_drafted",
            summary=(
                f"紫苑発の自動実行キューを生成: {queue.get('queued_count', 0)}件 "
                f"(safe={queue.get('shion_auto_safe_count', 0)})"
            ),
            observed_sources=["dispatch_queue.jsonl(source=shion)"],
            risk_level="low",
            requires_user_approval=bool(queue.get("queued_count", 0)),
            target=str(output_path),
            result="drafted",
        )
    except Exception:
        pass


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None, help="Queue JSON path.")
    parser.add_argument(
        "--state-file", type=Path, default=root / "reports" / STATE_FILE_NAME,
        help="キュー済みIDの記録先（再送防止）",
    )
    parser.add_argument("--limit", type=int, default=1, help="1回の実行でキューに乗せる最大件数（既定1件/日）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    date_tag = dt.date.today().strftime("%Y%m%d")
    output_path = args.output or root / "reports" / f"codex_auto_queue_shion_{date_tag}.json"

    state = load_state(args.state_file)
    already_queued_ids = set(state.get("queued_ids") or [])

    candidates = applied_candidates(root)
    queue = build_queue(candidates, already_queued_ids, max(0, args.limit), root)

    if args.dry_run:
        print(json.dumps(queue, ensure_ascii=False, indent=2))
        return

    dump_json(output_path, queue)

    newly_queued_ids = [str(item.get("id") or "") for item in queue["items"] if item.get("id")]
    if newly_queued_ids:
        state["queued_ids"] = sorted(already_queued_ids | set(newly_queued_ids))
        dump_json(args.state_file, state)

    _log_shion_request_drafted(root, queue, output_path)
    print(
        "Shion auto queue: "
        f"{queue['queued_count']} queued / {queue['shion_auto_safe_count']} safe "
        f"({output_path})"
    )


if __name__ == "__main__":
    main()
