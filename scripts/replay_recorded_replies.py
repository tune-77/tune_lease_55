#!/usr/bin/env python3
"""録画済みの返答で後処理を検証する（Gemini を呼ばない再生検証の既定経路, REV-485）。

返答の後処理（依頼文の照合・感情の主張抽出など）を確かめる時、本物の対話経路
（1回 約1.9万トークン）で返答を作り直さず、chat_messages に保存済みの
（ユーザー発話, 紫苑の返答）の組を読み取り専用で再生する。
採点は既定で Jev（Gemini 課金外）。件数は既定 20・上限 100。

本物の対話経路での検証がどうしても必要な時だけ、AI_LIVE_VERIFY=1 を付けて
別スクリプトで実行する（ai_budget の検証クラス上限 AI_VERIFY_MAX_CALLS が効く）。

例:
  python scripts/replay_recorded_replies.py --processor request_grounding
  python scripts/replay_recorded_replies.py --processor emotion_claims --limit 50 --judge none
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_runtime_client import _MAIN_ROOT  # noqa: E402  worktree からでも本体の DB を読む

MAX_LIMIT = 100
DEFAULT_USER_ID = "lease-intelligence-dialogue"


def load_pairs(db_path: Path, user_id: str, limit: int) -> list[tuple[str, str]]:
    """直近の（ユーザー発話, 直後の返答）を古い順で返す。DB は読み取り専用で開く。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT role, content FROM chat_messages WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (user_id, limit * 2 + 1),
        ).fetchall()
    finally:
        conn.close()
    rows.reverse()
    pairs = [
        (str(rows[i][1] or ""), str(rows[i + 1][1] or ""))
        for i in range(len(rows) - 1)
        if rows[i][0] == "user" and rows[i + 1][0] == "assistant"
    ]
    return pairs[-limit:]


def _request_grounding(message: str, reply: str) -> dict[str, Any]:
    from api.shion_request_grounding import ground_reply

    result = ground_reply(message, reply)
    return {"changed": result.changed, "removed": len(result.removed_promises), "missing_refs": len(result.missing_refs)}


def _request_grounding_jev(message: str, reply: str, processed: dict[str, Any]) -> dict[str, Any]:
    from api.shion_request_grounding import verify_reply_sentences

    jev = verify_reply_sentences(reply)
    flagged = len(jev.get("unexecutable_promises") or [])
    # Jev が実行できない約束を見つけたのに決定的処理が何も除かなかった＝取りこぼし候補
    return {"jev_status": jev.get("status"), "jev_flagged": flagged, "missed": bool(flagged and not processed["removed"])}


def _emotion_claims(_message: str, reply: str) -> dict[str, Any]:
    from api.shion_emotion_grounding import extract_state_claims

    return {"changed": False, "claims": len(extract_state_claims(reply))}


PROCESSORS: dict[str, tuple[Callable[[str, str], dict[str, Any]], Callable[..., dict[str, Any]] | None]] = {
    "request_grounding": (_request_grounding, _request_grounding_jev),
    "emotion_claims": (_emotion_claims, None),
}


def run(pairs: list[tuple[str, str]], processor: str, judge: str) -> dict[str, Any]:
    process, jev_judge = PROCESSORS[processor]
    totals: dict[str, int] = {"pairs": len(pairs), "changed": 0, "jev_judged": 0, "missed": 0, "errors": 0}
    for message, reply in pairs:
        try:
            processed = process(message, reply)
            totals["changed"] += int(bool(processed.get("changed")))
            if judge == "jev" and jev_judge is not None:
                judged = jev_judge(message, reply, processed)
                totals["jev_judged"] += int(judged.get("jev_status") == "applied")
                totals["missed"] += int(bool(judged.get("missed")))
        except Exception:
            totals["errors"] += 1
    totals["judge"] = judge if jev_judge is not None else "none（この処理器は Jev 採点なし）"
    return totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processor", choices=sorted(PROCESSORS), required=True)
    parser.add_argument("--limit", type=int, default=20, help=f"再生する組の数（上限 {MAX_LIMIT}）")
    parser.add_argument("--judge", choices=("jev", "none"), default="jev")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--db", type=Path, default=_MAIN_ROOT / "data" / "lease_data.db")
    args = parser.parse_args(argv)
    limit = max(1, min(args.limit, MAX_LIMIT))
    pairs = load_pairs(args.db, args.user_id, limit)
    # 件数と集計だけを出す（発話・返答の本文は出力しない）
    print(json.dumps({"processor": args.processor, "limit": limit, **run(pairs, args.processor, args.judge)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
