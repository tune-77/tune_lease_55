#!/usr/bin/env python3
"""既存の会話キーポイントに出所（content_source = user / shion / unknown）を付ける（REV-544）。

mind.json の conversation_keypoints は、対話室のやり取り（ユーザー発言＋紫苑の返答）から抜き出した要点。
紫苑自身の発言由来の要点が「以前教わった〜」として次の回答に戻っていたため、出所を付けて区別する。
要点の中身は消さない・変えない。content_source が無いものに付けるだけ。

判定（api/keypoint_source.py と同じ。AI 呼び出しなし）:
- 内省の学び（session_id=private_reflection_feedback_loop）は shion
- 対話の要点は、同じ日（日本時間）の対話室のやり取り（lease_data.db の chat_messages）全体と突き合わせ、
  文字の並びがユーザーの発言に多く含まれれば user、紫苑の返答にだけ多く含まれれば shion
- やり取りが見つからない・決めきれないものは unknown（不明）

既定は確認のみ（--apply で書き込む。書き込む前に mind.json をバックアップする）。
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.keypoint_source import SHION, UNKNOWN, _bigrams, _coverage, keypoint_content_source  # noqa: E402
from runtime_paths import get_data_path, get_db_path, resolve_obsidian_vault  # noqa: E402

REFLECTION_SESSION = "private_reflection_feedback_loop"
JST = dt.timezone(dt.timedelta(hours=9))


def _dialogue_pairs(db_path: str, session_id: str) -> dict[str, list[tuple[str, str]]]:
    """日本時間の日付ごとの（ユーザー発言, 紫苑の返答）の組。"""
    pairs: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM chat_messages WHERE user_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    for (role_a, text_a, ts), (role_b, text_b, _ts) in zip(rows, rows[1:]):
        if role_a != "user" or role_b != "assistant":
            continue
        try:
            stamp = dt.datetime.fromisoformat(str(ts).replace(" ", "T")).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        pairs[stamp.astimezone(JST).date().isoformat()].append((str(text_a), str(text_b)))
    return pairs


def classify(keypoint: dict[str, Any], pairs: dict[str, list[tuple[str, str]]]) -> str:
    if str(keypoint.get("session_id") or "") == REFLECTION_SESSION:
        return SHION
    candidates = pairs.get(str(keypoint.get("date") or ""), [])
    if not candidates:
        return UNKNOWN
    content = str(keypoint.get("content") or "")
    grams = _bigrams(content)
    # 要点の言葉が最も多く含まれるユーザー発言・紫苑の返答を、その日のやり取り全体から選ぶ
    # （ユーザーが前の発言で述べた考えを、紫苑が「判断資産に入れて」の返答でまとめた場合も拾う）
    user_message = max((pair[0] for pair in candidates), key=lambda text: _coverage(grams, text))
    reply = max((pair[1] for pair in candidates), key=lambda text: _coverage(grams, text))
    return keypoint_content_source(content, user_message, reply)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vault", default="", help="Obsidian Vault（既定は runtime_paths の解決結果）")
    parser.add_argument("--db", default="", help="chat_messages のある DB（既定は lease_data.db）")
    parser.add_argument("--apply", action="store_true", help="mind.json に書き込む（既定は確認のみ）")
    args = parser.parse_args(argv)

    from lease_intelligence_mind import _mind_locked, _write_state, load_lease_intelligence_mind, mind_directory

    vault = Path(args.vault) if args.vault else resolve_obsidian_vault()
    pairs = _dialogue_pairs(args.db or get_db_path(), "lease-intelligence-dialogue")
    with _mind_locked(vault):
        state = load_lease_intelligence_mind(vault)
        keypoints = [kp for kp in state.get("conversation_keypoints") or [] if isinstance(kp, dict)]
        targets = [kp for kp in keypoints if not kp.get("content_source")]
        decided = [(kp, classify(kp, pairs)) for kp in targets]
        counts = collections.Counter(source for _kp, source in decided)
        print(json.dumps({"keypoints": len(keypoints), "without_source": len(targets), **counts}, ensure_ascii=False))
        for kp, source in decided[-12:]:
            print(f"  {source:7} {kp.get('date')} {str(kp.get('content'))[:60]}")
        if not args.apply or not decided:
            print("確認のみ（--apply で書き込む）" if not args.apply else "付ける対象なし")
            return 0
        backup_dir = Path(get_data_path("backups", "keypoint_content_source"))
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"mind_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy2(mind_directory(vault) / "mind.json", backup)
        for kp, source in decided:
            kp["content_source"] = source
        _write_state(vault, state)
        print(f"書き込み: {len(decided)}件・バックアップ: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
