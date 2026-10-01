#!/usr/bin/env python3
"""過去のチャットで教わった審査ノウハウを、判断資産候補（要確認）として救済する。

2026-10 の調査で、対話室の1,371発言からの判断資産候補が0件だった（旧判定が狭すぎた）。
このスクリプトは過去ログを ``memory_promotion_policy.classify_lease_teaching`` で
判定し直し、該当した発言を /judgment-review の要確認に入れる。自動昇格はしない。

入力（存在するものだけ読む）:
* Obsidian ``Lease Intelligence/Dialogue/*.md`` の **ユーザー** 発言
* ``data/cloudrun_chat_log.jsonl`` の ``user_message``
* ``data/lease_data.db`` の ``chat_messages``（role=user）

既定は件数と代表例を出すだけ（dry run）。``--apply`` で候補を書き込む。
同じ文面は既存候補・他ソースと重複させない。

    python3 scripts/rescue_chat_teaching_candidates.py
    python3 scripts/rescue_chat_teaching_candidates.py --apply
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_promotion_policy import classify_lease_teaching  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
_USER_BLOCK_RE = re.compile(r"\*\*ユーザー\*\*\s*\n(.*?)\n\*\*リース知性体\*\*", re.S)


def _dialogue_note_messages(dialogue_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(dialogue_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in _USER_BLOCK_RE.finditer(text):
            rows.append({"date": path.stem[:10], "text": match.group(1).strip(), "source": "obsidian_dialogue"})
    return rows


def _cloudrun_messages(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = str(row.get("user_message") or "").strip()
        if text:
            rows.append({"date": str(row.get("ts") or "")[:10], "text": text, "source": "cloudrun_chat_log"})
    return rows


def _db_messages(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with sqlite3.connect(path) as conn:
            cur = conn.execute(
                "SELECT content, created_at FROM chat_messages WHERE role='user' ORDER BY created_at, id"
            )
            return [
                {"date": str(created or "")[:10], "text": str(content or "").strip(), "source": "chat_messages_db"}
                for content, created in cur.fetchall()
            ]
    except sqlite3.Error:
        return []


def collect_messages(vault: Path | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if vault is not None:
        from lease_intelligence_mind import mind_directory

        dialogue_dir = mind_directory(vault) / "Dialogue"
        if dialogue_dir.exists():
            messages.extend(_dialogue_note_messages(dialogue_dir))
    messages.extend(_cloudrun_messages(DATA_DIR / "cloudrun_chat_log.jsonl"))
    messages.extend(_db_messages(DATA_DIR / "lease_data.db"))
    return messages


def find_teachings(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    """各ソース内の並び順で判定し、指示だけの発言は直前のユーザー発言を本文にする。"""
    from api.chat_judgment_asset_capture import chat_judgment_asset_candidate_type
    from api.chat_teaching_capture import resolve_teaching_claim

    found: dict[str, dict[str, Any]] = {}
    previous_by_source: dict[str, str] = {}
    for message in messages:
        text = message["text"]
        source = message["source"]
        is_teaching, reason = classify_lease_teaching(text)
        if is_teaching:
            claim = resolve_teaching_claim(text, previous_by_source.get(source, ""))
            key = re.sub(r"\s+", "", claim)
            if key not in found:
                found[key] = {
                    "claim": claim,
                    "date": message["date"],
                    "source": source,
                    "reason": reason,
                    "candidate_type": chat_judgment_asset_candidate_type(claim),
                }
        previous_by_source[source] = text
    return sorted(found.values(), key=lambda item: item["date"])


def apply_candidates(teachings: list[dict[str, Any]]) -> Counter[str]:
    from api.chat_judgment_asset_capture import (
        CHAT_TEACHING_TOPIC,
        capture_chat_judgment_asset_if_needed,
        create_manual_judgment_asset_candidate,
        load_autoresearch_judgment_asset_candidates,
    )

    candidates_jsonl = DATA_DIR / "autoresearch_judgment_asset_candidates.jsonl"
    state_json = DATA_DIR / "autoresearch_judgment_asset_candidate_state.json"
    outcome: Counter[str] = Counter()
    for item in teachings:
        result = capture_chat_judgment_asset_if_needed(
            item["claim"],
            user_id="chat_teaching_rescue",
            surface="chat_teaching_rescue",
            response_mode="rescue",
            candidates_loader=lambda limit=1000: load_autoresearch_judgment_asset_candidates(
                candidates_jsonl=candidates_jsonl, candidate_state_json=state_json, limit=limit
            ),
            candidate_creator=lambda req: create_manual_judgment_asset_candidate(
                req, candidates_jsonl=candidates_jsonl, candidate_state_json=state_json
            ),
            request_factory=lambda **kwargs: SimpleNamespace(
                review_id=None, research_date=item["date"], **{**kwargs, "research_topic": CHAT_TEACHING_TOPIC}
            ),
            cloudrun_event_recorder=lambda **_kwargs: {"status": "skipped_local_rescue"},
        )
        if not result.get("captured"):
            outcome["failed"] += 1
        elif result.get("duplicate"):
            outcome["duplicate"] += 1
        else:
            outcome["created"] += 1
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="候補を書き込む（既定は件数表示のみ）")
    parser.add_argument("--show", type=int, default=10, help="表示する代表例の件数")
    args = parser.parse_args()

    try:
        from lease_news_digest import find_vault

        vault = find_vault()
    except Exception:  # noqa: BLE001
        vault = None
    messages = collect_messages(Path(vault) if vault else None)
    teachings = find_teachings(messages)
    summary: dict[str, Any] = {
        "messages": len(messages),
        "by_source": dict(Counter(m["source"] for m in messages)),
        "teachings": len(teachings),
        "by_reason": dict(Counter(t["reason"] for t in teachings)),
        "by_type": dict(Counter(t["candidate_type"] for t in teachings)),
        "examples": [f"{t['date']} {t['claim'][:80]}" for t in teachings[: max(0, args.show)]],
    }
    if args.apply:
        summary["applied"] = dict(apply_candidates(teachings))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
