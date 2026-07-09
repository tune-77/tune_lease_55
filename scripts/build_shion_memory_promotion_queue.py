#!/usr/bin/env python3
"""会話ログから長期記憶への昇格候補キューを生成する（human-in-the-loop）。

Cloud Run 会話ログ（data/cloudrun_chat_log.jsonl、夜間同期で蓄積）から
「長期記憶に昇格させる価値がありそうな発話」を抽出して承認キューに出す:

1. 教示発話 — memory_promotion_policy の TEACHING_PATTERNS（「覚えておいて」等）
   を含み、質問ではないユーザー発話
2. 反復話題 — 直近ウィンドウ内で3回以上別メッセージに現れるドメイン語を含む発話
   （ユーザーが繰り返し気にしている = 判断関心の兆候）

キューは自動では記憶にならない。ユーザーが確認して
scripts/apply_shion_memory_promotions.py で承認したものだけが
knowledge_base/shion_promoted_memories.md へ追記され、次回のインデックス
再構築で記憶になる。雑談断片が判断原則へ化ける誤学習を防ぐための設計。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from memory_promotion_policy import (  # noqa: E402
    TEACHING_PATTERNS,
    has_domain_keyword,
    is_question,
)

DEFAULT_CHAT_LOG = REPO_ROOT / "data" / "cloudrun_chat_log.jsonl"
DEFAULT_APPLIED_LOG = REPO_ROOT / "data" / "shion_memory_promotions.jsonl"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports" / "shion_memory_promotion_queue_latest.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "shion_memory_promotion_queue_latest.md"

_TOPIC_RE = re.compile(r"[一-龥]{2,6}")
_REDACT_PATTERNS = (
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[email]"),
    (re.compile(r"\b0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{3,4}\b"), "[phone]"),
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _redact(text: str, limit: int = 220) -> str:
    value = " ".join(str(text or "").replace("\n", " ").split())
    for pattern, repl in _REDACT_PATTERNS:
        value = pattern.sub(repl, value)
    return value[:limit]


def _candidate_id(content: str) -> str:
    return "promo_" + hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]


def _applied_ids(applied_log: Path) -> set[str]:
    return {
        str(row.get("candidate_id") or "")
        for row in _load_jsonl(applied_log)
        if row.get("candidate_id")
    }


def collect_candidates(
    chat_rows: list[dict],
    applied_ids: set[str],
    *,
    min_topic_repeat: int = 3,
    limit: int = 15,
) -> list[dict]:
    user_messages = [
        (str(row.get("ts") or ""), str(row.get("user_message") or "").strip(), str(row.get("event_id") or ""))
        for row in chat_rows
        if str(row.get("user_message") or "").strip()
    ]

    candidates: dict[str, dict] = {}

    # 1. 教示発話
    for ts, message, event_id in user_messages:
        if is_question(message):
            continue
        if not any(pattern in message for pattern in TEACHING_PATTERNS):
            continue
        content = _redact(message)
        if len(content) < 12:
            continue
        cid = _candidate_id(content)
        if cid in applied_ids or cid in candidates:
            continue
        candidates[cid] = {
            "candidate_id": cid,
            "kind": "teaching",
            "proposed_content": content,
            "source_event_ids": [event_id] if event_id else [],
            "ts": ts,
            "reason": "教示パターン（覚えて等）を含むユーザー発話",
        }

    # 2. 反復話題（ドメイン語を含むメッセージに限定）
    topic_messages: dict[str, list[tuple[str, str, str]]] = {}
    topic_counts: Counter = Counter()
    for ts, message, event_id in user_messages:
        if not has_domain_keyword(message):
            continue
        seen_in_message: set[str] = set()
        for topic in _TOPIC_RE.findall(message):
            if topic in seen_in_message:
                continue
            seen_in_message.add(topic)
            topic_counts[topic] += 1
            topic_messages.setdefault(topic, []).append((ts, message, event_id))
    for topic, count in topic_counts.most_common():
        if count < min_topic_repeat:
            break
        entries = topic_messages[topic]
        ts, message, event_id = entries[-1]  # 最新の発話を代表にする
        content = _redact(f"Userは「{topic}」を繰り返し気にしている。直近の文脈: {message}")
        cid = _candidate_id(f"topic:{topic}")
        if cid in applied_ids or cid in candidates:
            continue
        candidates[cid] = {
            "candidate_id": cid,
            "kind": "recurring_topic",
            "topic": topic,
            "proposed_content": content,
            "source_event_ids": [e for _, _, e in entries[-3:] if e],
            "ts": ts,
            "reason": f"直近の会話で {count} 回言及された話題",
        }

    ordered = sorted(
        candidates.values(), key=lambda c: (c["kind"] != "teaching", c.get("ts") or "")
    )
    return ordered[:limit]


def _render_markdown(candidates: list[dict]) -> str:
    lines = [
        "# 紫苑記憶 昇格候補キュー",
        "",
        f"- 生成: {datetime.now().isoformat(timespec='seconds')}",
        f"- 候補: {len(candidates)} 件",
        "- 承認方法: `python3 scripts/apply_shion_memory_promotions.py --ids <ID,...>`",
        "  （承認分だけ knowledge_base/shion_promoted_memories.md へ追記され、",
        "  次回のインデックス再構築で記憶になる。自動昇格はしない）",
        "",
    ]
    if not candidates:
        lines.append("候補はありません。")
        return "\n".join(lines) + "\n"
    for c in candidates:
        lines += [
            f"## `{c['candidate_id']}`（{c['kind']}）",
            f"- 提案内容: {c['proposed_content']}",
            f"- 根拠: {c['reason']}",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="会話ログから記憶昇格候補キューを生成")
    parser.add_argument("--chat-log", type=Path, default=DEFAULT_CHAT_LOG)
    parser.add_argument("--applied-log", type=Path, default=DEFAULT_APPLIED_LOG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--min-topic-repeat", type=int, default=3)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    candidates = collect_candidates(
        _load_jsonl(args.chat_log),
        _applied_ids(args.applied_log),
        min_topic_repeat=args.min_topic_repeat,
        limit=args.limit,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "candidates": candidates,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    args.output_md.write_text(_render_markdown(candidates), encoding="utf-8")
    print(f"昇格候補: {len(candidates)} 件 → {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
