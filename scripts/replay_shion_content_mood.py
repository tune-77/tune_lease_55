#!/usr/bin/env python3
"""REV-599: 本人の過去の対話室の会話で、話の内容による気分・関係性の動きを再生する（反映前の確認用）。

- 会話履歴（chat_messages）の対話室（lease-intelligence-dialogue）の往復だけ（気分が動くのは対話室の経路のため）。
  検証の会話（:verification）は除く
- 分類は本番と同じ api.shion_content_mood.classify_content（Gemini flash-lite）。結果は --cache に保存し、
  同じ往復は二度呼ばない。--offline ならキャッシュにあるものだけ使う（呼び出しなし）
- 気分は REV-598 までの決まり（dialogue_mood_causes）に、話の内容の原因（without_affect_overlap 済み）を足して、
  揺れの戻り（1ターン2割・毎日半分）と1回の幅の上限（3）で再生する。基調は今の値で固定
- 出力: 日ごとの気分（内容あり／なし）、1往復ごとの最大の動き、分類の内訳、事務的な往復での動き

本物の Gemini を呼ぶため、手動で動かす時は AI_LIVE_VERIFY=1 が要る（検証の区分・1日の上限内）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_paths import get_db_path  # noqa: E402

START = datetime(2026, 9, 20, 0, 50)
JST = timedelta(hours=9)
USER = "lease-intelligence-dialogue"
AXES = ("weariness", "curiosity", "attachment", "vigilance", "hope", "frustration", "loneliness", "accomplishment")


def load_pairs(db_path: Path, since: datetime) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        "select role, content, created_at from chat_messages where user_id=? and created_at >= ? order by created_at, id",
        (USER, since.isoformat(sep=" ")),
    ).fetchall()
    pairs: list[dict[str, Any]] = []
    for role, content, created in rows:
        if role == "user":
            pairs.append({"ts": datetime.fromisoformat(str(created)), "user": str(content or ""), "reply": ""})
        elif pairs and not pairs[-1]["reply"]:
            pairs[-1]["reply"] = str(content or "")
    return pairs


def _key(pair: dict[str, Any]) -> str:
    return hashlib.sha256(f"{pair['ts'].isoformat()}|{pair['user']}".encode()).hexdigest()[:16]


BATCH = 10


def classify_all(pairs, cache_path: Path, *, offline: bool, caller=None) -> dict[str, Any]:
    """短い発言は本番と同じく呼ばずに small_talk。残りは同じ分類の定義で10往復ずつまとめて分類する
    （本番は1往復ずつ。検証の呼び出し上限 1日30回に収めるため）。"""
    from api.shion_content_mood import MIN_CHARS, classify_batch

    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = []
    for pair in pairs:
        key = _key(pair)
        if key in cache:
            continue
        if len(" ".join(pair["user"].split())) < MIN_CHARS:
            cache[key] = {"category": "small_talk", "intensity": 0.0, "reason": "短い発言", "called": False}
        elif not offline:
            todo.append(pair)
    for start in range(0, len(todo), BATCH):
        chunk = todo[start:start + BATCH]
        results = classify_batch([(p["user"], p["reply"]) for p in chunk], caller=caller)
        for pair, result in zip(chunk, results):
            if result is not None:
                cache[_key(pair)] = result
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return cache


def replay(pairs, cache, *, mood_base: dict[str, int]) -> dict[str, Any]:
    from api.shion_content_mood import content_mood_causes, content_relationship_parts, without_affect_overlap
    from api.user_affect import estimate_user_affect, relationship_feedback_from_affect
    from lease_intelligence_mind import DIALOGUE_MOOD_CAP, DIALOGUE_MOOD_DECAY, MOOD_STEP_LIMIT, dialogue_mood_causes

    tracks = {name: {"adj": {a: 0 for a in AXES}, "mood": dict(mood_base)} for name in ("with", "without")}
    daily: dict[str, dict[str, Any]] = {}
    per_turn_max: list[int] = []
    business_moves = 0
    business_turns = 0
    categories: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    rel_sum = 0.0
    turns: list[dict[str, Any]] = []

    def settle(track, causes):
        track["adj"] = {a: int(v * DIALOGUE_MOOD_DECAY) for a, v in track["adj"].items()}
        for c in causes:
            track["adj"][c["axis"]] = max(-DIALOGUE_MOOD_CAP, min(DIALOGUE_MOOD_CAP, track["adj"][c["axis"]] + int(c["delta"])))
        before = dict(track["mood"])
        for a in AXES:
            goal = max(0, min(100, mood_base[a] + track["adj"][a]))
            track["mood"][a] += max(-MOOD_STEP_LIMIT, min(MOOD_STEP_LIMIT, goal - track["mood"][a]))
        return max(abs(track["mood"][a] - before[a]) for a in AXES)

    last_day = None
    last_ts = None
    for pair in pairs:
        day = (pair["ts"] + JST - timedelta(hours=4, minutes=5)).date().isoformat()  # 04:05 で日替わり
        if last_day and day != last_day:
            for track in tracks.values():
                track["adj"] = {a: int(v / 2) for a, v in track["adj"].items()}
        last_day = day
        affect = estimate_user_affect(pair["user"]).to_payload()
        gap = None if last_ts is None else (pair["ts"] - last_ts).total_seconds() / 3600
        last_ts = pair["ts"]
        rel_info = {"feedback": relationship_feedback_from_affect(affect["label"], affect["cues"], affect["signals"]),
                    "trend": "stable", "silence_hours": gap, "prior_negative_streak": 0}
        base = dialogue_mood_causes(pair["user"], {"affect": affect, "relationship": rel_info})
        result = cache.get(_key(pair))
        content = without_affect_overlap(content_mood_causes(result), [c for c in base if c["rule"] == "user_affect"])
        rel_sum += sum(d for _r, d in content_relationship_parts(result))
        moved = settle(tracks["with"], base + content)
        settle(tracks["without"], base)
        moved_by_content = max((abs(c["delta"]) for c in content), default=0)
        per_turn_max.append(moved)
        category = (result or {}).get("category") or "unclassified"
        categories[category] += 1
        if category in {"business", "small_talk", "unclassified"}:
            business_turns += 1
            business_moves += int(moved_by_content > 0)
        elif len(examples.setdefault(category, [])) < 3:
            examples[category].append(f"{(pair['ts'] + JST):%m/%d} {(result or {}).get('reason', '')}")
        daily[day] = {"with": dict(tracks["with"]["mood"]), "without": dict(tracks["without"]["mood"])}
        turns.append({"ts": (pair["ts"] + JST).isoformat(timespec="minutes"), "category": category,
                      "with": dict(tracks["with"]["mood"]), "without": dict(tracks["without"]["mood"])})
    return {
        "pairs": len(pairs),
        "categories": dict(categories),
        "examples": examples,
        "max_move_per_turn": max(per_turn_max, default=0),
        "turns_moving_3": sum(1 for m in per_turn_max if m >= 3),
        "business_turns": business_turns,
        "business_turns_moved_by_content": business_moves,
        "relationship_from_content_total": round(rel_sum, 3),
        "daily": daily,
        "turns": turns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mind", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    pairs = load_pairs(Path(get_db_path()), START)
    cache = classify_all(pairs, args.cache, offline=args.offline)
    mind = json.loads(args.mind.read_text(encoding="utf-8"))
    mood_base = {a: int((mind.get("mood_base") or mind.get("mood") or {}).get(a, 50)) for a in AXES}
    result = replay(pairs, cache, mood_base=mood_base)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("daily", "turns")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
