#!/usr/bin/env python3
"""REV-598: 関係性スコアと気分（不満・孤独など）を、過去の会話の記録で新しい決まりのとおり再生する。

関係性スコアは 9/20 の 7.0 から、対話ごとに必ず上がる決まりで上限 10.0 に張り付いていた。新しい決まり
（api/shion_relationship・lease_intelligence_mind.reaction_mood_causes）で、状態を作った 9/20 からの本人の会話を
再生し、今の値を一度だけ置き換える。根拠（再生した会話の数・期間・結果）は状態ファイルの migration に残す。

材料（読むだけ）:
- 会話履歴（chat_messages）: 対話室（lease-intelligence-dialogue）・/api/chat（shion-default・default）のユーザー発言。
  検証の会話（user_id が :verification で終わる・検証用ID）・めぶき・比較用・審査レビュー依頼は除く
- 予想の答え合わせ（shion_prediction_log.jsonl）: origin=verification を除く
- 毎日 04:05（JST）の無交流ペナルティと中立への戻り、話しかけられない日の孤独
気分は本番と同じ dialogue_mood_causes で原因を作り、揺れ（dialogue_mood）を1ターンごとに2割・毎日半分戻す。
基調（記憶の言葉から作る mood_base）は今の値で固定する（再生では記憶を作り直さないため）。

使い方:
  python scripts/migrate_shion_relationship_v2.py --out series.json            # 再生だけ（書き込みなし）
  python scripts/migrate_shion_relationship_v2.py --apply                      # バックアップして関係性スコアを置き換える
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_paths import get_data_dir, get_db_path  # noqa: E402

START = datetime(2026, 9, 20, 0, 50)  # 状態ファイルを作った時刻（UTC）
JST = timedelta(hours=9)
REAL_USERS = ("lease-intelligence-dialogue", "shion-default", "default")
SKIP_PREFIXES = ("[審査分析の紫苑レビュー依頼",)
MOOD_AXES = ("weariness", "curiosity", "attachment", "vigilance", "hope", "frustration", "loneliness", "accomplishment")


def load_turns(db_path: Path, since: datetime) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        "select user_id, content, created_at from chat_messages where role='user' and created_at >= ? order by created_at",
        (since.isoformat(sep=" "),),
    ).fetchall()
    turns = []
    for user_id, content, created in rows:
        if user_id not in REAL_USERS or str(content or "").startswith(SKIP_PREFIXES):
            continue
        ts = datetime.fromisoformat(str(created))
        turns.append({"ts": ts, "user_id": user_id, "message": str(content or "")})
    return turns


def load_predictions(path: Path, since: datetime) -> list[dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("origin") == "verification":
            continue
        ts = datetime.fromisoformat(str(row.get("at"))) - JST  # ログは JST の時刻
        if ts >= since:
            out.append({**row, "ts": ts})
    return sorted(out, key=lambda r: r["ts"])


def _old_rule(score: float, feedback: str, depth: str) -> float:
    """REV-220 の旧ルール（比較用）。"""
    delta = 0.05 + {"positive": 0.2, "negative": -0.3}.get(feedback, 0.0)
    delta += {"deep": 0.1, "shallow": -0.02}.get(depth, 0.0)
    return min(10.0, max(0.0, score + delta))


def replay(turns, predictions, *, mood_base: dict[str, int], end: datetime) -> dict[str, Any]:
    import api.shion_relationship as rel
    from api.chat_routing import chat_context_mode
    from api.game_theory.dialogue import select_dialogue_strategy
    from api.user_affect import estimate_user_affect, relationship_feedback_from_affect
    from lease_intelligence_mind import (
        DIALOGUE_MOOD_CAP,
        DIALOGUE_MOOD_DECAY,
        MOOD_STEP_LIMIT,
        dialogue_mood_causes,
        reaction_mood_causes,
        silence_mood_causes,
    )

    state = {"score": rel._SCORE_INITIAL, "trend": "stable", "delta_history": [], "negative_streak": 0,
             "total_interactions": 0, "last_interaction": START.isoformat(), "reverted_at": START.isoformat(),
             "events": [], "score_history": {}}
    old_score = rel._SCORE_INITIAL
    old_new = {"adj": {a: 0 for a in MOOD_AXES}, "mood": dict(mood_base)}
    new = {"adj": {a: 0 for a in MOOD_AXES}, "mood": dict(mood_base)}
    daily: dict[str, dict[str, Any]] = {}
    fear = {"low": 0, "falling": 0, "turns": 0, "strategies": {}}
    preds = list(predictions)

    def settle(track, causes, decay=True):
        if decay:
            track["adj"] = {a: int(v * DIALOGUE_MOOD_DECAY) for a, v in track["adj"].items()}
        for c in causes:
            if c["axis"] in track["adj"]:
                track["adj"][c["axis"]] = max(-DIALOGUE_MOOD_CAP, min(DIALOGUE_MOOD_CAP, track["adj"][c["axis"]] + int(c["delta"])))
        for a in MOOD_AXES:
            goal = max(0, min(100, mood_base[a] + track["adj"][a]))
            track["mood"][a] += max(-MOOD_STEP_LIMIT, min(MOOD_STEP_LIMIT, goal - track["mood"][a]))

    def snapshot(day: str):
        state["score_history"][day] = state["score"]
        daily[day] = {"score_new": state["score"], "score_old": round(old_score, 3),
                      "mood_new": dict(new["mood"]), "mood_old": dict(old_new["mood"])}

    def nightly(at: datetime):
        # 04:05 JST の無交流ペナルティ・中立への戻り・孤独。揺れは日替わりで半減
        rel._apply_reversion(state, at)
        last = rel._parse(state["last_interaction"])
        days = (at - last).total_seconds() / 86400
        if days > rel._INACTIVITY_GRACE_DAYS:
            pen = rel._INACTIVITY_PENALTY_PER_DAY * min(1.0, days - rel._INACTIVITY_GRACE_DAYS)
            state["score"] = round(max(0.0, state["score"] - pen), 3)
            state["delta_history"] = (state["delta_history"] + [-pen])[-10:]
            state["trend"] = rel._calc_trend(state["delta_history"])
        for track in (old_new, new):
            track["adj"] = {a: int(v / 2) for a, v in track["adj"].items()}
        settle(new, silence_mood_causes(days), decay=False)
        settle(old_new, [], decay=False)

    next_night = (START + JST).replace(hour=4, minute=5, second=0, microsecond=0) - JST
    if next_night <= START:
        next_night += timedelta(days=1)
    for turn in turns + [{"ts": end, "end": True}]:
        while next_night <= turn["ts"]:
            nightly(next_night)
            snapshot((next_night + JST).date().isoformat())
            next_night += timedelta(days=1)
        while preds and preds[0]["ts"] <= turn["ts"]:
            p = preds.pop(0)
            parts = []
            if p.get("reaction_hit") is False:
                parts.append(("返答への不満が見えた（予想の答え合わせ）", rel.DELTA_REACTION_MISS))
            elif p.get("affect_hit") and p.get("expected_affect") not in ("通常", ""):
                parts.append(("相手の様子の予想が当たった", rel.DELTA_PREDICTION_HIT))
            if parts:
                state["score"], d = rel.apply_parts(state["score"], parts)
                state["delta_history"] = (state["delta_history"] + [d])[-10:]
        if turn.get("end"):
            break
        now = turn["ts"]
        message = turn["message"]
        affect = estimate_user_affect(message).to_payload()
        feedback = relationship_feedback_from_affect(affect["label"], affect["cues"], affect["signals"])
        mode = chat_context_mode(message, "")
        depth = {"screening": "deep", "deep": "deep", "casual": "shallow"}.get(mode, "normal")
        # 旧ルール（比較用。様子の喜びだけ positive、紫苑への苛立ちだけ negative）
        old_feedback = "positive" if affect["label"] == "喜び" else ("negative" if "shion_complaint" in affect["signals"] else "neutral")
        old_score = _old_rule(old_score, old_feedback, depth)
        # 新ルール
        rel._apply_reversion(state, now)
        last = rel._parse(state["last_interaction"])
        gap = None if not state["total_interactions"] else (now - last).total_seconds() / 3600
        prior_streak = state["negative_streak"]
        parts = rel.interaction_parts(gap_hours=gap, feedback_type=feedback, topic_depth=depth,
                                      signals=affect["signals"], negative_streak=prior_streak)
        state["negative_streak"] = prior_streak + 1 if rel._is_negative(parts) else 0
        state["score"], d = rel.apply_parts(state["score"], parts)
        state["delta_history"] = (state["delta_history"] + [d])[-10:]
        state["trend"] = rel._calc_trend(state["delta_history"])
        state["last_interaction"] = now.isoformat()
        state["total_interactions"] += 1
        # 気分（本番と同じ原因の作り方。新は反応・久しぶりの会話を含む）
        rel_info = {"feedback": feedback, "trend": state["trend"], "silence_hours": gap,
                    "prior_negative_streak": prior_streak}
        signals = {"affect": affect, "relationship": rel_info}
        causes_new = dialogue_mood_causes(message, signals)
        reaction = reaction_mood_causes(affect["signals"], rel_info)
        causes_old = [c for c in causes_new if c not in reaction]
        settle(new, causes_new)
        settle(old_new, causes_old)
        # 恐れの行動変化（lease_intelligence_dialogue._build_fear_state_prompt_block と同じ条件）
        fear["turns"] += 1
        is_low, is_falling = rel.fear_flags(state["score"], state["trend"], rel.recent_peak(state, now))
        if is_low:
            fear["low"] += 1
        elif is_falling:
            fear["falling"] += 1
            fear.setdefault("falling_at", []).append(f"{(now + JST):%m-%d %H:%M}")
        old_low, old_falling = state["score"] < 4.0, state["score"] < 5.0 and state["trend"] == "falling"
        fear["old_rule_fired"] = fear.get("old_rule_fired", 0) + int(old_low or old_falling)
        days_since = (gap or 0) / 24
        strategy = select_dialogue_strategy(state["score"], state["trend"], state["negative_streak"],
                                            days_since_last=days_since, recent_feedback=feedback)["strategy"]
        fear["strategies"][strategy] = fear["strategies"].get(strategy, 0) + 1
    return {"state": state, "old_score": round(old_score, 3), "daily": daily, "fear": fear,
            "mood_new": new["mood"], "mood_old": old_new["mood"], "turns": len(turns)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="日ごとの推移を JSON で書き出す")
    parser.add_argument("--apply", action="store_true", help="バックアップして関係性スコアを置き換える")
    parser.add_argument("--mind", type=Path, default=None, help="mind.json（気分の基調を読む）")
    args = parser.parse_args()

    data_dir = get_data_dir()
    end = datetime.utcnow()
    turns = load_turns(Path(get_db_path()), START)
    predictions = load_predictions(data_dir / "shion_prediction_log.jsonl", START)
    mood_base = {a: 50 for a in MOOD_AXES}
    mind_path = args.mind
    if mind_path is None:
        from runtime_paths import resolve_obsidian_vault

        vault = resolve_obsidian_vault()
        mind_path = Path(vault) / "Projects/tune_lease_55/Lease Intelligence/mind.json" if vault else None
    if mind_path and mind_path.exists():
        mind = json.loads(mind_path.read_text(encoding="utf-8"))
        mood_base = {a: int((mind.get("mood_base") or mind.get("mood") or {}).get(a, 50)) for a in MOOD_AXES}
    result = replay(turns, predictions, mood_base=mood_base, end=end)
    summary = {"turns": result["turns"], "predictions": len(predictions), "new_score": result["state"]["score"],
               "old_rule_score": result["old_score"], "fear": result["fear"],
               "mood_new": result["mood_new"], "mood_old": result["mood_old"]}
    print(json.dumps(summary, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps({**summary, "daily": result["daily"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.apply:
        return 0

    import api.shion_relationship as rel

    path = rel._STATE_PATH
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else rel._default_state()
    if int(current.get("schema_version", 1)) >= rel.SCHEMA_VERSION and current.get("migration"):
        print("移行済みのため何もしない")
        return 0
    backup = data_dir / "backups" / "shion_relationship" / f"state_{end:%Y%m%d_%H%M%S}.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    replayed = result["state"]
    current.update({
        "score": replayed["score"],
        "trend": replayed["trend"],
        "delta_history": replayed["delta_history"],
        "negative_streak": replayed["negative_streak"],
        "reverted_at": end.isoformat(),
        "schema_version": rel.SCHEMA_VERSION,
        "events": [],
        "score_history": {day: row["score_new"] for day, row in sorted(result["daily"].items())},
        "migration": {
            "rev": "REV-598",
            "at": end.isoformat(timespec="seconds"),
            "from_score": current.get("score"),
            "to_score": replayed["score"],
            "basis": (f"状態を作った {START:%Y-%m-%d} の初期値 7.0 から、本人の会話 {result['turns']} 件と予想の答え合わせ "
                      f"{len(predictions)} 件を新しい決まりで再生した値（検証の会話は除外）。旧ルールでは同じ会話で "
                      f"{result['old_score']} になり上限に張り付いていた"),
            "backup": str(backup),
        },
    })
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    print(f"移行: {current['migration']['from_score']} → {replayed['score']}（バックアップ: {backup}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
