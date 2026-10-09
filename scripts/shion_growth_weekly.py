#!/usr/bin/env python3
"""紫苑の成長を週1回、既存の記録から数字にして Obsidian に残す（REV-586）。

ここ数日入れた機能（予想と答え合わせ・自己報告の接地・丸写し防止・Private Reflection・
関係性・判断資産・ニュース永続メモ・費用）が紫苑を実際に育てているかを、AI を呼ばずに
既存のログ・状態ファイルだけから集計する。記憶系（mind.json・内省ノート・関係性の状態など）は
読むだけで変更しない。書くのは次の3つだけ。

- 週のノート: <out-dir>/成長記録/週次_<開始>〜<終了>.md（遡り集計は 遡り集計_<開始>〜<終了>.md）
- 成長記録の目次: <out-dir>/14_成長記録.md（週のノート一覧と週ごとの推移）。00_目次 には1行だけ足す
- 週ごとの数字の履歴: DATA_DIR/shion_growth_weekly_history.jsonl（関係性・気分は現在値しか
  残らないので、ここに週ごとの値を貯めて推移にする）

launchd: com.tunelease.shion-growth-weekly（日曜 08:30）。前の日曜〜土曜の7日分を集計する。
遡り: python scripts/shion_growth_weekly.py --from 2026-10-01 --to 2026-10-09
検証: DATA_DIR=<作業用> python scripts/shion_growth_weekly.py --source-dir <本番data> --out-dir <作業用>
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_paths import get_data_dir, get_data_path, resolve_obsidian_vault  # noqa: E402
from shion_verification_origin import is_verification_row  # noqa: E402

JST = timezone(timedelta(hours=9))
PROJECT_REL = Path("Projects") / "tune_lease_55"
OUT_REL = PROJECT_REL / "紫苑の仕組み_2026-10"
MIND_REL = PROJECT_REL / "Lease Intelligence" / "mind.json"
REFLECTION_REL = PROJECT_REL / "Lease Intelligence" / "Private Reflection"
HISTORY_NAME = "shion_growth_weekly_history.jsonl"
INDEX_NAME = "14_成長記録"
TOC_NAME = "00_目次_紫苑の仕組み_2026-10.md"
NOTE_DIR = "成長記録"
# REV-544（同じ質問への聞き直しの指示）が master に入った時刻。これ以降の聞き直しには指示が入った推定
REPEAT_GUARD_SINCE = datetime(2026, 10, 9, 19, 58, tzinfo=JST)
REPEAT_RATIO = 0.85  # api/answer_repeat_guard._REPEAT_RATIO と同じ
COPY_RATIO = 0.9  # 前回の答えとの類似度がこれ以上なら「ほぼ丸写し」
REPLY_WINDOW = timedelta(minutes=60)  # 紫苑の質問に、この時間内に次の発言があれば「返答あり」
MOOD_AXES = {
    "curiosity": "好奇心", "vigilance": "警戒", "weariness": "疲労", "attachment": "愛着",
    "hope": "希望", "frustration": "不満", "accomplishment": "達成感", "loneliness": "孤独",
}
DIALOGUE_MOOD_CAP = 15  # lease_intelligence_mind.DIALOGUE_MOOD_CAP
RELATIONSHIP_MAX = 10.0  # api/shion_relationship._SCORE_MAX
GROUNDING_FLAGGED = ("contradicted", "unsupported_fact")


# ---------- 共通 ----------

def to_jst(value: Any) -> datetime | None:
    """ISO 文字列を JST に。タイムゾーン無しは JST（紫苑のログは Mac のローカル時刻）とみなす。"""
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=JST) if parsed.tzinfo is None else parsed.astimezone(JST)


def jst_day(value: Any) -> date | None:
    parsed = to_jst(value)
    return parsed.date() if parsed else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in (path.with_suffix(path.suffix + ".1"), path):  # 回転済みの古い分も読む
        if not candidate.exists():
            continue
        with candidate.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def rate(hit: int, total: int) -> float | None:
    return round(hit / total * 100, 1) if total else None


def normalize(text: str) -> str:
    """api/answer_repeat_guard._normalize と同じ（api パッケージを読み込まずに使う）。"""
    return re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", str(text or ""))).lower()


def similar(a: str, b: str, limit: int = 2000) -> float:
    return difflib.SequenceMatcher(None, a[:limit], b[:limit]).ratio()


def days_between(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def in_period(day: date | None, start: date, end: date) -> bool:
    return day is not None and start <= day <= end


# ---------- 指標 ----------

def prediction_metrics(rows: list[dict[str, Any]], start: date, end: date) -> dict[str, Any]:
    """#1291 予想と答え合わせ: 気持ち・反応・話題の当たり率。"""
    picked = [r for r in rows if in_period(jst_day(r.get("at")), start, end) and not is_verification_row(r)]
    daily: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for r in picked:
        key = jst_day(r.get("at")).isoformat()
        daily[key]["n"] += 1
        for name in ("affect", "reaction", "topic"):
            daily[key][name] += bool(r.get(f"{name}_hit"))
    n = len(picked)
    surprise = [float(r.get("surprise") or 0) for r in picked]
    return {
        "n": n,
        "affect_hit": rate(sum(bool(r.get("affect_hit")) for r in picked), n),
        "reaction_hit": rate(sum(bool(r.get("reaction_hit")) for r in picked), n),
        "topic_hit": rate(sum(bool(r.get("topic_hit")) for r in picked), n),
        "surprise_mean": round(sum(surprise) / n, 2) if n else None,
        "affect_labels": dict(Counter(str(r.get("actual_affect")) for r in picked).most_common()),
        "daily": {k: {"n": v["n"], "affect": rate(v["affect"], v["n"]), "topic": rate(v["topic"], v["n"])}
                  for k, v in sorted(daily.items())},
    }


def grounding_metrics(rows: list[dict[str, Any]], budget_rows: list[dict[str, Any]], start: date, end: date) -> dict[str, Any]:
    """#1297 自己報告の接地: 照合ログの結果と、記録（気分の変化記録）をプロンプトへ渡した回数。"""
    picked = [r for r in rows if in_period(jst_day(r.get("ts")), start, end)]
    applied = [r for r in picked if r.get("status") == "applied" and r.get("kind") != "screening"]
    counts: Counter[str] = Counter()
    for r in applied:
        counts.update({k: int(v or 0) for k, v in dict(r.get("counts") or {}).items()})
    claims = counts["verified"] + sum(counts[k] for k in GROUNDING_FLAGGED)
    injected = sum(
        1 for r in budget_rows
        if in_period(jst_day(r.get("ts")), start, end)
        and int(dict(dict(r.get("blocks") or {}).get("emotion_grounding_context") or {}).get("kept") or 0) > 0
    )
    return {
        "checks": len(applied),
        "skipped_or_error": len(picked) - len(applied),
        "grounded_rate": rate(counts["verified"], claims),
        "verified": counts["verified"],
        "unsupported_fact": counts["unsupported_fact"],
        "contradicted": counts["contradicted"],
        "marked_interpretation": counts["marked_interpretation"],
        "evidence_injected": injected,
    }


def _chat_turns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns = []
    for r in rows:
        at = to_jst(r.get("ts"))
        if at is None:
            continue
        turns.append({
            "at": at,
            "who": str(r.get("user_id") or r.get("surface") or ""),
            "user": str(r.get("user_message") or ""),
            "reply": str(r.get("assistant_reply") or ""),
        })
    return sorted(turns, key=lambda t: t["at"])


def repeat_metrics(chat_rows: list[dict[str, Any]], start: date, end: date) -> dict[str, Any]:
    """#1341 答えの丸写し: 同じ日に同じ人がほぼ同じ質問をした時の、前回の答えとの類似度。"""
    by_day_user: dict[tuple[date, str], list[dict[str, Any]]] = defaultdict(list)
    for t in _chat_turns(chat_rows):
        by_day_user[(t["at"].date(), t["who"])].append(t)
    pairs = []
    for (day, _), turns in by_day_user.items():
        if not start <= day <= end:
            continue
        for i, turn in enumerate(turns):
            current = normalize(turn["user"])
            if len(current) < 4:
                continue
            for prev in reversed(turns[:i]):
                previous = normalize(prev["user"])
                if previous and (previous == current or similar(previous, current) >= REPEAT_RATIO):
                    pairs.append({
                        "day": day.isoformat(),
                        "after_guard": turn["at"] >= REPEAT_GUARD_SINCE,
                        "similarity": round(similar(normalize(prev["reply"]), normalize(turn["reply"])), 3),
                    })
                    break
    daily: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        daily[p["day"]].append(p["similarity"])
    sims = [p["similarity"] for p in pairs]
    after = [p["similarity"] for p in pairs if p["after_guard"]]
    return {
        "repeats": len(pairs),
        "mean_similarity": round(sum(sims) / len(sims), 3) if sims else None,
        "near_copies": sum(s >= COPY_RATIO for s in sims),
        "guard_instructions_est": len(after),
        "mean_similarity_after_guard": round(sum(after) / len(after), 3) if after else None,
        "daily": {k: round(sum(v) / len(v), 3) for k, v in sorted(daily.items())},
    }


_LABEL_RE = re.compile(r"^[^:：]{1,15}[:：]\s*")


def reflection_lines(text: str) -> list[str]:
    body = text.split("---", 2)[-1] if text.startswith("---") else text
    lines = []
    for raw in body.splitlines():
        line = raw.strip().lstrip("-*>・ ").strip()
        if not line or line.startswith("#"):
            continue
        line = _LABEL_RE.sub("", line)  # 「今日の観察:」など毎日同じ見出し語は比べない
        if len(normalize(line)) >= 8:
            lines.append(normalize(line))
    return lines


def reflection_metrics(reflection_dir: Path, start: date, end: date) -> dict[str, Any]:
    """#1288 Private Reflection: 前日の内省にない行（類似度0.8未満）の割合。"""
    daily: dict[str, float] = {}
    for day in days_between(start, end):
        today = reflection_dir / f"{day.isoformat()}.md"
        prev = reflection_dir / f"{(day - timedelta(days=1)).isoformat()}.md"
        if not (today.exists() and prev.exists()):
            continue
        now_lines = reflection_lines(today.read_text(encoding="utf-8", errors="ignore"))
        old_lines = reflection_lines(prev.read_text(encoding="utf-8", errors="ignore"))
        if not now_lines:
            continue
        new = sum(1 for line in now_lines if max((similar(line, o, 400) for o in old_lines), default=0.0) < 0.8)
        daily[day.isoformat()] = rate(new, len(now_lines))
    values = list(daily.values())
    return {
        "days": len(values),
        "novel_rate": round(sum(values) / len(values), 1) if values else None,
        "min_novel_rate": min(values) if values else None,
        "daily": daily,
    }


_QUESTION_RE = re.compile(r"[？?]\s*[」』）)]?\s*$")


def asks_question(reply: str) -> bool:
    """返答の最後の3文のどれかが問いかけ（？で終わる）なら、紫苑からの質問とみなす。"""
    sentences = [s for s in re.split(r"(?<=[。！!？?])\s*|\n+", reply.strip()) if s.strip()]
    return any(_QUESTION_RE.search(s) for s in sentences[-3:])


def curiosity_metrics(chat_rows: list[dict[str, Any]], start: date, end: date) -> dict[str, Any]:
    """好奇心: 紫苑からの質問の回数と、それに相手が返した割合。"""
    turns = _chat_turns(chat_rows)
    asked = answered = replies = 0
    daily: dict[str, Counter[str]] = defaultdict(Counter)
    for i, turn in enumerate(turns):
        if not in_period(turn["at"].date(), start, end):
            continue
        replies += 1
        if not asks_question(turn["reply"]):
            continue
        asked += 1
        key = turn["at"].date().isoformat()
        daily[key]["asked"] += 1
        nxt = next((t for t in turns[i + 1:] if t["who"] == turn["who"]), None)
        if nxt and nxt["at"] - turn["at"] <= REPLY_WINDOW:
            answered += 1
            daily[key]["answered"] += 1
    return {
        "replies": replies,
        "questions": asked,
        "question_share": rate(asked, replies),
        "answered_rate": rate(answered, asked),
        "daily": {k: {"asked": v["asked"], "answered": rate(v["answered"], v["asked"])} for k, v in sorted(daily.items())},
    }


_SNAPSHOT_RE = re.compile(r"(好奇心|警戒|疲労|愛着|希望|不満|達成感|孤独)=(\d+)")


def mood_metrics(mind: dict[str, Any], relationship: dict[str, Any], affect: dict[str, Any], start: date, end: date) -> dict[str, Any]:
    """関係性スコア（#1285）・気分の値: 現在値・期間内の変化・上限/下限への張り付き。"""
    mood = {k: v for k, v in dict(mind.get("mood") or {}).items() if k in MOOD_AXES}
    dialogue = dict(mind.get("dialogue_mood") or {})
    changes = [
        c for c in list(mind.get("mood_change_log") or [])
        if in_period(jst_day(c.get("ts")), start, end) and not is_verification_row(c)
    ]
    moved: Counter[str] = Counter()
    net: Counter[str] = Counter()
    for entry in changes:
        for ch in entry.get("changes") or []:
            axis = str(ch.get("axis") or "")
            delta = int(ch.get("after") or 0) - int(ch.get("before") or 0)
            if delta:
                moved[axis] += 1
                net[axis] += delta
    pinned = [MOOD_AXES[a] for a, v in mood.items() if v in (0, 100)]
    pinned += [f"{MOOD_AXES.get(a, a)}（対話の揺れ）" for a, v in dialogue.items() if abs(int(v or 0)) >= DIALOGUE_MOOD_CAP]
    snapshots: dict[str, dict[str, int]] = {}
    for item in mind.get("long_term_memories") or []:
        day = jst_day(item.get("date"))
        if item.get("type") == "emotion_snapshot" and in_period(day, start, end):
            snapshots[day.isoformat()] = {k: int(v) for k, v in _SNAPSHOT_RE.findall(str(item.get("content") or ""))}
    observations = [
        o for uid, user in dict(affect.get("users") or {}).items() for o in list(dict(user).get("observations") or [])
        if in_period(jst_day(o.get("at")), start, end) and not is_verification_row({**o, "user_id": uid})
    ]
    score = relationship.get("score")
    return {
        "mood": mood,
        "pad": dict(mind.get("pad") or {}),
        "mood_changes": len(changes),
        "moved_axes": {MOOD_AXES.get(a, a): moved[a] for a in MOOD_AXES if moved[a]},
        "net_delta": {MOOD_AXES.get(a, a): net[a] for a in MOOD_AXES if net[a]},
        "still_axes": [MOOD_AXES[a] for a in MOOD_AXES if not moved[a]] if changes else [],
        "pinned": pinned,
        "snapshots": snapshots,
        "relationship_score": score,
        "relationship_pinned": score is not None and float(score) >= RELATIONSHIP_MAX,
        "understanding": relationship.get("understanding"),
        "prediction_hits": relationship.get("prediction_hits"),
        "prediction_misses": relationship.get("prediction_misses"),
        "relationship_trend": relationship.get("trend"),
        "user_affect_labels": dict(Counter(str(o.get("label")) for o in observations).most_common()),
    }


def judgment_asset_metrics(rules_doc: Any, queue_doc: Any, growth_rows: list[dict[str, Any]], start: date, end: date) -> dict[str, Any]:
    """判断資産: 件数・出所（user/shion）・方針/目安/知見の内訳・期間内の新規。"""
    rules = list(dict(rules_doc or {}).get("rules") or [])
    active = [r for r in rules if r.get("status") == "active"]
    tiers = {str(item.get("judgment_id") or ""): str(item.get("tier") or "")
             for item in dict(dict(queue_doc or {}).get("items") or {}).values()}
    tier_names = {"policy": "方針", "guideline": "目安", "insight": "知見"}
    kinds: Counter[str] = Counter()
    for r in active:
        tier = tiers.get(str(r.get("id"))) or str(r.get("knowledge_kind") or "")
        kinds[tier_names.get(tier, "未判定")] += 1
    sources = Counter(str(r.get("content_source") or "user") for r in active)
    daily, as_of = {}, None
    for row in sorted(growth_rows, key=lambda r: str(r.get("date"))):
        day = jst_day(row.get("date"))
        if day is None or day > end:
            continue
        as_of = int(dict(row.get("counts") or {}).get("active_rules") or 0)  # 期間の終わり時点の件数
        if day >= start:
            daily[day.isoformat()] = as_of
    return {
        "active": len(active),
        "active_as_of_end": as_of,
        "new_in_period": sum(1 for r in active if in_period(jst_day(r.get("created_at")), start, end)),
        "sources": dict(sources),
        "excluded_shion": sum(1 for r in rules if r.get("status") == "excluded_shion_reply"),
        "kinds": dict(kinds),
        "daily_active": dict(sorted(daily.items())),
    }


def news_zettel_metrics(state: Any, budget_rows: list[dict[str, Any]], start: date, end: date) -> dict[str, Any]:
    """ニュース永続メモ: 件数・ハブ接続率・参考メモがプロンプトに添えられた回数。"""
    entries = [v for v in dict(state or {}).values() if isinstance(v, dict)]
    written = [e for e in entries if e.get("status") == "written"
               and (jst_day(e.get("processed_at")) or end) <= end]  # 期間の終わり時点の累計
    new = [e for e in written if in_period(jst_day(e.get("processed_at")), start, end)]
    attached = Counter()
    for r in budget_rows:
        if not in_period(jst_day(r.get("ts")), start, end):
            continue
        if int(dict(dict(r.get("blocks") or {}).get("news_zettel_context") or {}).get("kept") or 0) > 0:
            attached[str(r.get("surface") or "")] += 1
    return {
        "total_written": len(written),
        "new_in_period": len(new),
        "hub_rate_total": rate(sum(bool(e.get("hubs")) for e in written), len(written)),
        "hub_rate_new": rate(sum(bool(e.get("hubs")) for e in new), len(new)),
        "status": dict(Counter(str(e.get("status")) for e in entries)),
        "attached": sum(attached.values()),
        "attached_by_surface": dict(attached),
    }


def cost_metrics(rows: list[dict[str, Any]], start: date, end: date) -> dict[str, Any]:
    """費用: 週の Gemini 概算（ai_budget と同じ換算）と機能別内訳。"""
    from ai_budget import entry_cost_usd, to_yen

    by_feature: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    daily: Counter[str] = Counter()
    first = None
    for r in rows:
        day = jst_day(r.get("timestamp"))
        if day and (first is None or day < first):
            first = day
        if not in_period(day, start, end):
            continue
        yen = to_yen(entry_cost_usd(r))
        by_feature[str(r.get("feature") or "unknown")] += yen
        by_class[str(r.get("call_class") or "未記録")] += yen
        daily[day.isoformat()] += yen
    return {
        "yen": round(sum(daily.values()), 1),
        "by_feature": {k: round(v, 1) for k, v in by_feature.most_common(8)},
        "by_class": {k: round(v, 1) for k, v in by_class.most_common()},
        "daily": {k: round(v, 1) for k, v in sorted(daily.items())},
        "log_starts": first.isoformat() if first else None,
    }


def collect(source: Path, vault: Path, start: date, end: date, *, today: date | None = None) -> dict[str, Any]:
    """today を渡すと、終わりが2日以上前の期間では現在値しか無い指標（関係性・気分）を空にする。"""
    budget = read_jsonl(source / "chat_prompt_budget_log.jsonl")
    # REV-591: 検証の会話（origin=verification・検証用ユーザーID）は成長記録に数えない
    chat = [row for row in read_jsonl(source / "cloudrun_chat_log.jsonl") if not is_verification_row(row)]
    mind = read_json(vault / MIND_REL) or {}
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "prediction": prediction_metrics(read_jsonl(source / "shion_prediction_log.jsonl"), start, end),
        "grounding": grounding_metrics(read_jsonl(source / "shion_emotion_grounding_log.jsonl"), budget, start, end),
        "repeat": repeat_metrics(chat, start, end),
        "reflection": reflection_metrics(vault / REFLECTION_REL, start, end),
        "curiosity": curiosity_metrics(chat, start, end),
        "mood": mood_metrics(mind, read_json(source / "shion_relationship_state.json") or {},
                             read_json(source / "user_affect_state.json") or {}, start, end),
        "judgment": judgment_asset_metrics(read_json(source / "canonical_judgment_rules.json"),
                                           read_json(source / "policy_likeness_queue.json"),
                                           read_jsonl(source / "judgment_asset_growth_history.jsonl"), start, end),
        "news": news_zettel_metrics(read_json(source / "news_zettel_state.json"), budget, start, end),
        "cost": cost_metrics(read_jsonl(source / "ai_usage.jsonl"), start, end),
        "current_state": today is None or end >= today - timedelta(days=1),
    }


# ---------- 週ごとの履歴 ----------

def week_buckets(start: date, end: date) -> list[tuple[date, date]]:
    """日曜始まりの週で区切る（日曜朝の定期実行が 日〜土 を集計するのに合わせる）。"""
    buckets = []
    cursor = start
    while cursor <= end:
        week_end = min(end, cursor + timedelta(days=(5 - cursor.weekday()) % 7))
        buckets.append((cursor, week_end))
        cursor = week_end + timedelta(days=1)
    return buckets


def history_row(m: dict[str, Any], *, partial: bool) -> dict[str, Any]:
    now = m["current_state"]  # 現在値しか無い指標は、最近の期間の行にだけ入れる
    return {
        "start": m["start"], "end": m["end"], "partial": partial,
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "prediction_n": m["prediction"]["n"], "affect_hit": m["prediction"]["affect_hit"],
        "topic_hit": m["prediction"]["topic_hit"],
        "grounding_checks": m["grounding"]["checks"], "grounded_rate": m["grounding"]["grounded_rate"],
        "unsupported_fact": m["grounding"]["unsupported_fact"],
        "repeats": m["repeat"]["repeats"], "repeat_similarity": m["repeat"]["mean_similarity"],
        "reflection_novel_rate": m["reflection"]["novel_rate"],
        "shion_questions": m["curiosity"]["questions"], "question_answered_rate": m["curiosity"]["answered_rate"],
        "relationship_score": m["mood"]["relationship_score"] if now else None,
        "understanding": m["mood"]["understanding"] if now else None,
        "mood": m["mood"]["mood"] if now else None,
        "judgment_active": m["judgment"]["active_as_of_end"] if m["judgment"]["active_as_of_end"] is not None
        else (m["judgment"]["active"] if now else None), "judgment_new": m["judgment"]["new_in_period"],
        "zettel_total": m["news"]["total_written"], "zettel_hub_rate": m["news"]["hub_rate_total"],
        "zettel_attached": m["news"]["attached"],
        "cost_yen": m["cost"]["yen"],
    }


def update_history(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同じ期間の行は置き換える（遡り集計と定期実行が重なっても1期間1行）。"""
    existing = {(r.get("start"), r.get("end")): r for r in read_jsonl(path)}
    for row in rows:
        # 定期実行の週と、それに含まれる途中の行（遡り集計の当日分など）は重ねない
        existing = {k: v for k, v in existing.items()
                    if not (v.get("partial") and k[0] == row["start"])}
        existing[(row["start"], row["end"])] = row
    merged = sorted(existing.values(), key=lambda r: (str(r.get("start")), str(r.get("end"))))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), encoding="utf-8")
    tmp.replace(path)
    return merged


# ---------- Markdown ----------

def fmt(value: Any, unit: str = "") -> str:
    if value is None:
        return "—"
    return f"{value}{unit}"


def mermaid_line(title: str, series: dict[str, Any], *, y_label: str = "%", y_max: float | None = 100) -> str:
    points = [(k, v) for k, v in series.items() if v is not None]
    if len(points) < 2:
        return f"（{title}: 点が2つ未満のためグラフなし）"
    labels = ", ".join(f'"{k[5:].replace("-", "/")}"' for k, _ in points)
    values = ", ".join(str(v) for _, v in points)
    top = y_max if y_max is not None else max(float(v) for _, v in points) * 1.2 or 1
    return "\n".join([
        "```mermaid", "xychart-beta", f'    title "{title}"', f"    x-axis [{labels}]",
        f'    y-axis "{y_label}" 0 --> {round(top, 1)}', f"    line [{values}]", "```",
    ])


def kv(data: dict[str, Any], unit: str = "") -> str:
    return "、".join(f"{k} {v}{unit}" for k, v in data.items()) or "—"


UNAVAILABLE = """\
| 取れない・弱い指標 | 理由 | 記録の追加案 |
|---|---|---|
| 聞き直しの指示が実際に入った回数（#1341） | 指示を足した時にログを残していない。ここでは REV-544 マージ後の聞き直しを「入った」と推定 | `apply_repeat_guard` で指示を足した時に `chat_prompt_budget_log` のブロック名（例 `repeat_question_block`）として残す |
| 話題別の予想の当たり率（#1291） | 答え合わせログに予想した話題・実際の話題が無く、当たり外れだけ | `shion_prediction_log.jsonl` に `expected_topic` / `actual_topic` を足す |
| 関係性スコア・気分の日ごとの推移 | 状態ファイルは現在値だけ。気分の変化記録は直近40件で古い分は消える | このスクリプトが週1で値を `shion_growth_weekly_history.jsonl` に貯める（実装済み）。日ごとが要るなら朝の内省時に1行残す |
| 参考メモが「回答に」添えられた回数 | 分かるのはプロンプトに入った回数まで。回答が実際に触れたかは未記録 | 返答後に参考メモの題名・語が返答に含まれたかを決定的に照合して1行残す（AI 不要） |
| 紫苑からの質問の意図 | 「？」で終わる文での判定。相手を知るための質問か、確認・社交辞令かは分けていない | 必要なら Jev classify で週1回まとめて分類（1回 数十件・費用はごく小さい見込み。予算ガード内の検証クラスで実行） |
"""


def render_note(m: dict[str, Any], weeks: list[dict[str, Any]], *, title: str, partial_note: str = "",
                prev: dict[str, Any] | None = None) -> str:
    p, g, r, rf, c, mo, j, n, co = (m[k] for k in ("prediction", "grounding", "repeat", "reflection", "curiosity",
                                                   "mood", "judgment", "news", "cost"))
    prev = prev or {}
    lines = [
        "---",
        "tags: [tune_lease_55, 紫苑, 成長記録, mermaid]",
        f"period: {m['start']}〜{m['end']}",
        f"generated: {datetime.now(JST).isoformat(timespec='minutes')}",
        "source: scripts/shion_growth_weekly.py（REV-586・AI 呼び出しなし）",
        "rag_exclude: true",
        "---",
        "",
        f"# {title}",
        "",
        f"紫苑の稼働開始 2026-06-12 から {(date.fromisoformat(m['end']) - date(2026, 6, 12)).days} 日目。"
        "既存のログ・状態ファイルだけから集計（AI 呼び出しなし・記憶は読むだけ）。" + partial_note,
        "",
        f"← [[{INDEX_NAME}]]",
        "",
        "## 要約",
        "",
        "| 指標 | この期間 | 前週 | 元データ |",
        "|---|---|---|---|",
        f"| 予想の当たり率（気持ち） | {fmt(p['affect_hit'], '%')}（{p['n']}回） | {fmt(prev.get('affect_hit'), '%')} | shion_prediction_log.jsonl |",
        f"| 予想の当たり率（話題） | {fmt(p['topic_hit'], '%')} | {fmt(prev.get('topic_hit'), '%')} | 同上 |",
        f"| 自己報告の接地率 | {fmt(g['grounded_rate'], '%')}（照合 {g['checks']}回） | {fmt(prev.get('grounded_rate'), '%')} | shion_emotion_grounding_log.jsonl |",
        f"| 記録にないことを事実として語った | {g['unsupported_fact']}件 | {fmt(prev.get('unsupported_fact'), '件')} | 同上 |",
        f"| 同じ質問の聞き直し / 前回の答えとの類似度 | {r['repeats']}回 / {fmt(r['mean_similarity'])} | {fmt(prev.get('repeats'), '回')} / {fmt(prev.get('repeat_similarity'))} | cloudrun_chat_log.jsonl |",
        f"| Private Reflection の前日と違う行の割合 | {fmt(rf['novel_rate'], '%')}（{rf['days']}日） | {fmt(prev.get('reflection_novel_rate'), '%')} | Private Reflection/*.md |",
        f"| 紫苑からの質問 / 返答率 | {c['questions']}回 / {fmt(c['answered_rate'], '%')} | {fmt(prev.get('shion_questions'), '回')} / {fmt(prev.get('question_answered_rate'), '%')} | cloudrun_chat_log.jsonl |",
        f"| 関係性スコア（現在値） | {fmt(mo['relationship_score'])}{'（上限に張り付き）' if mo['relationship_pinned'] else ''} | {fmt(prev.get('relationship_score'))} | shion_relationship_state.json |",
        f"| 判断資産（active / 期間内の新規） | {j['active']}件 / {j['new_in_period']}件 | {fmt(prev.get('judgment_active'), '件')} | canonical_judgment_rules.json |",
        f"| ニュース永続メモ（累計 / ハブ接続率） | {n['total_written']}件 / {fmt(n['hub_rate_total'], '%')} | {fmt(prev.get('zettel_total'), '件')} / {fmt(prev.get('zettel_hub_rate'), '%')} | news_zettel_state.json |",
        f"| 参考メモがプロンプトに添えられた回数 | {n['attached']}回 | {fmt(prev.get('zettel_attached'), '回')} | chat_prompt_budget_log.jsonl |",
        f"| Gemini 概算費用 | {co['yen']}円 | {fmt(prev.get('cost_yen'), '円')} | ai_usage.jsonl |",
        "",
        "## 1. 予想と答え合わせ（#1291）",
        "",
        f"- 答え合わせ {p['n']}回。当たり率は 気持ち {fmt(p['affect_hit'], '%')}・反応 {fmt(p['reaction_hit'], '%')}・"
        f"話題 {fmt(p['topic_hit'], '%')}。驚きの平均 {fmt(p['surprise_mean'])}",
        f"- 実際の相手の気持ち: {kv(p['affect_labels'], '回')}",
        "",
        mermaid_line("予想の当たり率（気持ち）", {k: v["affect"] for k, v in p["daily"].items()}),
        "",
        mermaid_line("予想の当たり率（話題）", {k: v["topic"] for k, v in p["daily"].items()}),
        "",
        "## 2. 自己報告の接地（#1297）",
        "",
        f"- 照合 {g['checks']}回（スキップ・失敗 {g['skipped_or_error']}回）。記録に裏づけ {g['verified']}文、"
        f"記録にない事実 {g['unsupported_fact']}文、記録と矛盾 {g['contradicted']}文、解釈と明示 {g['marked_interpretation']}文",
        f"- 気分の変化記録を対話プロンプトへ渡した回数: {g['evidence_injected']}回",
    ]
    if not g["checks"]:
        lines.append("- 照合ログがまだ無い（照合は「真面目な感情の質問」の時だけ走る）。渡した回数はあるので、根拠の材料は届いている")
    lines += [
        "",
        "## 3. 答えの丸写し（#1341）",
        "",
        f"- 同じ日に同じ人がほぼ同じ質問をした回数 {r['repeats']}回。前回の答えとの類似度の平均 {fmt(r['mean_similarity'])}、"
        f"ほぼ丸写し（{COPY_RATIO}以上）{r['near_copies']}回",
        f"- 聞き直しの指示が入った推定回数（REV-544 マージ後の聞き直し）{r['guard_instructions_est']}回、"
        f"その時の類似度の平均 {fmt(r['mean_similarity_after_guard'])}",
        "",
        mermaid_line("聞き直し時の前回の答えとの類似度", r["daily"], y_label="類似度", y_max=1),
        "",
        "## 4. Private Reflection（#1288）",
        "",
        f"- 前日の内省にない行の割合 平均 {fmt(rf['novel_rate'], '%')}・最低 {fmt(rf['min_novel_rate'], '%')}（{rf['days']}日分）",
        "",
        mermaid_line("内省の前日と違う行の割合", rf["daily"]),
        "",
        "## 5. 好奇心",
        "",
        f"- 紫苑の返答 {c['replies']}回のうち、相手への問いかけで終わるもの {c['questions']}回（{fmt(c['question_share'], '%')}）。"
        f"{int(REPLY_WINDOW.total_seconds() // 60)}分以内に相手が返した割合 {fmt(c['answered_rate'], '%')}",
        "",
        mermaid_line("紫苑からの質問の回数", {k: v["asked"] for k, v in c["daily"].items()}, y_label="回", y_max=None),
        "",
        "## 6. 関係性スコア（#1285）と気分",
        "",
        f"- 関係性スコア {fmt(mo['relationship_score'])} / {RELATIONSHIP_MAX}（傾向 {fmt(mo['relationship_trend'])}）、"
        f"相手の理解度 {fmt(mo['understanding'])}、予想の当たり {fmt(mo['prediction_hits'])}・外れ {fmt(mo['prediction_misses'])}",
        f"- 気分の現在値: {kv({MOOD_AXES[k]: v for k, v in mo['mood'].items()})}（PAD {kv(mo['pad'])}）",
        f"- 期間内の気分の変化記録 {mo['mood_changes']}件。動いた回数: {kv(mo['moved_axes'], '回')}。差し引き: {kv(mo['net_delta'])}",
        f"- 期間内に一度も動かなかった軸: {'、'.join(mo['still_axes']) or '—'}",
        f"- 上限・下限に張り付いている値: {'、'.join(mo['pinned'] + (['関係性スコア'] if mo['relationship_pinned'] else [])) or 'なし'}",
        f"- 相手の様子（推定）: {kv(mo['user_affect_labels'], '回')}",
    ]
    if mo["snapshots"]:
        lines += ["", "| 日付 | " + " | ".join(MOOD_AXES.values()) + " |", "|---|" + "---|" * len(MOOD_AXES)]
        for day, values in sorted(mo["snapshots"].items()):
            lines.append(f"| {day} | " + " | ".join(str(values.get(name, "—")) for name in MOOD_AXES.values()) + " |")
    lines += [
        "",
        "## 7. 判断資産",
        "",
        f"- active {j['active']}件（期間内の新規 {j['new_in_period']}件）。出所: {kv(j['sources'], '件')}"
        f"（出所未記入は user 扱い）。紫苑自身の返答として除外済み {j['excluded_shion']}件",
        f"- 方針/目安/知見: {kv(j['kinds'], '件')}",
        "",
        mermaid_line("判断資産（active）", j["daily_active"], y_label="件", y_max=None),
        "",
        "## 8. ニュース永続メモ",
        "",
        f"- 書いた永続メモ 累計 {n['total_written']}件（期間内 {n['new_in_period']}件）。ハブ接続率 累計 {fmt(n['hub_rate_total'], '%')}・"
        f"期間内 {fmt(n['hub_rate_new'], '%')}",
        f"- 参考メモがプロンプトに添えられた回数 {n['attached']}回（{kv(n['attached_by_surface'], '回')}）",
        "",
        "## 9. 費用",
        "",
        f"- Gemini 概算 {co['yen']}円（ai_budget と同じ換算・補正込み）。使用量ログは {fmt(co['log_starts'])} から",
        f"- 機能別（上位）: {kv(co['by_feature'], '円')}",
        f"- 呼び出しクラス別: {kv(co['by_class'], '円')}",
        "",
        mermaid_line("Gemini 概算費用（日別）", co["daily"], y_label="円", y_max=None),
        "",
    ]
    if len(weeks) >= 2:
        lines += ["## 週ごとの推移", "", *weekly_table(weeks), "", *weekly_charts(weeks), ""]
    lines += ["## 取れない指標・記録の追加案", "", UNAVAILABLE]
    return "\n".join(lines)


def weekly_charts(weeks: list[dict[str, Any]]) -> list[str]:
    def series(key: str) -> dict[str, Any]:
        return {w["start"]: w.get(key) for w in weeks}
    return [
        mermaid_line("予想の当たり率（気持ち）・週", series("affect_hit")), "",
        mermaid_line("内省の前日と違う行の割合・週", series("reflection_novel_rate")), "",
        mermaid_line("紫苑からの質問の回数・週", series("shion_questions"), y_label="回", y_max=None), "",
        mermaid_line("関係性スコア・週", series("relationship_score"), y_label="点", y_max=RELATIONSHIP_MAX), "",
        mermaid_line("判断資産（active）・週", series("judgment_active"), y_label="件", y_max=None), "",
        mermaid_line("Gemini 概算費用・週", series("cost_yen"), y_label="円", y_max=None),
    ]


def weekly_table(weeks: list[dict[str, Any]]) -> list[str]:
    head = ["| 期間 | 予想の当たり（気持ち） | 内省の新しさ | 紫苑の質問 | 聞き直し時の類似度 | 関係性 | 判断資産 | 永続メモ | 費用 |",
            "|---|---|---|---|---|---|---|---|---|"]
    return head + [
        f"| {w['start']}〜{w['end']}{'（途中）' if w.get('partial') else ''} | {fmt(w.get('affect_hit'), '%')} | "
        f"{fmt(w.get('reflection_novel_rate'), '%')} | {fmt(w.get('shion_questions'))} | {fmt(w.get('repeat_similarity'))} | "
        f"{fmt(w.get('relationship_score'))} | {fmt(w.get('judgment_active'))} | {fmt(w.get('zettel_total'))} | {fmt(w.get('cost_yen'), '円')} |"
        for w in weeks
    ]


def render_index(note_names: list[str], weeks: list[dict[str, Any]]) -> str:
    rows = "\n".join(f"| [[{NOTE_DIR}/{name}\\|{name}]] |" for name in sorted(note_names, reverse=True))
    return "\n".join([
        "---",
        "tags: [tune_lease_55, 紫苑, 成長記録, mermaid]",
        "source: scripts/shion_growth_weekly.py（REV-586）",
        "rag_exclude: true",
        "---",
        "",
        "# 紫苑の成長記録（週次）",
        "",
        "ここ数日入れた機能が紫苑を実際に育てているかを、毎週日曜 08:30（launchd `com.tunelease.shion-growth-weekly`）に"
        "前の日曜〜土曜の7日分で数字にしたノート群。AI 呼び出しなし、記憶系は読むだけ。"
        "学会発表（テーマ「関係性が意識を生む」・稼働開始 2026-06-12）の材料用。",
        "",
        f"← [[{TOC_NAME[:-3]}]]",
        "",
        "## 週ごとの数字",
        "",
        "関係性・気分は現在値しか残らないため、定期実行を始めた週から値が入る（遡り集計の過去の週は「—」）。",
        "",
        *weekly_table(weeks),
        "",
        *weekly_charts(weeks),
        "",
        "## ノート",
        "",
        "| ノート |",
        "|---|",
        rows,
        "",
    ])


def link_from_toc(out_dir: Path) -> bool:
    """00_目次 に成長記録への1行が無ければ「運用の見直し」の表の末尾へ足す。"""
    toc = out_dir / TOC_NAME
    if not toc.exists():
        return False
    text = toc.read_text(encoding="utf-8")
    if f"[[{INDEX_NAME}]]" in text:
        return False
    row = f"| [[{INDEX_NAME}]] | 紫苑の成長を週1回数字にした記録（予想の当たり率・内省の新しさ・質問・関係性・判断資産・費用など、REV-586） |"
    lines = text.splitlines()
    anchor = max((i for i, line in enumerate(lines) if line.startswith("| [[1")), default=None)
    if anchor is None:
        lines += ["", "## 成長の記録", "", "| ノート | 中身 |", "|---|---|", row]
    else:
        lines.insert(anchor + 1, row)
    toc.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return True


# ---------- 実行 ----------

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", help="開始日（遡り集計）。省略時は前の日曜")
    parser.add_argument("--to", dest="end", help="終了日。省略時は昨日")
    parser.add_argument("--source-dir", type=Path, default=None, help="読むログの場所（既定 DATA_DIR）")
    parser.add_argument("--vault", type=Path, default=None, help="読む Obsidian Vault（mind.json・内省）")
    parser.add_argument("--out-dir", type=Path, default=None, help="ノートの書き先（既定 Vault の紫苑の仕組み_2026-10）")
    parser.add_argument("--history", type=Path, default=None, help="週ごとの履歴（既定 DATA_DIR/" + HISTORY_NAME + "）")
    parser.add_argument("--dry-run", action="store_true", help="書き込まずにノートを表示する")
    args = parser.parse_args(list(argv) if argv is not None else None)

    today = datetime.now(JST).date()
    end = date.fromisoformat(args.end) if args.end else today - timedelta(days=1)
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=6)
    if start > end:
        parser.error("--from が --to より後です")
    source = args.source_dir or get_data_dir()
    vault = args.vault or resolve_obsidian_vault()
    out_dir = args.out_dir or vault / OUT_REL
    history_path = args.history or Path(get_data_path(HISTORY_NAME))

    backfill = bool(args.start)
    buckets = week_buckets(start, end) if backfill else [(start, end)]
    rows = []
    for b_start, b_end in buckets:
        partial = (b_end - b_start).days < 6 or b_end >= today
        rows.append(history_row(collect(source, vault, b_start, b_end, today=today), partial=partial))
    metrics = collect(source, vault, start, end, today=today)
    prefix = "遡り集計" if backfill else "週次"
    name = f"{prefix}_{start.isoformat()}〜{end.isoformat()}"
    title = f"紫苑の成長記録 {start.isoformat()}〜{end.isoformat()}" + ("（遡り集計）" if backfill else "")
    partial_note = "（遡り集計は使用量ログ・照合ログなどが始まった日以降だけ数字が出る。最終日は当日の途中まで）" if end >= today else ""

    if args.dry_run:
        print(render_note(metrics, rows, title=title, partial_note=partial_note))
        return 0
    before = [w for w in read_jsonl(history_path) if str(w.get("end")) < start.isoformat()]
    weeks = update_history(history_path, rows)
    note_dir = out_dir / NOTE_DIR
    note_dir.mkdir(parents=True, exist_ok=True)
    # 定期実行は「前の週」と比べる。遡り集計は期間の長さが揃わないので比べず、週ごとの表を載せる
    prev = None if backfill else (before[-1] if before else None)
    (note_dir / f"{name}.md").write_text(render_note(metrics, rows if backfill else weeks, title=title,
                                                     partial_note=partial_note, prev=prev), encoding="utf-8")
    notes = [p.stem for p in note_dir.glob("*.md")]
    (out_dir / f"{INDEX_NAME}.md").write_text(render_index(notes, weeks), encoding="utf-8")
    linked = link_from_toc(out_dir)
    print(json.dumps({"note": str(note_dir / f"{name}.md"), "weeks": len(weeks), "toc_linked": linked,
                      "history": str(history_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
