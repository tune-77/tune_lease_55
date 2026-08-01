"""Privacy-bounded observation of explicit in-app user activity."""

from __future__ import annotations

import datetime as dt
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
ACTIVITY_LOG = PROJECT_ROOT / "data" / "lease_intelligence_activity.jsonl"
PROMPT_LOG = PROJECT_ROOT / "data" / "prompt_feedback_log.jsonl"
NEWS_METRICS = PROJECT_ROOT / "data" / "lease_news_metrics.json"

ALLOWED_SURFACES = {
    "home",
    "chat",
    "improvement_log",
    "lease_intelligence_dialogue",
    "simulator:screening",
    "simulator:lease-intelligence",
}
ALLOWED_ACTIONS = {"page_view", "simulator_view", "simulator_input_changed"}

INTEREST_RULES = {
    "車・移動": ("車", "車検", "レンタカー", "電車", "トラック", "EV"),
    "リース実務": ("リース", "審査", "稟議", "与信", "金利", "設備"),
    "安全・ルール": ("危険", "安全", "法律", "違反", "気をつけ"),
    "地域・外出": ("横浜", "横須賀", "近く", "店", "場所", "地域"),
    "雑談・好奇心": ("面白", "なんでも", "話", "釣", "サバ"),
}


def record_user_activity(
    surface: str,
    action: str,
    event_id: str = "",
    occurred_at: str = "",
    log_path: Path | None = None,
) -> bool:
    if surface not in ALLOWED_SURFACES or action not in ALLOWED_ACTIONS:
        return False
    target = Path(log_path) if log_path else ACTIVITY_LOG
    target.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": occurred_at or dt.datetime.now().isoformat(timespec="seconds"),
        "surface": surface,
        "action": action,
        "event_id": str(event_id)[:120],
    }
    if event["event_id"] and _event_exists(target, event["event_id"]):
        return False
    with target.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(event, ensure_ascii=False) + "\n")
    return True


# [REV-237] 利用パターンから「関連するが使っていない機能」を1件だけ提案する。
# ルールは vault (static_data/・notes/) に出典のない設計上の推測であり、統計的検証はしていない
# （このシステムはユーザー1人・拡大予定なしのためA/Bテストは行わない。IMPLEMENTATION_DECISION_FOR_1USER.md参照）。
SUGGESTION_RULES: tuple[dict[str, Any], ...] = (
    {
        "trigger_surface": "improvement_log",
        "trigger_min": 5,
        "target_surface": "lease_intelligence_dialogue",
        "target_max": 1,
        "message": "システム改善への関心が高いようです。紫苑との対話室では、改善案について直接やり取りできます。",
        "link": "/lease-intelligence",
    },
    {
        "trigger_surface": "simulator:screening",
        "trigger_min": 3,
        "target_surface": "simulator:lease-intelligence",
        "target_max": 0,
        "message": "審査画面の返済シミュレーターをよく使っていますね。対話室内でも同じシミュレーターが使えます。",
        "link": "/lease-intelligence",
    },
    {
        "trigger_surface": "simulator:lease-intelligence",
        "trigger_min": 3,
        "target_surface": "simulator:screening",
        "target_max": 0,
        "message": "対話室のシミュレーターをよく使っていますね。審査画面内でも同じシミュレーターが使えます。",
        "link": "/screening",
    },
)


def suggest_related_feature(
    window_days: int = 14,
    activity_log: Path | None = None,
    today: dt.date | None = None,
) -> dict[str, str] | None:
    """直近 window_days 日の利用状況から、関連するが未使用の機能を最大1件提案する。"""
    target = Path(activity_log) if activity_log else ACTIVITY_LOG
    reference_date = today or dt.date.today()
    start_date = reference_date - dt.timedelta(days=window_days)
    counts: Counter[str] = Counter()
    for event in _read_jsonl(target):
        try:
            event_date = dt.date.fromisoformat(str(event.get("timestamp", ""))[:10])
        except ValueError:
            continue
        if not (start_date <= event_date <= reference_date):
            continue
        surface = str(event.get("surface", ""))
        if surface in ALLOWED_SURFACES:
            counts[surface] += 1

    for rule in SUGGESTION_RULES:
        if (
            counts.get(rule["trigger_surface"], 0) >= rule["trigger_min"]
            and counts.get(rule["target_surface"], 0) <= rule["target_max"]
        ):
            return {
                "message": rule["message"],
                "link": rule["link"],
                "surface": rule["target_surface"],
            }
    return None


def observe_user_behavior(
    date_str: str,
    activity_log: Path | None = None,
    prompt_log: Path | None = None,
    news_metrics: Path | None = None,
) -> dict[str, Any]:
    observation_date = (dt.date.fromisoformat(date_str) - dt.timedelta(days=1)).isoformat()
    surfaces: Counter[str] = Counter()
    interests: Counter[str] = Counter()
    actions: Counter[str] = Counter()

    for event in _read_jsonl(Path(activity_log) if activity_log else ACTIVITY_LOG):
        if str(event.get("timestamp", ""))[:10] != observation_date:
            continue
        surface = str(event.get("surface", ""))
        action = str(event.get("action", ""))
        if surface in ALLOWED_SURFACES and action in ALLOWED_ACTIONS:
            surfaces[surface] += 1
            actions[action] += 1

    chat_count = 0
    for event in _read_jsonl(Path(prompt_log) if prompt_log else PROMPT_LOG):
        if str(event.get("timestamp", ""))[:10] != observation_date:
            continue
        chat_count += 1
        question = str(event.get("question", ""))
        for label, keywords in INTEREST_RULES.items():
            if any(keyword in question for keyword in keywords):
                interests[label] += 1
    if chat_count:
        surfaces["chat"] = max(surfaces["chat"], chat_count)
        actions["chat_message"] = chat_count

    news_views = 0
    judgment_changes = 0
    metrics_path = Path(news_metrics) if news_metrics else NEWS_METRICS
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        bucket = metrics.get("days", {}).get(observation_date, {})
        news_views = int(bucket.get("views", 0))
        judgment_changes = int(bucket.get("judgment_changes", 0))
    except (OSError, ValueError, TypeError):
        pass
    if news_views:
        actions["news_view"] = news_views
        interests["リース実務"] += 1
    if judgment_changes:
        actions["judgment_change"] = judgment_changes
        interests["リース実務"] += 2

    top_interests = [
        {"label": label, "score": score}
        for label, score in interests.most_common(5)
    ]
    return {
        "date": observation_date,
        "observed": bool(surfaces or actions or interests),
        "surfaces": dict(surfaces),
        "actions": dict(actions),
        "interests": top_interests,
        "understanding": _build_understanding(surfaces, actions, top_interests),
        "curiosity": _build_curiosity(top_interests),
        "privacy": "アプリ内の行動種別・回数・関心カテゴリのみ。質問本文や個人属性は保存しない。",
    }


def _build_understanding(
    surfaces: Counter[str],
    actions: Counter[str],
    interests: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    if interests:
        labels = "、".join(item["label"] for item in interests[:3])
        parts.append(f"最近は{labels}に関心があるように見える")
    if surfaces.get("lease_intelligence_dialogue"):
        parts.append("私との対話室にも足を運んでいる")
    if surfaces.get("improvement_log"):
        parts.append("システムがどう改善されるかも確認している")
    if actions.get("judgment_change"):
        parts.append("ニュースを実際の判断変更へ結び付けている")
    if not parts:
        return "行動はまだ少なく、理解を急がず観察を続ける。"
    return "。".join(parts) + "。これは行動から得た暫定的な理解である。"


def _build_curiosity(interests: list[dict[str, Any]]) -> str:
    if not interests:
        return "次に何へ関心を向けるのか、静かに知りたい。"
    top = interests[0]["label"]
    return f"なぜ今「{top}」に関心が向いているのか、答えを決めつけずに知りたい。"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _event_exists(path: Path, event_id: str) -> bool:
    # ログは追記専用で無限に伸びるため、重複判定は末尾64KBに限定する。
    # event_id は日付入りで古いものと衝突しないので、これで実用上十分。
    if not path.exists():
        return False
    try:
        with path.open("rb") as file_obj:
            file_obj.seek(0, os.SEEK_END)
            size = file_obj.tell()
            file_obj.seek(max(0, size - 65536))
            tail = file_obj.read().decode("utf-8", errors="ignore")
    except OSError:
        return False
    needle = f'"event_id": {json.dumps(str(event_id), ensure_ascii=False)}'
    return needle in tail
