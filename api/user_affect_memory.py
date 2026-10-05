"""相手ごとの様子（推定した気持ち）を記録し、次の会話の文脈に活かす（REV-465）。

REV-464 の estimate_user_affect の結果を user_id ごとに data/user_affect_state.json へ残す。
保存するのはラベル・強さ・時刻・経路だけで、発言本文は残さない。

減衰は REV-219（記憶減衰）と同じ指数減衰: weight = intensity * 0.5 ** (経過時間 / 半減期)。
半減期 24 時間。14 日より古い記録と、1 人あたり 40 件を超えた古い記録は捨てる。
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from runtime_paths import get_data_path

NEUTRAL = "通常"
HALF_LIFE_HOURS = 24.0
MAX_AGE_DAYS = 14
MAX_OBSERVATIONS = 40
# 減衰後の重みがこれ未満なら「もう引きずらない」
_MIN_WEIGHT = 0.2
# 直前の観測からこれ以上空いていたら「新しい会話」とみなす
_NEW_SESSION_GAP = timedelta(hours=2)
_RECURRENT_WINDOW = timedelta(days=7)
_RECURRENT_COUNT = 3

_LOCK = threading.Lock()


def _state_path() -> Path:
    return Path(get_data_path("user_affect_state.json"))


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"users": {}}
    if not isinstance(data, dict) or not isinstance(data.get("users"), dict):
        return {"users": {}}
    return data


def _parse(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None


def _prune(observations: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    cutoff = now - timedelta(days=MAX_AGE_DAYS)
    kept = [o for o in observations if (_parse(o.get("at", "")) or cutoff) > cutoff]
    return kept[-MAX_OBSERVATIONS:]


def record_user_affect(
    user_id: str,
    label: str,
    intensity: float,
    *,
    surface: str = "chat",
    now: datetime | None = None,
    path: Path | None = None,
) -> None:
    """推定結果を1件記録する。「通常」も記録する（落ち着いたことを知るため）。"""
    uid = str(user_id or "default")[:100]
    now = now or datetime.now()
    path = path or _state_path()
    with _LOCK:
        data = _load(path)
        user = data["users"].setdefault(uid, {"observations": []})
        obs = list(user.get("observations") or [])
        obs.append({
            "label": str(label or NEUTRAL),
            "intensity": round(max(0.0, min(1.0, float(intensity or 0.0))), 2),
            "at": now.isoformat(timespec="seconds"),
            "surface": str(surface)[:40],
        })
        user["observations"] = _prune(obs, now)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)


def _decayed(o: dict[str, Any], now: datetime) -> float:
    at = _parse(o.get("at", ""))
    if at is None:
        return 0.0
    hours = max(0.0, (now - at).total_seconds() / 3600)
    return float(o.get("intensity") or 0.0) * math.pow(0.5, hours / HALF_LIFE_HOURS)


@dataclass(frozen=True)
class AffectRecall:
    label: str = NEUTRAL  # 減衰後もまだ残っている直近の様子
    weight: float = 0.0
    last_seen_at: datetime | None = None
    since_last: timedelta | None = None  # 最後の観測からの経過
    calmed_down: bool = False  # 気になる様子のあと「通常」が続いている
    recurrent: str = ""  # 直近7日で繰り返し出ている様子

    def to_payload(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "weight": round(self.weight, 2),
            "last_seen_at": self.last_seen_at.isoformat(timespec="seconds") if self.last_seen_at else "",
            "calmed_down": self.calmed_down,
            "recurrent": self.recurrent,
        }


def recall_user_affect(user_id: str, *, now: datetime | None = None, path: Path | None = None) -> AffectRecall:
    """減衰を考慮して、この相手の最近の様子をまとめる。"""
    now = now or datetime.now()
    data = _load(path or _state_path())
    obs = list((data["users"].get(str(user_id or "default")[:100]) or {}).get("observations") or [])
    obs = _prune(obs, now)
    if not obs:
        return AffectRecall()

    last_at = _parse(obs[-1].get("at", ""))
    weights: dict[str, float] = {}
    for o in obs:
        if o.get("label") != NEUTRAL:
            weights[o["label"]] = weights.get(o["label"], 0.0) + _decayed(o, now)
    label, weight = NEUTRAL, 0.0
    if weights:
        label, weight = max(weights.items(), key=lambda kv: kv[1])
        if weight < _MIN_WEIGHT:
            label, weight = NEUTRAL, 0.0

    # 最後の気になる様子のあと「通常」が2回以上続いていれば、落ち着いたとみなす
    calmed = False
    trailing_neutral = 0
    for o in reversed(obs):
        if o.get("label") == NEUTRAL:
            trailing_neutral += 1
        else:
            calmed = trailing_neutral >= 2
            break

    window_start = now - _RECURRENT_WINDOW
    counts: dict[str, int] = {}
    for o in obs:
        at = _parse(o.get("at", ""))
        if o.get("label") != NEUTRAL and at and at >= window_start:
            counts[o["label"]] = counts.get(o["label"], 0) + 1
    recurrent = ""
    if counts:
        top, n = max(counts.items(), key=lambda kv: kv[1])
        if n >= _RECURRENT_COUNT:
            recurrent = top

    return AffectRecall(
        label=label,
        weight=weight,
        last_seen_at=last_at,
        since_last=(now - last_at) if last_at else None,
        calmed_down=calmed,
        recurrent=recurrent,
    )


def _ago(delta: timedelta | None) -> str:
    if delta is None:
        return "以前"
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return "さっき"
    if hours < 20:
        return f"{int(hours)}時間ほど前"
    days = max(1, round(hours / 24))
    return "昨日" if days == 1 else f"{days}日前"


def build_user_affect_memory_block(recall: AffectRecall, *, current_label: str = NEUTRAL) -> str:
    """前回までの様子を、次の会話での気遣い方の指示にする。何も無ければ空文字。"""
    lines: list[str] = []
    new_session = recall.since_last is not None and recall.since_last >= _NEW_SESSION_GAP
    if recall.label != NEUTRAL and not recall.calmed_down:
        when = _ago(recall.since_last)
        if new_session:
            lines.append(
                f"- {when}の会話では「{recall.label}」の様子だった（減衰後の残り {recall.weight:.2f}）。"
                "今回の最初の返答で一言だけ、さりげなく様子を気遣ってよい（例: 「その後どうですか」）。"
                "細かく蒸し返したり、決めつけたりしない。2回目以降の返答では触れない。"
            )
        elif current_label == NEUTRAL:
            lines.append(
                f"- この会話の少し前は「{recall.label}」の様子だった。今の発言は落ち着いて見えるので、"
                "トーンは急に変えず、穏やかさを少し残す程度にする。"
            )
    elif recall.calmed_down and new_session:
        lines.append("- 以前は気になる様子もあったが、その後は落ち着いている。蒸し返さず普段どおりに話す。")
    if recall.recurrent and recall.recurrent != current_label:
        lines.append(
            f"- ここ1週間、「{recall.recurrent}」の様子が繰り返し出ている。指摘はしないが、"
            "負担を増やさない返し方（要点を絞る・次の一手を1つにする）を意識する。"
        )
    elif recall.recurrent and recall.recurrent == current_label:
        lines.append(
            f"- 「{recall.recurrent}」はここ1週間で繰り返し出ている。今回も同じなら、"
            "根本の負担を減らせる提案（優先順位の整理・後回しにしてよいこと）を1つだけ添えてよい。"
        )
    if not lines:
        return ""
    return (
        "【相手の最近の様子（記憶・時間とともに薄れる）】\n"
        + "\n".join(lines)
        + "\n- 様子の記録は推定であり、本人に「記録している」とは言わない。事実・審査判断には使わない。"
    )


def remember_and_build_block(
    user_id: str,
    label: str,
    intensity: float,
    *,
    surface: str,
    now: datetime | None = None,
    path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """過去の様子を読んでブロックを作ってから、今回の推定を記録する（今回分は前回扱いしない）。"""
    recall = recall_user_affect(user_id, now=now, path=path)
    block = build_user_affect_memory_block(recall, current_label=label)
    record_user_affect(user_id, label, intensity, surface=surface, now=now, path=path)
    return block, recall.to_payload()
