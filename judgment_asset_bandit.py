"""Bandit-style reward helpers for judgment asset ranking.

This module is deliberately small and deterministic. It learns only the display
priority of judgment assets from human feedback; it does not change scoring,
approval decisions, prompts, or canonical promotion status.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_OUTCOMES = {"used", "helped", "challenged", "rejected", "neutral"}
OUTCOME_REWARDS = {
    "helped": 3.0,
    "challenged": 0.8,
    "used": 0.2,
    "neutral": -0.1,
    "rejected": -4.0,
}
DEFAULT_FEEDBACK_PATH = Path(__file__).resolve().parent / "data" / "judgment_asset_usage_feedback.jsonl"


@dataclass(frozen=True)
class BanditSignal:
    asset_id: str
    reward_score: float
    exploration_bonus: float
    total_feedback: int
    counts: dict[str, int]
    last_feedback_at: str = ""

    @property
    def rank_bonus(self) -> float:
        return self.reward_score + self.exploration_bonus

    def as_dict(self) -> dict[str, Any]:
        return {
            "reward_score": round(self.reward_score, 4),
            "exploration_bonus": round(self.exploration_bonus, 4),
            "rank_bonus": round(self.rank_bonus, 4),
            "total_feedback": self.total_feedback,
            "counts": self.counts,
            "last_feedback_at": self.last_feedback_at,
            "mode": "bandit_style_display_ranking_only",
        }


def normalize_asset_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("JA-"):
        text = text[3:]
    if text.startswith("cr-"):
        text = text[3:]
    return text


def _feedback_asset_ids(row: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("rule_id", "judgment_asset_id", "asset_id", "candidate_id", "id"):
        raw = str(row.get(key) or "").strip()
        if not raw:
            continue
        ids.add(raw)
        normalized = normalize_asset_id(raw)
        if normalized:
            ids.add(normalized)
            ids.add(f"cr-{normalized}")
    return ids


def normalize_outcome(row: dict[str, Any]) -> str:
    outcome = str(row.get("outcome") or row.get("status") or row.get("feedback") or "").strip().lower()
    feedback_map = {
        "useful": "helped",
        "needs_fix": "challenged",
        "wrong": "rejected",
    }
    outcome = feedback_map.get(outcome, outcome)
    if outcome in VALID_OUTCOMES:
        return outcome
    if row.get("helped") is True:
        return "helped"
    if row.get("challenged") is True:
        return "challenged"
    if row.get("wrong") is True or row.get("rejected") is True:
        return "rejected"
    if row.get("used") is True:
        return "used"
    return ""


def read_feedback_rows(path: Path | str = DEFAULT_FEEDBACK_PATH) -> list[dict[str, Any]]:
    target = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _latest_timestamp(rows: list[dict[str, Any]]) -> str:
    values = [
        str(row.get("used_at") or row.get("timestamp") or row.get("date") or row.get("created_at") or "").strip()
        for row in rows
    ]
    return max([value for value in values if value], default="")


def build_bandit_signals(feedback_rows: list[dict[str, Any]]) -> dict[str, BanditSignal]:
    rows_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feedback_rows:
        outcome = normalize_outcome(row)
        if not outcome:
            continue
        enriched = {**row, "outcome": outcome}
        for asset_id in _feedback_asset_ids(row):
            rows_by_id[asset_id].append(enriched)

    signals: dict[str, BanditSignal] = {}
    total_events = sum(len(rows) for rows in rows_by_id.values())
    for asset_id, rows in rows_by_id.items():
        counts = Counter(str(row.get("outcome") or "") for row in rows)
        total_feedback = sum(counts.values())
        if total_feedback <= 0:
            continue
        reward_sum = sum(OUTCOME_REWARDS.get(outcome, 0.0) * count for outcome, count in counts.items())
        smoothed_average = reward_sum / (total_feedback + 3.0)
        confidence = min(1.0, total_feedback / 5.0)
        reward_score = smoothed_average * confidence
        exploration_bonus = math.sqrt(math.log(max(total_events, 2)) / (total_feedback + 1.0)) * 0.25
        signals[asset_id] = BanditSignal(
            asset_id=asset_id,
            reward_score=reward_score,
            exploration_bonus=exploration_bonus,
            total_feedback=total_feedback,
            counts={outcome: int(counts.get(outcome, 0)) for outcome in sorted(VALID_OUTCOMES)},
            last_feedback_at=_latest_timestamp(rows),
        )
    return signals


def signal_for_candidate(
    candidate: dict[str, Any],
    signals: dict[str, BanditSignal],
) -> BanditSignal:
    candidate_id = str(candidate.get("id") or "").strip()
    canonical_id = normalize_asset_id(candidate_id)
    for key in (candidate_id, canonical_id, f"cr-{canonical_id}" if canonical_id else ""):
        if key and key in signals:
            return signals[key]
    return BanditSignal(
        asset_id=candidate_id,
        reward_score=0.0,
        exploration_bonus=0.0,
        total_feedback=0,
        counts={outcome: 0 for outcome in sorted(VALID_OUTCOMES)},
    )


def append_feedback_event(
    *,
    asset_id: str,
    outcome: str,
    case_id: str = "",
    note: str = "",
    source: str = "screening",
    path: Path | str = DEFAULT_FEEDBACK_PATH,
) -> bool:
    normalized = normalize_outcome({"outcome": outcome})
    clean_id = str(asset_id or "").strip()
    if not clean_id or normalized not in VALID_OUTCOMES:
        return False
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "schema_version": "1",
        "rule_id": clean_id,
        "outcome": normalized,
        "case_id": str(case_id or "")[:120],
        "note": str(note or "")[:240],
        "source": source,
        "used_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return True
