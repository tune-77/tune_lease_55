"""
改善案の処理履歴を追跡するJSONL台帳。
キー: title + description の正規化後 SHA1
状態: applied / deleted / needs_review / parked / rejected
"""
from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

LEDGER_PATH = Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl"

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.improvement_state_resolver import load_latest_by_alias, load_latest_by_key  # noqa: E402


def _ensure_dir() -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)


def compute_key(title: str, description: str) -> str:
    """正規化してハッシュ化"""
    normalized = f"{title.strip().lower()}|{description.strip().lower()}"
    return hashlib.sha1(normalized.encode()).hexdigest()[:16]


def is_processed(
    key: str,
    cooldown_days: int = 7,
    needs_review_cooldown_days: int = 30,
) -> tuple[bool, str]:
    """
    処理済みかチェック。

    - applied → 常に True（再実行しない）
    - deleted → 常に True（ユーザーが一覧から削除した候補は復活させない）
    - needs_review → needs_review_cooldown_days 経過後に再評価可能（False を返す）
    - parked → needs_review_cooldown_days 経過後に再評価可能（False を返す）
    - rejected → cooldown_days 経過後に再評価可能（False を返す）
    """
    if not LEDGER_PATH.exists():
        return False, ""

    now = datetime.datetime.now()
    latest = load_latest_by_alias(LEDGER_PATH).get(key)
    if not latest:
        return False, ""

    status = latest.get("status", "")

    if status in {"applied", "deleted"}:
        return True, status

    if status in {"needs_review", "parked", "suppressed"}:
        recorded_at = latest.get("recorded_at", "")
        try:
            recorded_time = datetime.datetime.fromisoformat(recorded_at)
            if (now - recorded_time).days < needs_review_cooldown_days:
                return True, f"{status} (cooldown: {needs_review_cooldown_days}日)"
        except (ValueError, TypeError):
            pass

    if status == "rejected":
        recorded_at = latest.get("recorded_at", "")
        try:
            recorded_time = datetime.datetime.fromisoformat(recorded_at)
            if (now - recorded_time).days < cooldown_days:
                return True, f"rejected (cooldown: {cooldown_days}日)"
        except (ValueError, TypeError):
            pass

    return False, ""


def record(
    key: str,
    status: str,
    title: str,
    pr_url: str = "",
    reason: str = "",
    canonical_key: str = "",
) -> None:
    """台帳に記録（追記）"""
    _ensure_dir()
    entry = {
        "key": key,
        "status": status,
        "title": title,
        "canonical_key": canonical_key,
        "pr_url": pr_url,
        "reason": reason,
        "recorded_at": datetime.datetime.now().isoformat(),
    }
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_summary() -> dict[str, int]:
    """applied / deleted / needs_review / parked / rejected の件数サマリ（最新ステータスで集計）"""
    if not LEDGER_PATH.exists():
        return {"applied": 0, "deleted": 0, "needs_review": 0, "parked": 0, "rejected": 0, "total": 0}

    latest_status = {
        key: str(entry.get("status") or "")
        for key, entry in load_latest_by_key(LEDGER_PATH).items()
    }

    counts: dict[str, int] = {"applied": 0, "deleted": 0, "needs_review": 0, "parked": 0, "rejected": 0}
    for s in latest_status.values():
        if s in counts:
            counts[s] += 1

    counts["total"] = sum(counts.values())
    return counts
