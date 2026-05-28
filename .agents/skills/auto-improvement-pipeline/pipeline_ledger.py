"""
改善案の処理履歴を追跡するJSONL台帳。
キー: title + description の正規化後 SHA1
状態: applied / needs_review / rejected
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

LEDGER_PATH = Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl"


def _ensure_dir() -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)


def compute_key(title: str, description: str) -> str:
    """正規化してハッシュ化"""
    normalized = f"{title.strip().lower()}|{description.strip().lower()}"
    return hashlib.sha1(normalized.encode()).hexdigest()[:16]


def is_processed(key: str, cooldown_days: int = 7) -> tuple[bool, str]:
    """
    処理済みかチェック。

    - applied / needs_review → 常に True（再実行しない）
    - rejected → cooldown_days 経過後に再評価可能（False を返す）
    """
    if not LEDGER_PATH.exists():
        return False, ""

    now = datetime.datetime.now()
    latest: dict | None = None

    try:
        for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("key") == key:
                    latest = entry
            except json.JSONDecodeError:
                continue
    except OSError:
        return False, ""

    if latest is None:
        return False, ""

    status = latest.get("status", "")

    if status in ("applied", "needs_review"):
        return True, status

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
) -> None:
    """台帳に記録（追記）"""
    _ensure_dir()
    entry = {
        "key": key,
        "status": status,
        "title": title,
        "pr_url": pr_url,
        "reason": reason,
        "recorded_at": datetime.datetime.now().isoformat(),
    }
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_summary() -> dict[str, int]:
    """applied / needs_review / rejected の件数サマリ（最新ステータスで集計）"""
    if not LEDGER_PATH.exists():
        return {"applied": 0, "needs_review": 0, "rejected": 0, "total": 0}

    latest_status: dict[str, str] = {}

    try:
        for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                k = entry.get("key", "")
                s = entry.get("status", "")
                if k and s:
                    latest_status[k] = s
            except json.JSONDecodeError:
                continue
    except OSError:
        return {"applied": 0, "needs_review": 0, "rejected": 0, "total": 0}

    counts: dict[str, int] = {"applied": 0, "needs_review": 0, "rejected": 0}
    for s in latest_status.values():
        if s in counts:
            counts[s] += 1

    counts["total"] = sum(counts.values())
    return counts
