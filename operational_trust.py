"""Read-only operational trust summaries for practical lease AI use."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path, limit: int = 500) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    except OSError:
        return []
    return rows


def _parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:19])
    except ValueError:
        return None


def summarize_memory_usage(repo_root: Path, *, days: int = 14) -> dict[str, Any]:
    """Summarize memory-audit log without exposing raw questions."""
    path = repo_root / "data" / "case_memory_usage_log.jsonl"
    rows = _load_jsonl(path)
    since = datetime.now() - timedelta(days=max(1, days))
    recent = [row for row in rows if (_parse_dt(row.get("timestamp")) or datetime.min) >= since]
    surface_counts = Counter(str(row.get("surface") or "unknown") for row in recent)
    pdca_count = sum(1 for row in recent if row.get("pdca_applied"))
    judgment_count = sum(1 for row in recent if row.get("judgment_learning_used"))
    latest = max((_parse_dt(row.get("timestamp")) for row in rows), default=None)
    recent_items = []
    for row in recent[-8:]:
        refs = row.get("knowledge_refs") or []
        recent_items.append(
            {
                "timestamp": row.get("timestamp", ""),
                "surface": row.get("surface", "unknown"),
                "knowledge_ref_count": len(refs) if isinstance(refs, list) else 0,
                "pdca_applied": bool(row.get("pdca_applied")),
                "judgment_learning_used": bool(row.get("judgment_learning_used")),
                "question_hash": str(row.get("question_hash") or "")[:12],
            }
        )
    return {
        "source": str(path),
        "total": len(rows),
        "recent_days": days,
        "recent_total": len(recent),
        "pdca_applied_count": pdca_count,
        "judgment_learning_count": judgment_count,
        "latest_timestamp": latest.isoformat(timespec="seconds") if latest else "",
        "by_surface": dict(surface_counts.most_common()),
        "recent_items": recent_items,
    }


def summarize_pdca_rules(repo_root: Path) -> dict[str, Any]:
    from prompt_feedback import load_pdca_rules

    data = load_pdca_rules(str(repo_root / "data" / "pdca_ai_rules.json"))
    today = datetime.now().date()
    active = expiring = expired = inactive = 0
    rules: list[dict[str, Any]] = []
    for item in data.get("pdca_rule_meta") or []:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "active")
        expires_text = str(item.get("expires_at") or "")
        days_left: int | None = None
        if expires_text:
            try:
                days_left = (datetime.fromisoformat(expires_text[:10]).date() - today).days
            except ValueError:
                days_left = None
        is_active = status != "inactive" and (days_left is None or days_left >= 0)
        if not is_active:
            inactive += int(status == "inactive")
            expired += int(status != "inactive" and days_left is not None and days_left < 0)
        else:
            active += 1
            if days_left is not None and days_left <= 14:
                expiring += 1
        rules.append(
            {
                "rule": str(item.get("rule") or "")[:120],
                "source": item.get("source", ""),
                "status": "active" if is_active else "expired_or_inactive",
                "expires_at": expires_text,
                "days_left": days_left,
            }
        )
    return {
        "source": str(repo_root / "data" / "pdca_ai_rules.json"),
        "active": active,
        "expiring_soon": expiring,
        "expired": expired,
        "inactive": inactive,
        "manual_rule_count": int(data.get("manual_rule_count") or 0),
        "rules": rules[:12],
    }


def _frontmatter_status(text: str) -> str:
    if not text.startswith("---"):
        return "unknown"
    parts = text.split("---", 2)
    if len(parts) < 3:
        return "unknown"
    for line in parts[1].splitlines():
        if line.strip().startswith("status:"):
            return line.split(":", 1)[1].strip().strip('"')
    return "unknown"


def summarize_knowledge_corrections(vault: Path | None) -> dict[str, Any]:
    if not vault:
        return {"available": False, "total": 0, "needs_review": 0, "items": []}
    corrections_dir = (
        Path(vault)
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Knowledge Corrections"
    )
    if not corrections_dir.exists():
        return {
            "available": True,
            "source": str(corrections_dir),
            "total": 0,
            "needs_review": 0,
            "items": [],
        }
    items: list[dict[str, Any]] = []
    for path in sorted(corrections_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        text = path.read_text(encoding="utf-8", errors="ignore")
        status = _frontmatter_status(text)
        items.append(
            {
                "path": str(path),
                "name": path.name,
                "status": status,
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return {
        "available": True,
        "source": str(corrections_dir),
        "total": len(items),
        "needs_review": sum(1 for item in items if item["status"] == "needs_review"),
        "items": items[:10],
    }


def build_operational_trust_summary(repo_root: Path, vault: Path | None = None) -> dict[str, Any]:
    memory = summarize_memory_usage(Path(repo_root))
    pdca = summarize_pdca_rules(Path(repo_root))
    corrections = summarize_knowledge_corrections(vault)
    attention: list[str] = []
    if corrections.get("needs_review", 0):
        attention.append("knowledge_corrections_need_review")
    if pdca.get("expired", 0):
        attention.append("pdca_rules_expired")
    if pdca.get("expiring_soon", 0):
        attention.append("pdca_rules_expiring_soon")
    if memory.get("recent_total", 0) == 0:
        attention.append("memory_usage_log_not_recent")
    return {
        "status": "attention" if attention else "ok",
        "attention": attention,
        "memory_usage": memory,
        "pdca_rules": pdca,
        "knowledge_corrections": corrections,
    }
