#!/usr/bin/env python3
"""Build a small Codex auto-improvement queue from the latest report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


BLOCKED_KEYWORDS = [
    "db",
    "api連携",
    "帝国データバンク",
    "モデル",
    "ocr",
    "ポートフォリオ",
    "公平性",
    "バイアス",
    "migration",
    "kubernetes",
    "スコアリング",
    "認証",
    "セキュリティ",
]

SAFE_TITLE_KEYWORDS = [
    "表示",
    "文言",
    "ホーム",
    "入力",
    "質問",
    "マウスオーバー",
    "初期設定",
    "整理",
    "参照",
    "音声入力",
    "背景",
    "フローティングUI",
    "補助金",
]

EXECUTION_STATUS_FILE = "codex_auto_execution_status.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def latest_report_path(root: Path) -> Path:
    reports = sorted((root / "reports").glob("improvement_report_*.json"))
    if not reports:
        raise SystemExit("No reports/improvement_report_*.json files found.")
    return reports[-1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_execution_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"items": {}}
    try:
        data = load_json(path)
    except Exception:
        return {"items": {}}
    if not isinstance(data.get("items"), dict):
        data["items"] = {}
    return data


def quota_blocked_items(status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = status.get("items") or {}
    blocked: dict[str, dict[str, Any]] = {}
    for rev_id, record in items.items():
        if not isinstance(record, dict):
            continue
        if record.get("status") == "blocked_by_quota":
            blocked[str(rev_id)] = record
    return blocked


def item_text(item: dict[str, Any]) -> str:
    policy = item.get("auto_fix_policy") or {}
    return " ".join(
        str(value or "")
        for value in (
            item.get("id"),
            item.get("title"),
            item.get("reason"),
            item.get("detail"),
            item.get("description"),
            policy.get("reason"),
        )
    ).lower()


def is_blocked(item: dict[str, Any]) -> tuple[bool, str]:
    policy = item.get("auto_fix_policy") or {}
    risk = str(policy.get("risk") or "").lower()
    max_files = policy.get("max_files")
    text = item_text(item)
    hits = [keyword for keyword in BLOCKED_KEYWORDS if keyword.lower() in text]
    if risk == "high":
        return True, "risk=high"
    if max_files == 0:
        return True, "max_files=0"
    if hits:
        return True, "blocked_keyword: " + ", ".join(hits[:5])
    return False, ""


def is_codex_safe(item: dict[str, Any]) -> bool:
    policy = item.get("auto_fix_policy") or {}
    risk = str(policy.get("risk") or "").lower()
    max_files = policy.get("max_files")
    title = str(item.get("title") or "")
    if risk not in {"low", "medium"}:
        return False
    if max_files is not None and max_files > 1:
        return False
    return any(keyword in title for keyword in SAFE_TITLE_KEYWORDS)


def queue_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    order = item.get("recommended_order")
    if not isinstance(order, int):
        order = item.get("_source_index")
    if not isinstance(order, int):
        order = 9999
    return (order, str(item.get("id") or ""))


def queue_item(item: dict[str, Any]) -> dict[str, Any]:
    policy = item.get("auto_fix_policy") or {}
    title = str(item.get("title") or "")
    return {
        "id": item.get("id"),
        "title": title,
        "reason": item.get("reason") or policy.get("reason") or "",
        "detail": item.get("detail") or item.get("description") or "",
        "risk": policy.get("risk") or "",
        "max_files": policy.get("max_files"),
        "required_checks": policy.get("required_checks") or ["py_compile", "targeted_test"],
        "mode": "codex_dry_run_first",
        "prompt": f"{item.get('id')} {title} を小さく実装し、テスト後に差分を報告してください。data/models/.claude/state は触らないでください。",
    }


def build_queue(report: dict[str, Any], limit: int, execution_status: dict[str, Any] | None = None) -> dict[str, Any]:
    needs_review = [item for item in report.get("needs_review") or [] if isinstance(item, dict)]
    quota_blocked = quota_blocked_items(execution_status or {})
    safe: list[dict[str, Any]] = []
    maybe: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    quota_hold: list[dict[str, Any]] = []

    for index, item in enumerate(needs_review):
        item["_source_index"] = index
        rev_id = str(item.get("id") or "")
        if rev_id in quota_blocked:
            quota_hold.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "reason": "blocked_by_quota",
                    "last_attempted_at": quota_blocked[rev_id].get("updated_at", ""),
                    "detail": quota_blocked[rev_id].get("detail", ""),
                }
            )
            continue
        is_manual, reason = is_blocked(item)
        if is_manual:
            blocked.append({"id": item.get("id"), "title": item.get("title"), "reason": reason})
        elif is_codex_safe(item):
            safe.append(item)
        else:
            maybe.append(item)

    safe_sorted = sorted(safe, key=queue_sort_key)
    queued = safe_sorted[:limit]
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_date": report.get("date"),
        "source_generated_at": report.get("generated_at"),
        "limit": limit,
        "needs_review_count": len(needs_review),
        "codex_auto_safe_count": len(safe),
        "codex_auto_maybe_count": len(maybe),
        "manual_or_blocked_count": len(blocked),
        "blocked_by_quota_count": len(quota_hold),
        "queued_count": len(queued),
        "status": "READY" if queued else "EMPTY",
        "items": [queue_item(item) for item in queued],
        "skipped_safe_ids": [item.get("id") for item in safe_sorted[limit:]],
        "blocked_by_quota": quota_hold,
        "manual_or_blocked": blocked,
    }


def update_latest(latest_path: Path, queue_path: Path, queue: dict[str, Any]) -> None:
    if latest_path.exists():
        latest = load_json(latest_path)
    else:
        latest = {}
    latest["codex_auto_queue"] = {
        "status": queue.get("status"),
        "path": str(queue_path),
        "queued_count": queue.get("queued_count", 0),
        "safe_count": queue.get("codex_auto_safe_count", 0),
        "maybe_count": queue.get("codex_auto_maybe_count", 0),
        "manual_or_blocked_count": queue.get("manual_or_blocked_count", 0),
        "blocked_by_quota_count": queue.get("blocked_by_quota_count", 0),
        "limit": queue.get("limit"),
        "generated_at": queue.get("generated_at"),
    }
    latest["codex_auto_queue_count"] = queue.get("queued_count", 0)
    latest["codex_auto_safe_count"] = queue.get("codex_auto_safe_count", 0)
    dump_json(latest_path, latest)


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=None, help="Improvement report JSON.")
    parser.add_argument("--latest", type=Path, default=root / "reports" / "latest.json")
    parser.add_argument("--output", type=Path, default=None, help="Queue JSON path.")
    parser.add_argument("--status-file", type=Path, default=root / "reports" / EXECUTION_STATUS_FILE)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report_path = args.report or latest_report_path(root)
    report = load_json(report_path)
    execution_status = load_execution_status(args.status_file)
    report_date = str(report.get("date") or dt.date.today().isoformat()).replace("-", "")
    output_path = args.output or root / "reports" / f"codex_auto_queue_{report_date}.json"

    queue = build_queue(report, max(0, args.limit), execution_status)
    queue["source_report"] = str(report_path)
    queue["execution_status_file"] = str(args.status_file)

    if args.dry_run:
        print(json.dumps(queue, ensure_ascii=False, indent=2))
        return

    dump_json(output_path, queue)
    update_latest(args.latest, output_path, queue)
    print(
        "Codex auto queue: "
        f"{queue['queued_count']} queued / {queue['codex_auto_safe_count']} safe "
        f"({output_path})"
    )


if __name__ == "__main__":
    main()
