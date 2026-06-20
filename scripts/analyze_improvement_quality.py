#!/usr/bin/env python3
"""改善レポートの品質評価スクリプト。

improvement_report_*.json の needs_review アイテムのうち、
Codex キューに乗り成功した割合を計算して
data/improvement_quality_log.jsonl に追記する（コミット禁止）。
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_execution_status(root: Path) -> dict[str, dict[str, Any]]:
    """codex_auto_execution_status.json を読み込む。"""
    path = root / "reports" / "codex_auto_execution_status.json"
    if not path.exists():
        return {}
    try:
        data = load_json(path)
    except Exception:
        return {}
    items = data.get("items") or {}
    return {str(k): v for k, v in items.items() if isinstance(v, dict)}


def load_queue_ids(root: Path) -> set[str]:
    """全 codex_auto_queue_*.json からキューに乗った REV ID を収集する。"""
    queued: set[str] = set()
    for path in (root / "reports").glob("codex_auto_queue_*.json"):
        try:
            data = load_json(path)
        except Exception:
            continue
        for item in data.get("items") or []:
            rev_id = str(item.get("id") or "").upper()
            if rev_id:
                queued.add(rev_id)
    return queued


def compute_quality(root: Path) -> dict[str, Any]:
    """最新レポートの needs_review アイテムで品質スコアを計算する。"""
    reports = sorted((root / "reports").glob("improvement_report_*.json"))
    if not reports:
        return {}
    report_path = reports[-1]
    try:
        report = load_json(report_path)
    except Exception:
        return {}

    needs_review = [i for i in (report.get("needs_review") or []) if isinstance(i, dict)]
    if not needs_review:
        return {
            "date": str(report.get("date") or dt.date.today().isoformat()),
            "report": report_path.name,
            "needs_review_count": 0,
            "queued_count": 0,
            "succeeded_count": 0,
            "quality_score": None,
            "note": "needs_review が空",
        }

    execution_status = load_execution_status(root)
    queued_ids = load_queue_ids(root)

    needs_review_ids = {str(i.get("id") or "").upper() for i in needs_review}
    queued_from_report = needs_review_ids & queued_ids

    succeeded = {
        rev_id
        for rev_id in queued_from_report
        if execution_status.get(rev_id, {}).get("status") in (
            "completed_pending_review", "merged"
        )
    }

    queued_count = len(queued_from_report)
    succeeded_count = len(succeeded)
    quality_score = round(succeeded_count / queued_count, 4) if queued_count > 0 else None

    return {
        "date": str(report.get("date") or dt.date.today().isoformat()),
        "report": report_path.name,
        "needs_review_count": len(needs_review),
        "queued_count": queued_count,
        "succeeded_count": succeeded_count,
        "quality_score": quality_score,
        "queued_ids": sorted(queued_from_report),
        "succeeded_ids": sorted(succeeded),
    }


def append_log(root: Path, record: dict[str, Any]) -> None:
    log_path = root / "data" / "improvement_quality_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    root = repo_root()
    record = compute_quality(root)
    if not record:
        print("improvement_report が見つかりません。スキップします。", flush=True)
        return

    record["computed_at"] = dt.datetime.now().isoformat(timespec="seconds")
    append_log(root, record)

    score = record.get("quality_score")
    score_str = f"{score:.1%}" if score is not None else "N/A（キューなし）"
    print(
        f"[analyze_improvement_quality] {record['report']}: "
        f"needs_review={record['needs_review_count']} / "
        f"queued={record['queued_count']} / "
        f"succeeded={record['succeeded_count']} / "
        f"quality={score_str}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
