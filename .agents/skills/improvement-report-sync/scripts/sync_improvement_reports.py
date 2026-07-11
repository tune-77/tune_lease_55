#!/usr/bin/env python3
"""Sync applied improvement IDs into the latest improvement reports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


KNOWN_APPLIED_TITLE_PATTERNS = [
    "担当者の最終判断を記録する登録ボタンがない",
    "AIチャット履歴の表示を制御する機能がない",
    "対話AIへの接続エラー",
    "企業登録後、結果登録画面に直前に登録した企業が表示されない",
    "ダッシュボードにニュースが表示されていない",
    "リース審査専門Wikiが参照するVault名",
    "紫苑」が過去の会話内容を記憶していない",
]

KNOWN_PARKED_TITLE_PATTERNS = [
    "同期とDB監査",
    "Q_riskは既存数式",
    "スコアリング外",
    "60-80帯",
    "40-60帯",
    "モデルドリフト",
    "キャリブレーション",
    "PSI/CSI",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def latest_report_path(root: Path) -> Path:
    reports = sorted((root / "reports").glob("improvement_report_*.json"))
    if not reports:
        raise SystemExit("No reports/improvement_report_*.json files found.")
    return reports[-1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def id_map(items: Iterable[dict]) -> dict[str, dict]:
    mapped: dict[str, dict] = {}
    for item in items:
        item_id = str(item.get("id") or "")
        if item_id and item_id not in mapped:
            mapped[item_id] = item
    return mapped


def normalize_text(value: str) -> str:
    return re.sub(r"[\s　・/（）()【】\[\]「」:：,，.。_-]+", "", value.lower())


def title_matches(item: dict, patterns: Iterable[str]) -> bool:
    text = normalize_text(" ".join([
        str(item.get("title") or ""),
        str(item.get("detail") or ""),
        str(item.get("reason") or ""),
    ]))
    return any(normalize_text(pattern) in text for pattern in patterns)


def applied_entry(item: dict) -> dict:
    return {
        "id": item.get("id"),
        "file": item.get("file"),
        "title": item.get("title"),
        "pr_url": item.get("pr_url"),
    }


def status_entry(item: dict, reason: str = "") -> dict:
    entry = applied_entry(item)
    if reason:
        entry["reason"] = reason
    return entry


def sync_report(
    report: dict,
    applied_ids: list[str],
    applied_titles: list[str],
    parked_ids: list[str],
    parked_titles: list[str],
) -> tuple[dict, list[str], list[str], list[str]]:
    applied_ids_set = set(applied_ids)
    parked_ids_set = set(parked_ids)
    needs_review = list(report.get("needs_review") or [])
    applied = list(report.get("applied") or [])
    parked = list(report.get("parked") or [])

    needs_map = id_map(needs_review)
    applied_map = id_map(applied)
    moved: list[str] = []
    parked_moved: list[str] = []
    skipped: list[str] = []

    for item_id in applied_ids:
        if item_id in applied_map:
            continue
        item = needs_map.pop(item_id, None)
        if item is None:
            skipped.append(item_id)
            continue
        moved.append(item_id)
        applied.append(applied_entry(item))

    for item in list(needs_map.values()):
        item_id = str(item.get("id") or "")
        if item_id in applied_ids_set or item_id in parked_ids_set:
            continue
        if title_matches(item, applied_titles):
            needs_map.pop(item_id, None)
            moved.append(item_id)
            applied.append(applied_entry(item))

    for item_id in parked_ids:
        item = needs_map.pop(item_id, None)
        if item is None:
            skipped.append(item_id)
            continue
        parked_moved.append(item_id)
        parked.append(status_entry(item, "実装タスクではなく継続監視・研究テーマとして保留"))

    for item in list(needs_map.values()):
        item_id = str(item.get("id") or "")
        if title_matches(item, parked_titles):
            needs_map.pop(item_id, None)
            parked_moved.append(item_id)
            parked.append(status_entry(item, "実装タスクではなく継続監視・研究テーマとして保留"))

    removed_ids = applied_ids_set | parked_ids_set | set(moved) | set(parked_moved)
    report["needs_review"] = [item for item in needs_review if str(item.get("id") or "") not in removed_ids]
    report["applied"] = applied
    if parked:
        report["parked"] = parked
    summary = report.get("summary") or {}
    summary["applied_count"] = len(applied)
    summary["needs_review_count"] = len(report["needs_review"])
    if parked:
        summary["parked_count"] = len(parked)
    report["summary"] = summary
    return report, moved, parked_moved, skipped


def sync_latest(
    latest: dict,
    applied_ids: list[str],
    applied_titles: list[str],
    parked_ids: list[str],
    parked_titles: list[str],
) -> dict:
    applied_ids_set = set(applied_ids)
    parked_ids_set = set(parked_ids)
    needs_review = list(latest.get("needs_review") or [])
    applied_improvements = list(latest.get("applied_improvements") or [])
    parked_improvements = list(latest.get("parked_improvements") or [])
    items = list(latest.get("items") or [])

    needs_map = id_map(needs_review)
    applied_map = id_map(applied_improvements)
    item_map = id_map(items)

    for item_id in applied_ids:
        item = item_map.get(item_id)
        if item is None:
            item = needs_map.get(item_id)
        if item is None:
            continue

        if item_id not in applied_map:
            applied_improvements.append(applied_entry(item))

        for existing in items:
            if str(existing.get("id") or "") == item_id:
                existing["status"] = "APPLIED"
                existing["reason"] = "改善済み登録済み"

    for item in needs_review:
        item_id = str(item.get("id") or "")
        if item_id in applied_ids_set or item_id in applied_map:
            continue
        if title_matches(item, applied_titles):
            applied_improvements.append(applied_entry(item))
            applied_ids_set.add(item_id)

    for item in needs_review:
        item_id = str(item.get("id") or "")
        if item_id in parked_ids_set or title_matches(item, parked_titles):
            parked_improvements.append(status_entry(item, "実装タスクではなく継続監視・研究テーマとして保留"))
            parked_ids_set.add(item_id)

    removed_ids = applied_ids_set | parked_ids_set
    latest["needs_review"] = [item for item in needs_review if str(item.get("id") or "") not in removed_ids]
    latest["applied_improvements"] = applied_improvements
    if parked_improvements:
        latest["parked_improvements"] = parked_improvements

    if items:
        latest["approved"] = sum(1 for item in items if item.get("status") in {"APPROVED", "AUTO_FIX_CANDIDATE"})
        latest["auto_fix_candidates"] = sum(1 for item in items if item.get("status") == "AUTO_FIX_CANDIDATE")
        latest["needs_review"] = [item for item in latest["needs_review"] if str(item.get("id") or "") not in removed_ids]
        latest["parked"] = sum(1 for item in items if item.get("status") == "PARKED")
        latest["rejected"] = sum(1 for item in items if item.get("status") == "REJECTED")
        latest["applied"] = sum(1 for item in items if item.get("status") == "APPLIED")
    else:
        latest["applied"] = len(applied_improvements)

    latest["applied_count"] = len(applied_improvements)
    latest["needs_review_count"] = len(latest.get("needs_review") or [])
    if parked_improvements:
        latest["parked_count"] = len(parked_improvements)
    latest["status"] = "COMPLETED" if latest["applied_count"] > 0 else "NO_APPLIED"
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Target improvement report JSON",
    )
    parser.add_argument(
        "--latest",
        type=Path,
        default=repo_root() / "reports" / "latest.json",
        help="Latest summary JSON to keep in sync",
    )
    parser.add_argument(
        "--applied",
        action="append",
        default=[],
        help="Improvement ID to mark as applied. Repeat for multiple IDs.",
    )
    parser.add_argument(
        "--applied-title",
        action="append",
        default=[],
        help="Title/detail substring to mark as applied. Repeat for multiple patterns.",
    )
    parser.add_argument(
        "--parked",
        action="append",
        default=[],
        help="Improvement ID to move out of needs_review as a parked monitoring/research item.",
    )
    parser.add_argument(
        "--parked-title",
        action="append",
        default=[],
        help="Title/detail substring to move out of needs_review as parked. Repeat for multiple patterns.",
    )
    parser.add_argument(
        "--from-report",
        action="store_true",
        help="Infer applied IDs from the report's applied/applied_improvements entries.",
    )
    parser.add_argument(
        "--include-known-cleanup",
        action="store_true",
        help="Apply known Shion cleanup title patterns for already-fixed or strategic-monitoring items.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned changes without writing files.")
    args = parser.parse_args()

    root = repo_root()
    report_path = args.report or latest_report_path(root)
    report = load_json(report_path)
    latest = load_json(args.latest) if args.latest.exists() else {}

    applied_ids = list(args.applied)
    if args.from_report or not applied_ids:
        applied_ids = [
            str(item.get("id") or "")
            for item in (report.get("applied") or report.get("applied_improvements") or [])
            if str(item.get("id") or "")
        ]

    applied_titles = list(args.applied_title)
    parked_titles = list(args.parked_title)
    if args.include_known_cleanup:
        applied_titles.extend(KNOWN_APPLIED_TITLE_PATTERNS)
        parked_titles.extend(KNOWN_PARKED_TITLE_PATTERNS)

    if not applied_ids and not applied_titles and not args.parked and not parked_titles:
        print("適用済み改善項目なし（スキップ）")
        return

    updated_report, moved, parked_moved, skipped = sync_report(
        report,
        applied_ids,
        applied_titles,
        args.parked,
        parked_titles,
    )
    updated_latest = sync_latest(
        latest,
        applied_ids,
        applied_titles,
        args.parked,
        parked_titles,
    ) if latest else {}

    if args.dry_run:
        print(json.dumps({
            "report": str(report_path),
            "latest": str(args.latest),
            "moved": moved,
            "parked": parked_moved,
            "skipped": skipped,
            "applied_count": updated_report.get("summary", {}).get("applied_count", 0),
            "needs_review_count": updated_report.get("summary", {}).get("needs_review_count", 0),
            "parked_count": updated_report.get("summary", {}).get("parked_count", 0),
        }, ensure_ascii=False, indent=2))
        return

    dump_json(report_path, updated_report)
    if updated_latest:
        dump_json(args.latest, updated_latest)

    print(f"Updated report: {report_path}")
    print(f"Updated latest: {args.latest}")
    print(f"Moved: {', '.join(moved) if moved else '(none)'}")
    print(f"Parked: {', '.join(parked_moved) if parked_moved else '(none)'}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
