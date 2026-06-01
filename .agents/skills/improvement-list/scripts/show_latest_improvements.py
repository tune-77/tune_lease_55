#!/usr/bin/env python3
"""Show and analyze the latest reports/improvement_report_YYYYMMDD.json."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


REPORT_RE = re.compile(r"improvement_report_(\d{8})\.json$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def latest_report(root: Path) -> Path:
    reports_dir = root / "reports"
    candidates: list[tuple[str, Path]] = []
    for path in reports_dir.glob("improvement_report_*.json"):
        match = REPORT_RE.match(path.name)
        if match:
            candidates.append((match.group(1), path))
    if not candidates:
        raise SystemExit("No reports/improvement_report_YYYYMMDD.json files found.")
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def item_risk(item: dict) -> str:
    return item.get("auto_fix_policy", {}).get("risk") or "unknown"


def item_reason(item: dict) -> str:
    return item.get("reason") or item.get("auto_fix_policy", {}).get("reason") or ""


def theme(title: str) -> str:
    rules = [
        ("ホーム画面", ["ホーム"]),
        ("ニュース・業界動向", ["ニュース", "業界動向", "業界情報"]),
        ("補助金", ["補助金"]),
        ("知識宇宙・Obsidian", ["知識宇宙", "Obsidian", "グラフビュー"]),
        ("条件付き承認・銀行支援", ["条件付き承認", "前受金", "銀行", "支援依頼書", "協調リース"]),
        ("入力改善", ["OCR", "音声入力", "数字入力", "入力"]),
        ("データ・API・モデル", ["DB", "API", "モデル", "スコアリング", "Kubernetes", "EDINET", "帝国データバンク"]),
    ]
    for label, needles in rules:
        if any(needle in title for needle in needles):
            return label
    return "その他"


def priority_score(item: dict) -> tuple[int, str]:
    title = item.get("title", "")
    reason = item_reason(item)
    risk = item_risk(item)
    if risk == "high":
        base = 70
    elif "対象ファイル未特定" in reason:
        base = 40
    else:
        base = 20
    if any(word in title for word in ["ホーム", "表示", "整理", "参照", "文言", "FAQ"]):
        base -= 20
    if any(word in title for word in ["DB", "API", "モデル", "スコアリング", "Kubernetes", "OCR"]):
        base += 25
    return (base, item.get("id", ""))


def main() -> None:
    root = repo_root()
    path = latest_report(root)
    data = json.loads(path.read_text(encoding="utf-8"))
    needs_review = data.get("needs_review", [])
    rejected = data.get("rejected", [])
    applied = data.get("applied", [])
    summary = data.get("summary", {})

    print(f"# 最新改善リスト")
    print()
    print(f"- file: `{path.relative_to(root)}`")
    print(f"- date: `{data.get('date', 'unknown')}`")
    print(f"- generated_at: `{data.get('generated_at', 'unknown')}`")
    print(f"- applied: `{summary.get('applied_count', len(applied))}`")
    print(f"- needs_review: `{summary.get('needs_review_count', len(needs_review))}`")
    print(f"- rejected: `{summary.get('rejected_count', len(rejected))}`")
    print()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in needs_review:
        grouped[theme(item.get("title", ""))].append(item)

    print("## テーマ別")
    for label in sorted(grouped):
        print(f"- {label}: {len(grouped[label])}件")
    print()

    print("## 着手しやすい候補")
    for item in sorted(needs_review, key=priority_score)[:12]:
        print(f"- `{item.get('id')}` {item.get('title')} ({item_risk(item)})")
    print()

    high_risk = [item for item in needs_review if item_risk(item) == "high"]
    print("## 高リスク/手動確認")
    for item in high_risk[:15]:
        print(f"- `{item.get('id')}` {item.get('title')} - {item_reason(item)}")
    if len(high_risk) > 15:
        print(f"- ...ほか {len(high_risk) - 15}件")
    print()

    print("## 要レビュー全件")
    for item in needs_review:
        print(f"- `{item.get('id')}` {item.get('title')} ({item_risk(item)})")
    print()

    if rejected:
        print("## 却下")
        for item in rejected:
            print(f"- `{item.get('id')}` {item.get('title')} - {item.get('reason', '')}")


if __name__ == "__main__":
    main()
