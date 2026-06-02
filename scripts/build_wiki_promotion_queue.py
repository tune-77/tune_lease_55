#!/usr/bin/env python3
"""Build a small queue of Obsidian notes worth promoting into the wiki."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any


REUSABLE_KEYWORDS = [
    "判断",
    "再利用",
    "審査",
    "リスク",
    "承認条件",
    "否認",
    "稟議",
    "物件",
    "残価",
    "再販",
    "金利",
    "改善",
    "方針",
    "教訓",
    "失敗",
    "次アクション",
]

SOURCE_BONUS = {
    "Projects/tune_lease_55/AI Chat": 18,
    "Projects/tune_lease_55/Cases": 16,
    "Projects/tune_lease_55/AI Chat/Improvement Log": 14,
    "Daily": 10,
    "Projects/tune_lease_55/AI Chat/Weekly Review": 8,
}

CHAT_LOG_MARKERS = (
    "AI Chat",
    "Daily",
    "Improvement Log",
    "Weekly Review",
    "Chat/",
    "Debates/",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_excerpt(text: str, limit: int = 260) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    for line in lines:
        if any(keyword in line for keyword in REUSABLE_KEYWORDS):
            return line[:limit]
    return lines[0][:limit]


def slugify(value: str) -> str:
    text = re.sub(r"[^\wぁ-んァ-ン一-龥-]+", "-", value.lower()).strip("-")
    return text[:60] or "wiki-promotion"


def source_bonus(path: str) -> int:
    for prefix, bonus in SOURCE_BONUS.items():
        if path.startswith(prefix):
            return bonus
    return 0


def is_candidate_source(path: str) -> bool:
    return any(marker in path for marker in CHAT_LOG_MARKERS)


def target_for(path: str, text: str) -> str:
    hay = f"{path} {text}"
    if any(k in hay for k in ("物件", "残価", "再販", "中古", "Asset")):
        return "Projects/tune_lease_55/Asset Knowledge/"
    if any(k in hay for k in ("案件", "承認", "否認", "スコア", "金利", "稟議")):
        return "Projects/tune_lease_55/Cases/"
    if any(k in hay for k in ("改善", "実装", "不具合", "修正")):
        return "Projects/tune_lease_55/tune_lease_55 Wiki.md"
    return "Projects/tune_lease_55/tune_lease_55 Wiki.md"


def score_document(doc: dict[str, str]) -> tuple[int, list[str]]:
    path = str(doc.get("path") or "")
    text = f"{doc.get('title', '')}\n{doc.get('content', '')}"
    reasons: list[str] = []
    score = source_bonus(path)
    if score:
        reasons.append("source_priority")
    hits = [keyword for keyword in REUSABLE_KEYWORDS if keyword in text]
    if hits:
        score += min(42, len(hits) * 6)
        reasons.append("reusable_terms: " + ", ".join(hits[:5]))
    if "[[" in text:
        score += 8
        reasons.append("has_wikilinks")
    if "###" in text or "##" in text:
        score += 4
        reasons.append("structured_note")
    if len(text) < 120:
        score -= 12
        reasons.append("too_short")
    return score, reasons


def load_documents() -> list[dict[str, str]]:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from mobile_app.obsidian_bridge import iter_indexed_obsidian_documents
    except Exception as exc:
        raise SystemExit(f"Obsidian bridge unavailable: {exc}") from exc
    return iter_indexed_obsidian_documents(include_chat_logs=True, max_chars=1600)


def build_queue(limit: int) -> dict[str, Any]:
    docs = load_documents()
    candidates: list[dict[str, Any]] = []
    for doc in docs:
        path = str(doc.get("path") or "")
        if not path or not is_candidate_source(path):
            continue
        score, reasons = score_document(doc)
        if score < 20:
            continue
        title = str(doc.get("title") or Path(path).stem)
        content = str(doc.get("content") or "")
        candidates.append(
            {
                "title": title[:80],
                "source_note": path,
                "suggested_target": target_for(path, content),
                "score": score,
                "reason": "; ".join(reasons),
                "excerpt": normalize_excerpt(content),
                "mode": "review_then_promote",
                "prompt": (
                    f"{path} を確認し、再利用できる判断・ルール・教訓だけを "
                    f"{target_for(path, content)} へ昇格してください。"
                    " source_notes と related を必ず残してください。"
                ),
            }
        )

    candidates.sort(key=lambda item: (-int(item["score"]), item["source_note"]))
    queued = candidates[:limit]
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "date": dt.date.today().isoformat(),
        "limit": limit,
        "candidate_count": len(candidates),
        "queued_count": len(queued),
        "status": "READY" if queued else "EMPTY",
        "items": queued,
        "skipped_source_notes": [item["source_note"] for item in candidates[limit:]],
    }


def update_latest(latest_path: Path, queue_path: Path, queue: dict[str, Any]) -> None:
    latest = load_json(latest_path)
    latest["wiki_promotion_queue"] = {
        "status": queue.get("status"),
        "path": str(queue_path),
        "queued_count": queue.get("queued_count", 0),
        "candidate_count": queue.get("candidate_count", 0),
        "limit": queue.get("limit"),
        "generated_at": queue.get("generated_at"),
    }
    latest["wiki_promotion_queue_count"] = queue.get("queued_count", 0)
    dump_json(latest_path, latest)


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest", type=Path, default=root / "reports" / "latest.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report_date = dt.date.today().strftime("%Y%m%d")
    output_path = args.output or root / "reports" / f"wiki_promotion_queue_{report_date}.json"
    queue = build_queue(max(0, args.limit))

    if args.dry_run:
        print(json.dumps(queue, ensure_ascii=False, indent=2))
        return

    dump_json(output_path, queue)
    update_latest(args.latest, output_path, queue)
    print(
        "Wiki promotion queue: "
        f"{queue['queued_count']} queued / {queue['candidate_count']} candidates "
        f"({output_path})"
    )


if __name__ == "__main__":
    main()
