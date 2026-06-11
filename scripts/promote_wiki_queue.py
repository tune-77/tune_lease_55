#!/usr/bin/env python3
"""Promote queued Obsidian notes into curated wiki knowledge."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def latest_queue_path(root: Path, latest_path: Path) -> Path:
    latest = load_json(latest_path)
    raw = ((latest.get("wiki_promotion_queue") or {}).get("path") or "").strip()
    if raw:
        return Path(raw)
    queues = sorted((root / "reports").glob("wiki_promotion_queue_*.json"))
    if not queues:
        raise SystemExit("No wiki_promotion_queue_*.json found.")
    return queues[-1]


def promotion_id(source_note: str) -> str:
    digest = hashlib.sha1(source_note.encode("utf-8")).hexdigest()[:12]
    return f"WIKI-PROMOTE-{digest}"


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def target_rel_path(raw_target: str) -> str:
    target = str(raw_target or "Projects/tune_lease_55/tune_lease_55 Wiki.md").strip()
    if not target:
        target = "Projects/tune_lease_55/tune_lease_55 Wiki.md"
    if target.endswith("/"):
        return target.rstrip("/") + "/Promoted Knowledge.md"
    if not target.lower().endswith(".md"):
        return target + ".md"
    return target


def split_sections(text: str) -> list[list[str]]:
    sections: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("#") and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)
    return sections


def clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\s*[-*]\s+", "", line)
    line = re.sub(r"\s+", " ", line)
    return line[:260]


def extract_reusable_points(text: str, limit: int = 8) -> list[str]:
    points: list[str] = []
    seen: set[str] = set()

    for section in split_sections(text):
        section_text = "\n".join(section)
        if not any(keyword in section_text for keyword in REUSABLE_KEYWORDS):
            continue
        for line in section:
            cleaned = clean_line(line)
            if len(cleaned) < 10:
                continue
            if not any(keyword in cleaned for keyword in REUSABLE_KEYWORDS):
                continue
            if cleaned in seen:
                continue
            points.append(cleaned)
            seen.add(cleaned)
            if len(points) >= limit:
                return points

    if not points:
        for line in text.splitlines():
            cleaned = clean_line(line)
            if len(cleaned) >= 10 and cleaned not in seen:
                points.append(cleaned)
                seen.add(cleaned)
            if len(points) >= min(3, limit):
                break
    return points


def safe_note_path(vault: Path, rel: str) -> Path:
    target = (vault / rel).resolve()
    if vault not in target.parents and target != vault:
        raise SystemExit(f"Refusing to write outside vault: {rel}")
    if target.suffix.lower() != ".md":
        target = target.with_suffix(".md")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def to_wikilink(rel_path: str, alias: str | None = None) -> str:
    stem = rel_path[:-3] if rel_path.lower().endswith(".md") else rel_path
    return f"[[{stem}|{alias}]]" if alias else f"[[{stem}]]"


def build_section(item: dict[str, Any], source_text: str, target_rel: str) -> str:
    source_note = str(item.get("source_note") or "")
    prom_id = promotion_id(source_note)
    today = dt.date.today().isoformat()
    points = extract_reusable_points(source_text)
    title = str(item.get("title") or Path(source_note).stem or "Wiki Promotion").strip()
    excerpt = str(item.get("excerpt") or "").strip()
    reason = str(item.get("reason") or "").strip()

    lines = [
        f"## {today} {title} ({prom_id})",
        "",
        "### Summary",
        f"- 昇格元: {to_wikilink(source_note, Path(source_note).stem)}",
        f"- 昇格先: {target_rel}",
    ]
    if excerpt:
        lines.append(f"- 抽出起点: {excerpt}")
    if reason:
        lines.append(f"- 選定理由: {reason}")

    lines.extend(["", "### Reusable Points"])
    for point in points:
        lines.append(f"- {point}")

    lines.extend(
        [
            "",
            "### Source Metadata",
            f"- promotion_id: {prom_id}",
            f"- promoted_at: {dt.datetime.now().isoformat(timespec='seconds')}",
            "- source_notes:",
            f"  - {source_note}",
            "- used_wiki_pages:",
            f"  - {source_note}",
            "",
            "### Related",
            "- [[Projects/tune_lease_55/tune_lease_55 Wiki|tune_lease_55 Wiki]]",
            f"- {to_wikilink(source_note, Path(source_note).stem)}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def load_bridge():
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from mobile_app.obsidian_bridge import find_vault
    except Exception as exc:
        raise SystemExit(f"Obsidian bridge unavailable: {exc}") from exc
    return find_vault


def promote_queue(
    queue_path: Path,
    status_path: Path,
    latest_path: Path,
    *,
    limit: int,
    dry_run: bool,
) -> dict[str, Any]:
    find_vault = load_bridge()
    vault = find_vault()
    if not vault:
        raise SystemExit("iCloud 上の Obsidian Vault が見つかりません。")

    queue = load_json(queue_path)
    status = load_json(status_path)
    promoted_items = status.setdefault("items", {})
    results: list[dict[str, Any]] = []

    for item in (queue.get("items") or [])[:limit]:
        if not isinstance(item, dict):
            continue
        source_note = str(item.get("source_note") or "")
        if not source_note:
            continue
        prom_id = promotion_id(source_note)
        if prom_id in promoted_items:
            results.append({"promotion_id": prom_id, "source_note": source_note, "status": "skipped_duplicate"})
            continue

        source_path = safe_note_path(vault, source_note)
        if not source_path.exists():
            results.append({"promotion_id": prom_id, "source_note": source_note, "status": "missing_source"})
            continue
        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        target_rel = target_rel_path(str(item.get("suggested_target") or ""))
        target_path = safe_note_path(vault, target_rel)
        section = build_section(item, source_text, target_rel)

        if target_path.exists() and prom_id in target_path.read_text(encoding="utf-8", errors="ignore"):
            promoted_items[prom_id] = {
                "source_note": source_note,
                "target_note": target_rel,
                "status": "already_present",
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            }
            results.append({"promotion_id": prom_id, "source_note": source_note, "target_note": target_rel, "status": "already_present"})
            continue

        if dry_run:
            results.append(
                {
                    "promotion_id": prom_id,
                    "source_note": source_note,
                    "target_note": target_rel,
                    "status": "dry_run",
                    "preview": section[:700],
                }
            )
            continue

        prefix = "\n\n" if target_path.exists() and target_path.read_text(encoding="utf-8", errors="ignore").strip() else ""
        with target_path.open("a", encoding="utf-8") as handle:
            handle.write(prefix + section)
        promoted_items[prom_id] = {
            "source_note": source_note,
            "target_note": target_rel,
            "status": "promoted",
            "source_hash": source_hash(source_text),
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        results.append({"promotion_id": prom_id, "source_note": source_note, "target_note": target_rel, "status": "promoted"})

    promoted_count = sum(1 for item in results if item.get("status") == "promoted")
    skipped_count = sum(1 for item in results if str(item.get("status", "")).startswith("skipped") or item.get("status") == "already_present")
    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "queue_path": str(queue_path),
        "dry_run": dry_run,
        "processed_count": len(results),
        "promoted_count": promoted_count,
        "skipped_count": skipped_count,
        "results": results,
    }

    if not dry_run:
        status["updated_at"] = summary["generated_at"]
        status["last_queue_path"] = str(queue_path)
        dump_json(status_path, status)
        latest = load_json(latest_path)
        latest["wiki_promotion_execution"] = {
            "status": "PROMOTED" if promoted_count else "NO_NEW_PROMOTIONS",
            "promoted_count": promoted_count,
            "skipped_count": skipped_count,
            "processed_count": len(results),
            "status_path": str(status_path),
            "updated_at": summary["generated_at"],
        }
        dump_json(latest_path, latest)
    return summary


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--latest", type=Path, default=root / "reports" / "latest.json")
    parser.add_argument("--status-file", type=Path, default=root / "reports" / "wiki_promotion_status.json")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    queue_path = args.queue or latest_queue_path(root, args.latest)
    summary = promote_queue(
        queue_path,
        args.status_file,
        args.latest,
        limit=max(0, args.limit),
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
