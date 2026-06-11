#!/usr/bin/env python3
"""Promote durable items from memory/YYYY-MM-DD.md into MEMORY.md.

The script scans daily raw logs for a "Promotable Items" section, filters out
boilerplate template text, deduplicates against a small state file, and appends
new long-term memory bullets to MEMORY.md with source references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = PROJECT_ROOT / "memory"
MEMORY_FILE = PROJECT_ROOT / "MEMORY.md"
STATE_FILE = MEMORY_DIR / "memory_sync_state.json"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")
_SECTION_RE = re.compile(
    r"^##\s+Promotable Items\s*$\n(.*?)(?=^##\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*\S)\s*$")
_IGNORED_ITEMS = {
    "preserve exact decisions, failure modes, and verification details when they matter.",
    "if a rule repeats across days, it should move to `memory.md`.",
    "keep long-term memory concise: fact, impact, next action.",
    "decision that should live in `memory.md`:",
    "context:",
    "action:",
    "result:",
    "next:",
}


@dataclass
class Promotion:
    source: Path
    text: str

    @property
    def key(self) -> str:
        normalized = re.sub(r"\s+", " ", self.text.strip().lower())
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def _load_state() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        keys = data.get("promoted_keys", [])
        return {str(key) for key in keys}
    except Exception:
        return set()


def _save_state(promoted_keys: set[str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "promoted_keys": sorted(promoted_keys),
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_memory_text() -> str:
    if not MEMORY_FILE.exists():
        return ""
    return MEMORY_FILE.read_text(encoding="utf-8", errors="ignore")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _is_boilerplate(text: str) -> bool:
    return _normalize(text) in _IGNORED_ITEMS


def _extract_promotions(path: Path) -> list[Promotion]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    m = _SECTION_RE.search(text)
    if not m:
        return []
    promotions: list[Promotion] = []
    for raw_line in m.group(1).splitlines():
        bullet = _BULLET_RE.match(raw_line)
        if not bullet:
            continue
        item = bullet.group(1).strip()
        if not item or _is_boilerplate(item):
            continue
        promotions.append(Promotion(source=path, text=item))
    return promotions


def _extract_existing_keys(memory_text: str) -> set[str]:
    keys: set[str] = set()
    for line in memory_text.splitlines():
        if not line.startswith("- [") and not line.startswith("- "):
            continue
        normalized = _normalize(line)
        if normalized:
            keys.add(hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12])
    return keys


def _render_section(promotions: list[Promotion]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "",
        f"## Auto Promotions {now}",
    ]
    for promo in promotions:
        rel = promo.source.relative_to(PROJECT_ROOT)
        lines.append(f"- [{promo.source.stem}] {promo.text}  (`{rel}`)")
    return "\n".join(lines) + "\n"


def sync_memory(dry_run: bool = False) -> dict[str, int]:
    if not MEMORY_DIR.exists():
        return {"scanned": 0, "promoted": 0, "skipped": 0}

    state_keys = _load_state()
    memory_text = _load_memory_text()
    existing_keys = _extract_existing_keys(memory_text)

    promotions: list[Promotion] = []
    scanned = 0
    for path in sorted(MEMORY_DIR.glob("*.md")):
        if path.name == "TEMPLATE.md":
            continue
        if path.name == MEMORY_FILE.name:
            continue
        if not _DATE_RE.match(path.name):
            continue
        scanned += 1
        promotions.extend(_extract_promotions(path))

    new_promotions: list[Promotion] = []
    for promo in promotions:
        key = promo.key
        if key in state_keys or key in existing_keys:
            continue
        new_promotions.append(promo)
        state_keys.add(key)

    if new_promotions and not dry_run:
        with MEMORY_FILE.open("a", encoding="utf-8") as f:
            f.write(_render_section(new_promotions))
        _save_state(state_keys)
    elif not dry_run:
        _save_state(state_keys)

    return {"scanned": scanned, "promoted": len(new_promotions), "skipped": max(0, len(promotions) - len(new_promotions))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote durable items from memory daily logs into MEMORY.md.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be promoted without writing.")
    args = parser.parse_args()

    result = sync_memory(dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
