#!/usr/bin/env python3
"""Validate the project's OKF-style Markdown + YAML frontmatter subset."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

ALLOWED_TYPES = {
    "lease_concept",
    "lease_rule",
    "asset_profile",
    "risk_signal",
    "agent_policy",
    "index",
}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}
ALLOWED_STATUS = {"active", "draft", "deprecated"}


def _load_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    try:
        import yaml

        loaded = yaml.safe_load(match.group(1)) or {}
    except Exception as exc:
        return {"__parse_error__": str(exc)}, raw[match.end():]
    if not isinstance(loaded, dict):
        return {"__parse_error__": "frontmatter is not a mapping"}, raw[match.end():]
    return loaded, raw[match.end():]


def _iter_markdown_paths(targets: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        target = target.expanduser()
        if target.is_dir():
            paths.extend(sorted(target.rglob("*.md")))
        elif target.is_file() and target.suffix.lower() == ".md":
            paths.append(target)
    return paths


def validate_file(path: Path) -> list[str]:
    meta, body = _load_frontmatter(path)
    errors: list[str] = []
    if "__parse_error__" in meta:
        return [f"frontmatter parse error: {meta['__parse_error__']}"]
    if not meta:
        return ["missing frontmatter"]

    note_type = str(meta.get("type") or "").strip()
    if not note_type:
        errors.append("missing required field: type")
    elif note_type not in ALLOWED_TYPES:
        errors.append(f"unsupported type: {note_type}")

    title = str(meta.get("title") or "").strip()
    if not title:
        errors.append("recommended field missing: title")

    tags = meta.get("tags")
    if tags is not None and not isinstance(tags, list):
        errors.append("tags must be a YAML list")

    related = meta.get("related")
    if related is not None and not isinstance(related, list):
        errors.append("related must be a YAML list")

    confidence = str(meta.get("confidence") or "").strip()
    if confidence and confidence not in ALLOWED_CONFIDENCE:
        errors.append(f"unsupported confidence: {confidence}")

    status = str(meta.get("status") or "").strip()
    if status and status not in ALLOWED_STATUS:
        errors.append(f"unsupported status: {status}")

    if not body.strip():
        errors.append("empty markdown body")

    return errors


def build_report(paths: list[Path]) -> dict[str, Any]:
    files = _iter_markdown_paths(paths)
    items = []
    failed = 0
    for path in files:
        errors = validate_file(path)
        if errors:
            failed += 1
        items.append({"path": str(path), "ok": not errors, "errors": errors})
    return {
        "status": "ok" if failed == 0 else "failed",
        "checked": len(files),
        "failed": failed,
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    args = parser.parse_args()

    report = build_report(args.paths)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"status={report['status']} checked={report['checked']} failed={report['failed']}")
        for item in report["items"]:
            if not item["ok"]:
                print(f"- {item['path']}: {'; '.join(item['errors'])}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

