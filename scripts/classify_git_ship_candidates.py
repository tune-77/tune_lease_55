#!/usr/bin/env python3
"""Classify dirty git paths before git-ship.

This is intentionally read-only. It does not stage, unstage, delete, or rewrite
files. The goal is to make the routine "commit code, avoid generated outputs"
decision explicit before running git-ship.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path


INCLUDE_PATTERNS = (
    ".agents/skills/**/*.py",
    ".agents/skills/**/SKILL.md",
    "api/**/*.py",
    "frontend/src/**/*",
    "mobile_app/**/*.py",
    "scripts/*.py",
    "scripts/*.sh",
    "scripts/**/*.py",
    "scripts/**/*.sh",
    "tests/*.py",
    "tests/**/*.py",
    "docs/*.md",
    "docs/**/*",
    "specs/**/*",
)

AVOID_PATTERNS = (
    "reports/*_latest.*",
    "reports/*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].*",
    "reports/*.json",
    "static_data/*.json",
    "data/*.json",
    "data/*.jsonl",
    "data/**/*.jsonl",
    "frontend/public/**/*.png",
    "frontend/public/**/*.jpg",
    "frontend/public/**/*.jpeg",
    "frontend/public/**/*.webp",
    "frontend/public/**/*.gif",
    "frontend/public/judgment-asset-graph/*",
    "memory/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].md",
)

REVIEW_PATTERNS = (
    "api/rule_engine/*.json",
    "config/**/*.json",
    "requirements*.txt",
    "package.json",
    "package-lock.json",
    "pyproject.toml",
    ".gitignore",
    "README*",
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}


@dataclass(frozen=True)
class Classification:
    path: str
    bucket: str
    reason: str


def _matches(path: str, patterns: tuple[str, ...]) -> str:
    normalized = path.replace("\\", "/")
    for pattern in patterns:
        if fnmatch.fnmatch(normalized, pattern):
            return pattern
    return ""


def classify_path(path: str) -> Classification:
    normalized = path.replace("\\", "/")
    suffix = Path(normalized).suffix.lower()

    pattern = _matches(normalized, AVOID_PATTERNS)
    if pattern:
        return Classification(normalized, "avoid", f"generated/runtime artifact: {pattern}")
    if suffix in IMAGE_SUFFIXES and normalized.startswith(("reports/", "frontend/public/", "dashboard_images/")):
        return Classification(normalized, "avoid", "generated image artifact")

    pattern = _matches(normalized, INCLUDE_PATTERNS)
    if pattern:
        return Classification(normalized, "include", f"source/test/doc path: {pattern}")

    pattern = _matches(normalized, REVIEW_PATTERNS)
    if pattern:
        return Classification(normalized, "review", f"configuration or rule file: {pattern}")

    return Classification(normalized, "review", "unclassified path; inspect before staging")


def parse_porcelain_line(line: str) -> str:
    raw = line[3:] if len(line) > 3 else line
    if " -> " in raw:
        raw = raw.rsplit(" -> ", 1)[1]
    return raw.strip()


def dirty_paths(cwd: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "git status failed")
    return [parse_porcelain_line(line) for line in result.stdout.splitlines() if line.strip()]


def render(classifications: list[Classification]) -> str:
    buckets = ("include", "review", "avoid")
    lines = ["# git-ship candidate classification", ""]
    for bucket in buckets:
        rows = [item for item in classifications if item.bucket == bucket]
        lines.append(f"## {bucket} ({len(rows)})")
        if not rows:
            lines.append("- none")
        for item in rows:
            lines.append(f"- `{item.path}` — {item.reason}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    args = parser.parse_args()

    paths = dirty_paths(args.cwd)
    print(render([classify_path(path) for path in paths]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
