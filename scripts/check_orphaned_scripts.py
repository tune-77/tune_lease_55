#!/usr/bin/env python3
"""Detect scripts/*.py entry points that no other tracked file ever calls.

A script counts as wired in when its filename (with or without the .py
suffix) appears in some other git-tracked file — a pipeline shell script,
a GitHub Actions workflow, or another Python module that imports/subprocesses
it. The script's own file and its own tests/test_<name>.py are excluded from
that search, since a script importing itself or being covered by a unit test
does not mean anything actually invokes it in production.

This is a read-only audit: it never edits scripts, pipelines, or workflows.
False negatives are expected for scripts invoked only from outside this repo
(e.g. a local launchd/cron job), so results are advisory, not authoritative.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports" / "orphaned_scripts_latest.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "orphaned_scripts_latest.md"

MAIN_GUARD_RE = re.compile(r"""if\s+__name__\s*==\s*['"]__main__['"]""")
TEXT_SUFFIXES = {
    ".py", ".sh", ".yml", ".yaml", ".md", ".toml", ".cfg", ".ini", ".json",
}


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    return [root / line for line in out.splitlines() if line]


def is_entry_point_script(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return bool(MAIN_GUARD_RE.search(text))


def discover_entry_point_scripts(scripts_dir: Path) -> list[Path]:
    if not scripts_dir.is_dir():
        return []
    return sorted(p for p in scripts_dir.glob("*.py") if p.is_file() and is_entry_point_script(p))


def read_text_corpus(paths: list[Path]) -> dict[Path, str]:
    corpus: dict[Path, str] = {}
    for path in paths:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            corpus[path] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return corpus


def own_test_path(script: Path, root: Path) -> Path:
    return root / "tests" / f"test_{script.stem}.py"


def find_references(
    script: Path, *, corpus: dict[Path, str], root: Path
) -> list[str]:
    """Return relative paths of tracked files that mention this script's name,
    excluding the script itself and its own test file."""
    variants = {script.name, script.stem}
    excluded = {script.resolve(), own_test_path(script, root).resolve()}
    hits: list[str] = []
    for path, text in corpus.items():
        if path.resolve() in excluded:
            continue
        if any(variant in text for variant in variants):
            hits.append(path.relative_to(root).as_posix())
    return sorted(hits)


def build_report(*, root: Path = REPO_ROOT, generated_at: str | None = None) -> dict[str, Any]:
    scripts = discover_entry_point_scripts(root / "scripts")
    files = tracked_files(root)
    corpus = read_text_corpus(files)

    orphaned: list[dict[str, Any]] = []
    wired: list[str] = []
    for script in scripts:
        rel = script.relative_to(root).as_posix()
        references = find_references(script, corpus=corpus, root=root)
        if references:
            wired.append(rel)
        else:
            orphaned.append({"script": rel, "has_test": own_test_path(script, root).exists()})

    orphaned.sort(key=lambda item: item["script"])

    return {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "read_only_orphaned_script_audit",
        "guardrail": "no script, pipeline, or workflow edit; advisory only",
        "caveat": (
            "スクリプトがリポジトリ外（cron/launchd等）から呼ばれている場合は検知できない。"
            "誤検知の可能性があるため、必ず人間が内容を確認してから対応すること。"
        ),
        "summary": {
            "entry_point_scripts": len(scripts),
            "wired": len(wired),
            "orphaned": len(orphaned),
        },
        "orphaned_scripts": orphaned,
    }


def build_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines: list[str] = []
    lines.append("# Orphaned Scripts Report")
    lines.append("")
    lines.append(f"- Generated at: `{report.get('generated_at')}`")
    lines.append(f"- Mode: `{report.get('mode')}`")
    lines.append(f"- Caveat: {report.get('caveat')}")
    lines.append(f"- Entry-point scripts scanned: {summary.get('entry_point_scripts', 0)}")
    lines.append(f"- Wired (referenced elsewhere): {summary.get('wired', 0)}")
    lines.append(f"- Orphaned (no reference found): {summary.get('orphaned', 0)}")
    lines.append("")
    lines.append("## Orphaned Scripts")
    orphaned = report.get("orphaned_scripts") or []
    for item in orphaned:
        test_note = "test exists" if item.get("has_test") else "no test"
        lines.append(f"- `{item['script']}` ({test_note})")
    if not orphaned:
        lines.append("- None detected.")
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(build_markdown(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true", help="print report, do not write files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(root=args.root.resolve())
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        write_outputs(report, output_json=args.output_json, output_md=args.output_md)
    summary = report["summary"]
    print(
        "orphaned_scripts "
        f"scanned={summary['entry_point_scripts']} "
        f"wired={summary['wired']} "
        f"orphaned={summary['orphaned']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
