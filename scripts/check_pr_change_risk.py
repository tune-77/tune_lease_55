#!/usr/bin/env python3
"""Guard PRs against generated artifact drift and oversized changes.

This check is intentionally simple and deterministic:
- generated/runtime artifacts classified as ``avoid`` fail the check
- changed file count above the configured threshold fails the check
- added+deleted text lines above the configured threshold fails the check

Use ``origin/master...HEAD`` style ranges so merge-base comparison is used for
pull requests. The script is read-only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from classify_git_ship_candidates import Classification, classify_path  # noqa: E402


@dataclass(frozen=True)
class Numstat:
    path: str
    additions: int
    deletions: int


@dataclass(frozen=True)
class RiskReport:
    changed_files: list[str]
    generated_artifacts: list[Classification]
    total_text_changes: int
    max_files: int
    max_text_changes: int

    @property
    def failed(self) -> bool:
        return bool(self.generated_artifacts) or len(self.changed_files) > self.max_files or self.total_text_changes > self.max_text_changes


def _run_git(args: list[str], cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def changed_files(base: str) -> list[str]:
    out = _run_git(["diff", "--name-only", base])
    return [line.strip() for line in out.splitlines() if line.strip()]


def parse_numstat_line(line: str) -> Numstat | None:
    parts = line.split("\t")
    if len(parts) < 3:
        return None
    add_raw, del_raw, path = parts[0], parts[1], parts[-1]
    additions = int(add_raw) if add_raw.isdigit() else 0
    deletions = int(del_raw) if del_raw.isdigit() else 0
    return Numstat(path=path, additions=additions, deletions=deletions)


def changed_numstat(base: str) -> list[Numstat]:
    out = _run_git(["diff", "--numstat", base])
    rows: list[Numstat] = []
    for line in out.splitlines():
        parsed = parse_numstat_line(line)
        if parsed:
            rows.append(parsed)
    return rows


def build_report(files: list[str], numstat: list[Numstat], *, max_files: int, max_text_changes: int) -> RiskReport:
    classifications = [classify_path(path) for path in files]
    generated = [item for item in classifications if item.bucket == "avoid"]
    total = sum(row.additions + row.deletions for row in numstat)
    return RiskReport(
        changed_files=files,
        generated_artifacts=generated,
        total_text_changes=total,
        max_files=max_files,
        max_text_changes=max_text_changes,
    )


def render_report(report: RiskReport) -> str:
    lines = [
        "# PR Change Risk Guard",
        "",
        f"- changed_files: {len(report.changed_files)} / max {report.max_files}",
        f"- text_changes: {report.total_text_changes} / max {report.max_text_changes}",
        f"- generated_artifacts: {len(report.generated_artifacts)}",
    ]
    if report.generated_artifacts:
        lines.append("")
        lines.append("## Generated/runtime artifacts")
        for item in report.generated_artifacts[:50]:
            lines.append(f"- `{item.path}` — {item.reason}")
        if len(report.generated_artifacts) > 50:
            lines.append(f"- ... and {len(report.generated_artifacts) - 50} more")
    if len(report.changed_files) > report.max_files:
        lines.append("")
        lines.append(f"ERROR: changed file count exceeds limit ({len(report.changed_files)} > {report.max_files}).")
    if report.total_text_changes > report.max_text_changes:
        lines.append("")
        lines.append(f"ERROR: changed line count exceeds limit ({report.total_text_changes} > {report.max_text_changes}).")
    if report.generated_artifacts:
        lines.append("")
        lines.append("ERROR: generated/runtime artifacts must be stashed or committed separately.")
    if not report.failed:
        lines.append("")
        lines.append("OK: PR size and generated artifact checks passed.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/master...HEAD", help="git diff range/ref")
    parser.add_argument("--max-files", type=int, default=80)
    parser.add_argument("--max-text-changes", type=int, default=2500)
    args = parser.parse_args()

    files = changed_files(args.base)
    nums = changed_numstat(args.base)
    report = build_report(
        files,
        nums,
        max_files=args.max_files,
        max_text_changes=args.max_text_changes,
    )
    print(render_report(report))
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
