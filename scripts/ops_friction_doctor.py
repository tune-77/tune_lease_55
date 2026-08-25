#!/usr/bin/env python3
"""Find recurring operational friction before the next agent run."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATTERNS = (
    "memory/*.md",
    "reports/*_latest.md",
    ".claude/reports/*/latest.md",
)
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "ops_friction_latest.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "ops_friction_latest.md"

FRICTION_RULES = (
    {
        "id": "gitship_noise",
        "title": "Gitship dirty noise",
        "keywords": ("generated/runtime artifact", "dirty", "worktree", "Protected branch", "GH006", "direct push"),
        "command": "python scripts/classify_git_ship_candidates.py",
        "auto_command": "",
        "scope": "Gitship前",
    },
    {
        "id": "local_deploy_restart",
        "title": "Local deploy restart ambiguity",
        "keywords": ("run_next_stable", "LaunchAgent", "cloudflared", "RESTART_SCOPE", "uvicorn", "next-server"),
        "command": "python scripts/local_deploy_doctor.py --public-tunnel",
        "auto_command": "",
        "scope": "ローカル起動/再起動前",
    },
    {
        "id": "pr_review_followup",
        "title": "PR review follow-up",
        "keywords": ("Codex Review", "review comment", "REQUEST_CHANGES", "P1 Badge", "P2 Badge"),
        "command": "check PR review threads before merge or create a follow-up PR if already merged",
        "auto_command": "",
        "scope": "PR作成後/merge前",
    },
    {
        "id": "memory_pipeline_review",
        "title": "Memory pipeline review backlog",
        "keywords": ("needs_feedback", "needs_review", "human approval", "review_required", "promotion_queue"),
        "command": "python scripts/build_shion_memory_sentinel_report.py",
        "auto_command": "python scripts/build_shion_memory_sentinel_report.py",
        "scope": "記憶/判断資産パイプライン確認時",
    },
    {
        "id": "cloudrun_sync_gap",
        "title": "Cloud Run return sync gap",
        "keywords": ("Cloud Run", "GCS", "writeback", "materialize", "sync_cloudrun_inputs_from_gcs"),
        "required_groups": (("cloud run", "cloudrun"), ("writeback", "materialize", "sync_cloudrun_inputs")),
        "command": "CLOUDRUN_INPUT_SYNC_DAYS=2 python scripts/sync_cloudrun_inputs_from_gcs.py",
        "auto_command": "",
        "scope": "Cloud Runで入力/feedbackを使った後",
    },
)


@dataclass(frozen=True)
class LogHit:
    path: str
    count: int
    samples: list[str]


@dataclass(frozen=True)
class Finding:
    id: str
    title: str
    severity: str
    score: int
    scope: str
    reason: str
    command: str
    auto_command: str
    hits: list[LogHit]


@dataclass(frozen=True)
class AutoActionResult:
    finding_id: str
    command: str
    exit_code: int
    applied: bool
    output_tail: str = ""
    error_tail: str = ""


@dataclass(frozen=True)
class OpsFrictionReport:
    findings: list[Finding]
    auto_actions: list[AutoActionResult] = field(default_factory=list)


def run_git(args: list[str], root: Path = PROJECT_ROOT) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip()


def expand_logs(root: Path, patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            paths.append(path)
    paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return paths


def scan_logs(root: Path, patterns: Iterable[str], *, max_files: int = 40) -> dict[str, list[LogHit]]:
    results: dict[str, list[LogHit]] = {str(rule["id"]): [] for rule in FRICTION_RULES}
    for path in expand_logs(root, patterns)[:max_files]:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        rel = str(path.relative_to(root))
        lowered_lines = [line.lower() for line in lines]
        for rule in FRICTION_RULES:
            keywords = [str(keyword).lower() for keyword in rule["keywords"]]
            required_groups = [
                tuple(str(keyword).lower() for keyword in group)
                for group in rule.get("required_groups", ())
            ]
            matched: list[str] = []
            count = 0
            for raw, lowered in zip(lines, lowered_lines):
                if not any(keyword in lowered for keyword in keywords):
                    continue
                if required_groups and not all(any(keyword in lowered for keyword in group) for group in required_groups):
                    continue
                count += 1
                if len(matched) < 3:
                    matched.append(raw.strip()[:180])
            if count:
                results[str(rule["id"])].append(LogHit(path=rel, count=count, samples=matched))
    return results


def git_dirty_counts(root: Path = PROJECT_ROOT) -> dict[str, int]:
    status = run_git(["status", "--porcelain"], root)
    counts = {"total": 0, "generated_like": 0}
    generated_prefixes = (
        ".claude/reports/",
        "data/",
        "reports/",
        "static_data/",
        "frontend/public/",
    )
    for line in status.splitlines():
        path = line[3:] if len(line) > 3 else line
        counts["total"] += 1
        if path.startswith(generated_prefixes) or "_wt_" in path:
            counts["generated_like"] += 1
    return counts


def build_findings(root: Path = PROJECT_ROOT, patterns: Iterable[str] = DEFAULT_LOG_PATTERNS) -> list[Finding]:
    log_hits = scan_logs(root, patterns)
    dirty = git_dirty_counts(root)
    findings: list[Finding] = []
    for rule in FRICTION_RULES:
        rid = str(rule["id"])
        hits = log_hits.get(rid, [])
        hit_count = sum(item.count for item in hits)
        score = hit_count
        reason = "recent logs mention this pattern %s times" % hit_count
        if rid == "gitship_noise" and dirty["generated_like"]:
            score += dirty["generated_like"]
            reason += "; current dirty tree has %s generated-like files" % dirty["generated_like"]
        if score <= 0:
            continue
        if score >= 20:
            severity = "high"
        elif score >= 6:
            severity = "medium"
        else:
            severity = "low"
        findings.append(
            Finding(
                id=rid,
                title=str(rule["title"]),
                severity=severity,
                score=score,
                scope=str(rule["scope"]),
                reason=reason,
                command=str(rule["command"]),
                auto_command=str(rule.get("auto_command") or ""),
                hits=hits[:5],
            )
        )
    findings.sort(key=lambda item: ({"high": 0, "medium": 1, "low": 2}[item.severity], -item.score, item.id))
    return findings


def run_auto_command(command: str, root: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    args = shlex.split(command)
    if args and args[0] == "python":
        args[0] = sys.executable
    return subprocess.run(
        args,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def apply_safe_findings(
    findings: list[Finding],
    *,
    root: Path = PROJECT_ROOT,
    runner=run_auto_command,
) -> list[AutoActionResult]:
    results: list[AutoActionResult] = []
    for finding in findings:
        if not finding.auto_command:
            continue
        proc = runner(finding.auto_command, root)
        results.append(
            AutoActionResult(
                finding_id=finding.id,
                command=finding.auto_command,
                exit_code=int(proc.returncode),
                applied=proc.returncode == 0,
                output_tail=(proc.stdout or "").strip()[-500:],
                error_tail=(proc.stderr or "").strip()[-500:],
            )
        )
    return results


def write_report(report: OpsFrictionReport, *, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render(report), encoding="utf-8")


def render(report: OpsFrictionReport | list[Finding]) -> str:
    if isinstance(report, list):
        report = OpsFrictionReport(report)
    findings = report.findings
    lines = ["# Ops Friction Doctor", ""]
    if not findings:
        lines.append("No recurring operational friction found in the scanned logs.")
    else:
        for item in findings:
            lines.extend(
                [
                    "## %s [%s]" % (item.title, item.severity),
                    "- score: %s" % item.score,
                    "- scope: %s" % item.scope,
                    "- reason: %s" % item.reason,
                    "- next: %s" % item.command,
                ]
            )
            if item.auto_command:
                lines.append("- auto-safe: %s" % item.auto_command)
            for hit in item.hits[:3]:
                sample = " / ".join(text for text in hit.samples if text)
                lines.append("- log: %s (%s hits)%s" % (hit.path, hit.count, ": " + sample if sample else ""))
            lines.append("")
    if report.auto_actions:
        lines.extend(["", "## Auto-Safe Results"])
        for action in report.auto_actions:
            status = "applied" if action.applied else "failed"
            lines.append("- %s: %s exit=%s `%s`" % (action.finding_id, status, action.exit_code, action.command))
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--log-pattern", action="append", dest="log_patterns")
    parser.add_argument("--apply-safe", action="store_true", help="run allowlisted low-risk remediation commands")
    parser.add_argument("--output-json", type=Path, help="write JSON report")
    parser.add_argument("--output-md", type=Path, help="write Markdown report")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    patterns = tuple(args.log_patterns or DEFAULT_LOG_PATTERNS)
    findings = build_findings(PROJECT_ROOT, patterns)
    auto_actions = apply_safe_findings(findings) if args.apply_safe else []
    report = OpsFrictionReport(findings=findings, auto_actions=auto_actions)
    if args.output_json or args.output_md:
        write_report(
            report,
            json_path=(args.output_json or DEFAULT_OUTPUT_JSON).expanduser(),
            md_path=(args.output_md or DEFAULT_OUTPUT_MD).expanduser(),
        )
    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        print(render(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
