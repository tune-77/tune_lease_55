#!/usr/bin/env python3
"""Audit tune_lease_55 LaunchAgent installation drift.

Checks are read-only:
- every repo-managed plist exists in ~/Library/LaunchAgents;
- installed plists match the repo source of truth byte-for-byte;
- no extra com.tunelease*.plist exists outside the repo source of truth;
- no LaunchAgent hardcodes secret environment variables;
- no LaunchAgent launches Python via Anaconda directly;
- optionally, launchd has loaded each service.

The report intentionally prints secret key names only, never values.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import plistlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTALLED_DIR = Path.home() / "Library" / "LaunchAgents"

SECRET_ENV_KEYS = frozenset(
    {
        "DATABASE_URL",
        "ESTAT_APP_ID",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "SLACK_WEBHOOK_URL",
    }
)

ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass(frozen=True)
class Finding:
    level: str
    check: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"level": self.level, "check": self.check, "message": self.message}


def repo_plists(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Return all repo-managed tunelease LaunchAgent source plists."""
    candidates: list[Path] = []
    for directory in (repo_root / "launchd", repo_root / "scripts" / "launchd", repo_root / "scripts"):
        if directory.exists():
            candidates.extend(directory.glob("com.tunelease*.plist"))
    return sorted(candidates, key=lambda path: path.name)


def load_plist(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return plistlib.load(handle)


def _label(path: Path) -> str:
    try:
        return str(load_plist(path).get("Label") or path.stem)
    except Exception:
        return path.stem


def _env(data: dict[str, Any]) -> dict[str, str]:
    return {str(k): str(v) for k, v in (data.get("EnvironmentVariables") or {}).items()}


def _program_args(data: dict[str, Any]) -> list[str]:
    return [str(arg) for arg in (data.get("ProgramArguments") or [])]


def _uses_anaconda_python(data: dict[str, Any]) -> bool:
    args = _program_args(data)
    env = _env(data)
    candidates = args + [env.get("PYTHON_BIN", "")]
    return any("anaconda3/bin/python" in value for value in candidates)


def _secret_keys(data: dict[str, Any]) -> list[str]:
    return sorted(SECRET_ENV_KEYS & set(_env(data)))


def check_installed_matches_repo(
    installed_dir: Path = DEFAULT_INSTALLED_DIR,
    repo_root: Path = REPO_ROOT,
) -> list[Finding]:
    findings: list[Finding] = []
    sources = repo_plists(repo_root)
    source_names = {path.name for path in sources}
    installed = sorted(installed_dir.glob("com.tunelease*.plist")) if installed_dir.exists() else []
    installed_names = {path.name for path in installed}

    for src in sources:
        dst = installed_dir / src.name
        if not dst.exists():
            findings.append(
                Finding(ERROR, "installed_match", f"{src.name}: installed plist is missing")
            )
            continue
        if src.read_bytes() != dst.read_bytes():
            findings.append(
                Finding(ERROR, "installed_match", f"{src.name}: installed plist differs from repo")
            )

    for name in sorted(installed_names - source_names):
        findings.append(
            Finding(ERROR, "installed_match", f"{name}: installed plist has no repo source of truth")
        )

    if not findings:
        findings.append(
            Finding(INFO, "installed_match", f"{len(sources)} installed plist(s) match the repo")
        )
    return findings


def check_plist_policy(paths: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        try:
            data = load_plist(path)
        except Exception as exc:
            findings.append(Finding(ERROR, "plist_parse", f"{path.name}: plist parse failed: {exc}"))
            continue

        keys = _secret_keys(data)
        if keys:
            findings.append(
                Finding(
                    ERROR,
                    "secret_env",
                    f"{path.name}: secret environment key(s) are hardcoded in plist: {keys}",
                )
            )
        if _uses_anaconda_python(data):
            findings.append(
                Finding(
                    ERROR,
                    "python_interpreter",
                    f"{path.name}: launches Python via anaconda3 instead of the project venv",
                )
            )
    if not findings:
        findings.append(Finding(INFO, "plist_policy", "no secret env keys or Anaconda Python launchers found"))
    return findings


def _launchctl_print(label: str, *, uid: str | None = None) -> subprocess.CompletedProcess[str]:
    uid = uid or str(os.getuid())
    return subprocess.run(
        ["launchctl", "print", f"gui/{uid}/{label}"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


def check_loaded_state(paths: Iterable[Path], *, uid: str | None = None) -> list[Finding]:
    """Check whether repo-managed services are currently loaded in launchd.

    Missing launchctl or non-macOS environments are reported as a warning instead
    of an error so CI can run this script without macOS launchd.
    """
    findings: list[Finding] = []
    uid = uid or str(os.getuid())
    checked = 0
    for path in paths:
        label = _label(path)
        try:
            proc = _launchctl_print(label, uid=uid)
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            return [
                Finding(
                    WARNING,
                    "loaded_state",
                    f"launchctl state check skipped: {type(exc).__name__}",
                )
            ]
        checked += 1
        if proc.returncode != 0:
            findings.append(
                Finding(WARNING, "loaded_state", f"{label}: repo plist exists but service is not loaded")
            )

    if not findings:
        findings.append(Finding(INFO, "loaded_state", f"{checked} service(s) are loaded in launchd"))
    return findings


def run_audit(
    *,
    installed_dir: Path = DEFAULT_INSTALLED_DIR,
    repo_root: Path = REPO_ROOT,
    check_launchctl: bool = True,
    uid: str | None = None,
) -> dict[str, Any]:
    sources = repo_plists(repo_root)
    findings: list[Finding] = []
    findings.extend(check_installed_matches_repo(installed_dir=installed_dir, repo_root=repo_root))
    findings.extend(check_plist_policy(sources))
    if check_launchctl:
        findings.extend(check_loaded_state(sources, uid=uid))

    counts = {
        level: sum(1 for finding in findings if finding.level == level)
        for level in (ERROR, WARNING, INFO)
    }
    return {
        "repo_root": str(repo_root),
        "installed_dir": str(installed_dir),
        "repo_plist_count": len(sources),
        "user": getpass.getuser(),
        "counts": counts,
        "status": ERROR if counts[ERROR] else (WARNING if counts[WARNING] else "ok"),
        "findings": [finding.as_dict() for finding in findings],
    }


def format_report(report: dict[str, Any]) -> str:
    icons = {ERROR: "ERROR", WARNING: "WARN", INFO: "INFO"}
    lines = [
        "LaunchAgent audit",
        f"  repo plists: {report['repo_plist_count']}",
        f"  installed: {report['installed_dir']}",
        f"  status: {report['status'].upper()} "
        f"(error={report['counts']['error']} warning={report['counts']['warning']})",
        "",
    ]
    for finding in report["findings"]:
        lines.append(f"{icons.get(finding['level'], finding['level']).ljust(5)} [{finding['check']}] {finding['message']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--installed-dir", type=Path, default=DEFAULT_INSTALLED_DIR)
    parser.add_argument("--no-launchctl", action="store_true", help="skip live launchd loaded-state checks")
    parser.add_argument("--uid", help="launchd gui uid to inspect; defaults to current uid")
    parser.add_argument("--json", type=Path, help="write JSON report")
    args = parser.parse_args(argv)

    report = run_audit(
        installed_dir=args.installed_dir,
        repo_root=args.repo_root,
        check_launchctl=not args.no_launchctl,
        uid=args.uid,
    )
    print(format_report(report))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON: {args.json}")

    return 1 if report["counts"][ERROR] else 0


if __name__ == "__main__":
    raise SystemExit(main())
