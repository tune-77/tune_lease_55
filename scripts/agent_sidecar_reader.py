#!/usr/bin/env python3
"""Read-only sidecar bridge for Claude/agent reports.

This script does not run agents and does not change lease scoring, databases, or
model behavior. It only distills existing `.claude/reports` Markdown files into
repo-local artifacts that the lease system can read as advisory context.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_ROOT = PROJECT_ROOT / ".claude" / "reports"
OUT_MD = PROJECT_ROOT / "reports" / "agent_sidecar_brief.md"
OUT_JSON = PROJECT_ROOT / "reports" / "agent_sidecar_brief.json"

PREFERRED_REPORTS = (
    "scoring-audit/latest.md",
    "data-quality/latest.md",
    "rule-validation/latest.md",
    "impact-analysis/latest.md",
    "security/latest.md",
    "code-review/latest.md",
    "build/latest.md",
    "agent-team/asset_value_discussion.md",
    "agent-team/subsidy_plan.md",
    "agent-team/report_ux_plan.md",
    "agent-discussion/ux_debate_latest.md",
)

SECTION_NAMES = ("サマリー", "課題・リスク", "後続エージェントへの申し送り")


@dataclass
class SidecarReport:
    path: str
    agent: str
    task: str
    timestamp: str
    status: str
    summary: str
    risks: str
    handoff: str
    stale: bool


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"')
    return meta, text[match.end():]


def _extract_section(body: str, name: str, max_chars: int = 650) -> str:
    pattern = re.compile(
        rf"^##\s+{re.escape(name)}\s*$\n(?P<body>.*?)(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(body)
    if not match:
        return ""
    section = match.group("body").strip()
    section = re.sub(r"\n{3,}", "\n\n", section)
    return section[:max_chars].strip()


def _is_stale(timestamp: str, max_age_days: int = 30) -> bool:
    if not timestamp:
        return True
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(timestamp[: len(fmt)], fmt)
            return (datetime.now() - dt).days > max_age_days
        except ValueError:
            continue
    return True


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    for rel in PREFERRED_REPORTS:
        path = REPORT_ROOT / rel
        if path.exists():
            paths.append(path)
    for path in sorted(REPORT_ROOT.glob("*/latest.md")):
        if path not in paths:
            paths.append(path)
    return paths


def load_sidecar_reports() -> list[SidecarReport]:
    reports: list[SidecarReport] = []
    for path in _candidate_paths():
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        meta, body = _parse_frontmatter(raw)
        summary = _extract_section(body, "サマリー")
        risks = _extract_section(body, "課題・リスク")
        handoff = _extract_section(body, "後続エージェントへの申し送り")
        if not any((summary, risks, handoff)):
            # Narrative reports may not follow the schema; keep a tiny preview.
            summary = body.strip()[:450]
        rel = str(path.relative_to(PROJECT_ROOT))
        timestamp = meta.get("timestamp", "")
        reports.append(
            SidecarReport(
                path=rel,
                agent=meta.get("agent", path.parent.name),
                task=meta.get("task", path.stem),
                timestamp=timestamp,
                status=meta.get("status", "unknown"),
                summary=summary,
                risks=risks,
                handoff=handoff,
                stale=_is_stale(timestamp),
            )
        )
    return reports


def _format_block(title: str, text: str) -> str:
    if not text:
        return ""
    return f"**{title}:**\n{text.strip()}"


def build_markdown(reports: list[SidecarReport]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Agent Sidecar Brief",
        "",
        f"> Generated: {now} | source: `.claude/reports` | mode: read-only advisory",
        "",
        "## Operating Boundary",
        "- This brief is advisory context only.",
        "- Do not let sidecar reports update scores, models, production DBs, or final approvals directly.",
        "- Use findings as review prompts, RAG hints, or weekly PDCA inputs.",
        "",
        "## Reports",
    ]
    if not reports:
        lines.append("_No sidecar reports found._")
        return "\n".join(lines).strip() + "\n"

    for report in reports:
        stale_label = " / stale" if report.stale else ""
        lines.extend(
            [
                "",
                f"### {report.agent} ({report.status}{stale_label})",
                f"- Source: `{report.path}`",
                f"- Task: {report.task or '-'}",
                f"- Timestamp: {report.timestamp or '-'}",
            ]
        )
        for title, text in (
            ("Summary", report.summary),
            ("Risks", report.risks),
            ("Handoff", report.handoff),
        ):
            block = _format_block(title, text)
            if block:
                lines.extend(["", block])
    return "\n".join(lines).strip() + "\n"


def build_context(max_chars: int = 3000) -> str:
    """Return a compact advisory block for prompts or daily briefs."""
    markdown = build_markdown(load_sidecar_reports())
    return markdown[:max_chars].strip()


def write_outputs() -> tuple[Path, Path]:
    reports = load_sidecar_reports()
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_markdown(reports), encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "mode": "read-only advisory",
                "reports": [asdict(report) for report in reports],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return OUT_MD, OUT_JSON


def main() -> None:
    out_md, out_json = write_outputs()
    print(f"[agent_sidecar_reader] wrote: {out_md}")
    print(f"[agent_sidecar_reader] wrote: {out_json}")


if __name__ == "__main__":
    main()
