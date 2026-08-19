#!/usr/bin/env python3
"""Read-only sidecar bridge for Claude/agent reports.

This script does not run agents and does not change lease scoring, databases, or
model behavior. It only distills existing `.claude/reports` Markdown files into
repo-local artifacts that the lease system can read as advisory context.
"""

from __future__ import annotations

import json
import re
import subprocess
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
)

SECTION_NAMES = ("サマリー", "課題・リスク", "後続エージェントへの申し送り")

# Reports older than this are demoted out of the findings section (GAP-003).
STALE_AFTER_DAYS = 30

# Headings are load-bearing: downstream readers split the brief on them to keep
# stale findings out of prompt material. Changing them requires updating
# lease_intelligence_reflection._extract_sidecar_findings().
FINDINGS_HEADING = "## Reports"
RECHECK_LABEL = "再確認TODO"
RECHECK_HEADING = f"## {RECHECK_LABEL}"


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


# 変更駆動エージェントの「担当ファイル」。壁時計の経過日数ではなく、担当ファイルが
# レポートより後に変わったかで鮮度を判定する。出典は .claude/AGENTS.md の
# 「起動タイミング」欄。ここに無いディレクトリは経過日数で判定する（定期監査系で、
# 担当ファイルという概念が無いため）。
AGENT_TRIGGER_PATHS: dict[str, tuple[str, ...]] = {
    "scoring-audit": ("asset_scorer.py", "total_scorer.py", "category_config.py"),
    "rule-validation": ("rule_manager.py", "coeff_definitions.py", "category_config.py"),
    "migration": ("migrate_to_sqlite.py",),
    # 「コード変更発生」が起動条件のもの。静穏期には fresh のままにできる。
    "code-review": ("*.py", "frontend/src"),
    "security": ("*.py", "frontend/src"),
    "impact-analysis": ("*.py", "frontend/src"),
    "file-searcher": ("*.py", "frontend/src"),
    "report-stylist": ("components/analysis_results.py",),
}


def _trigger_last_changed(paths: tuple[str, ...], root: Path | None = None) -> datetime | None:
    """担当ファイルの最終コミット時刻。git が使えなければ None。"""
    if not paths:
        return None
    try:
        proc = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", *paths],
            cwd=str(root or PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 or not out.isdigit():
        return None
    return datetime.fromtimestamp(int(out))


def _parse_timestamp(timestamp: str) -> datetime | None:
    text = (timestamp or "").strip()
    if not text:
        return None
    for fmt, width in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text[:width], fmt)
        except ValueError:
            continue
    return None


def is_stale_for(report_dir: str, timestamp: str, root: Path | None = None) -> bool:
    """レポートが古いか。担当ファイルがあればそれ基準、無ければ経過日数。

    経過日数だけで見ると、その後まったく変更が無いレポートまで捨ててしまう。
    逆に閾値内でも担当ファイルが変わっていれば、その指摘はもう当てにならない。
    """
    triggers = AGENT_TRIGGER_PATHS.get(report_dir)
    if not triggers:
        return _is_stale(timestamp)
    written = _parse_timestamp(timestamp)
    if written is None:
        return True
    changed = _trigger_last_changed(triggers, root=root)
    if changed is None:
        # git が使えない環境では経過日数へ退避する（判定不能を fresh にしない）。
        return _is_stale(timestamp)
    return changed > written


# (書式, その書式が生成する文字数)。スライス幅は書式文字列の長さではなく
# 出力の長さで決まる（"%Y" は2文字だが 4桁を生成する）。len(fmt) で切ると
# どの書式もパースに失敗し、全レポートが恒久的に stale 扱いになる。
_TIMESTAMP_FORMATS = (
    ("%Y-%m-%d %H:%M", 16),
    ("%Y-%m-%dT%H:%M:%S", 19),
    ("%Y-%m-%d", 10),
)


def _is_stale(timestamp: str, max_age_days: int = STALE_AFTER_DAYS) -> bool:
    text = (timestamp or "").strip()
    if not text:
        return True
    for fmt, width in _TIMESTAMP_FORMATS:
        try:
            parsed = datetime.strptime(text[:width], fmt)
        except ValueError:
            continue
        return (datetime.now() - parsed).days > max_age_days
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
                stale=is_stale_for(path.parent.name, timestamp),
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
        f"- Reports older than {STALE_AFTER_DAYS} days are demoted to the"
        f" {RECHECK_LABEL} section and must be re-verified against current code"
        " before use.",
        "",
        FINDINGS_HEADING,
    ]
    if not reports:
        lines.append("_No sidecar reports found._")
        return "\n".join(lines).strip() + "\n"

    fresh = [r for r in reports if not r.stale]
    stale = [r for r in reports if r.stale]

    if not fresh:
        lines.append(
            f"_No fresh reports. All {len(stale)} report(s) are older than"
            f" {STALE_AFTER_DAYS} days — see {RECHECK_LABEL}._"
        )
    for report in fresh:
        lines.extend(
            [
                "",
                f"### {report.agent} ({report.status})",
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

    if stale:
        lines.extend(
            [
                "",
                RECHECK_HEADING,
                "",
                "古い指摘を現在の真実として扱わないこと。"
                "最新コードで再検証したものだけを有効扱いにする。",
                "",
            ]
        )
        for report in stale:
            lines.append(
                f"- [ ] `{report.agent}` — {report.timestamp or '時刻不明'} 時点"
                f" / `{report.path}`"
            )
            hint = (report.risks or report.summary or "").strip().splitlines()
            if hint:
                lines.append(f"  - 当時の指摘: {hint[0].strip(' -')[:120]}")
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
