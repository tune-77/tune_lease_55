#!/usr/bin/env python3
"""Build a read-only instruction debt report for agent guidance files.

The report looks for instructions that may be hard to delete later because they
do not carry nearby rationale or retirement conditions. It never edits prompts,
memory, skills, or source-of-truth reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports" / "instruction_debt_latest.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "instruction_debt_latest.md"

DEFAULT_SCAN_PATHS = (
    "AGENTS.md",
    "CLAUDE.md",
    "MEMORY.md",
    ".agents/skills",
    "shared-ai/skills",
)

RATIONALE_RE = re.compile(
    r"(理由|背景|目的|意図|影響|根拠|なぜ|because|rationale|reason|why|so that|in order to)",
    re.IGNORECASE,
)
SCOPE_RE = re.compile(
    r"(適用|対象|条件|場合|when|if|only|unless|applies|scope|trigger)",
    re.IGNORECASE,
)
RETIREMENT_RE = re.compile(
    r"(削除|廃止|解除|期限|失効|見直し|消せる|retire|remove|delete|expire|until|superseded)",
    re.IGNORECASE,
)
INSTRUCTION_RE = re.compile(
    r"("
    r"\b(?:must|always|never|required|forbidden|prefer|avoid|ask|use|read|write|check)\b|"
    r"\bdo not\b|\bdon't\b|"
    r"必ず|禁止|すること|しない|読|確認|使う|避ける|尋ね|書く|保存|削除|更新"
    r")",
    re.IGNORECASE,
)
STRONG_POLICY_RE = re.compile(
    r"(\b(?:must|should|never|required|forbidden)\b|\bdo not\b|\bdon't\b|必ず|禁止|しない|べき)",
    re.IGNORECASE,
)

LOW_SIGNAL_RE = re.compile(r"^\s*(---+|\*{3,}|#+\s*$)$")
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
METADATA_LINE_RE = re.compile(
    r"^\s*(?:理由|背景|目的|意図|影響|根拠|適用条件|対象|削除条件|見直し条件|"
    r"reason|rationale|scope|trigger|retirement|removal)\s*[:：]",
    re.IGNORECASE,
)
DATED_MEMORY_LINE_RE = re.compile(r"^\s*-\s*\[\d{4}-\d{2}-\d{2}\]")


@dataclass(frozen=True)
class Instruction:
    file: str
    line: int
    text: str
    rationale: bool
    scope: bool
    retirement: bool
    duplicate_count: int = 0

    @property
    def severity(self) -> str:
        if not self.rationale and not self.scope:
            return "high"
        if not self.rationale or not self.retirement:
            return "medium"
        return "low"

    @property
    def issue_codes(self) -> list[str]:
        codes: list[str] = []
        if not self.rationale:
            codes.append("missing_rationale")
        if not self.scope:
            codes.append("missing_scope")
        if not self.retirement:
            codes.append("missing_retirement_condition")
        if self.duplicate_count and not (self.rationale and self.scope and self.retirement):
            codes.append("possible_duplicate")
        return codes


def discover_files(root: Path, scan_paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for item in scan_paths:
        path = (root / item).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            continue
        if path.is_file() and path.suffix.lower() in {".md", ""}:
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("SKILL.md")))
    return sorted({path for path in files if path.exists()})


def extract_instructions(path: Path, root: Path) -> list[Instruction]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []

    rel = path.relative_to(root).as_posix()
    in_code = False
    in_frontmatter = False
    instructions: list[Instruction] = []
    for idx, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if idx == 1 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            continue
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not stripped or LOW_SIGNAL_RE.match(stripped) or METADATA_LINE_RE.match(stripped):
            continue
        if stripped.startswith("→"):
            continue
        if stripped.startswith("#"):
            continue
        if rel == "MEMORY.md" and DATED_MEMORY_LINE_RE.match(stripped) and not STRONG_POLICY_RE.search(stripped):
            continue
        if not is_instruction_line(stripped):
            continue
        window = instruction_context(lines, idx)
        section = section_context(lines, idx)
        instructions.append(
            Instruction(
                file=rel,
                line=idx,
                text=_collapse(stripped),
                rationale=bool(RATIONALE_RE.search(window) or RATIONALE_RE.search(section)),
                scope=bool(SCOPE_RE.search(window) or SCOPE_RE.search(section)),
                retirement=bool(RETIREMENT_RE.search(window) or RETIREMENT_RE.search(section)),
            )
        )
    return instructions


def instruction_context(lines: list[str], line_number: int) -> str:
    """Return the current instruction plus its immediately attached metadata."""
    current = lines[line_number - 1]
    current_indent = len(current) - len(current.lstrip(" "))
    context = [current]
    for raw_line in lines[line_number : min(len(lines), line_number + 5)]:
        stripped = raw_line.strip()
        if not stripped:
            break
        if LIST_PREFIX_RE.match(stripped) or stripped.startswith("#"):
            break
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent > current_indent or METADATA_LINE_RE.match(stripped):
            context.append(raw_line)
            continue
        break
    return "\n".join(context)


def section_context(lines: list[str], line_number: int) -> str:
    """Return section-level prose above an instruction for grouped rationale."""
    context: list[str] = []
    for raw_line in reversed(lines[: line_number - 1]):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if len(context) >= 25:
            break
        context.append(raw_line)
    return "\n".join(reversed(context))


def is_instruction_line(line: str) -> bool:
    if line.startswith("|"):
        return False
    if LIST_PREFIX_RE.match(line):
        return bool(INSTRUCTION_RE.search(line))
    return bool(STRONG_POLICY_RE.search(line)) and len(line) <= 220


def attach_duplicate_counts(instructions: list[Instruction]) -> list[Instruction]:
    keys = [canonical_instruction_key(item.text) for item in instructions]
    counts = Counter(keys)
    updated: list[Instruction] = []
    for item, key in zip(instructions, keys):
        updated.append(
            Instruction(
                file=item.file,
                line=item.line,
                text=item.text,
                rationale=item.rationale,
                scope=item.scope,
                retirement=item.retirement,
                duplicate_count=max(0, counts[key] - 1),
            )
        )
    return updated


def canonical_instruction_key(text: str) -> str:
    normalized = re.sub(r"[`*_#>\[\]():/.,、。・「」『』（）]", " ", text.lower())
    normalized = re.sub(r"\b(must|always|never|required|do|not|use|read|write|check)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def build_instruction_debt_report(
    *,
    root: Path = REPO_ROOT,
    scan_paths: Iterable[str] = DEFAULT_SCAN_PATHS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    files = discover_files(root, scan_paths)
    instructions = attach_duplicate_counts(
        [item for path in files for item in extract_instructions(path, root)]
    )
    debt_items = [item for item in instructions if item.issue_codes]
    debt_items.sort(key=lambda item: (severity_rank(item.severity), item.duplicate_count, item.file, -item.line), reverse=True)

    by_file: dict[str, dict[str, Any]] = defaultdict(lambda: {"instructions": 0, "debt_items": 0})
    by_issue = Counter()
    by_severity = Counter()
    for item in instructions:
        by_file[item.file]["instructions"] += 1
    for item in debt_items:
        by_file[item.file]["debt_items"] += 1
        by_severity[item.severity] += 1
        by_issue.update(item.issue_codes)

    return {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "read_only_instruction_debt_audit",
        "guardrail": "no prompt edit, no memory promotion, no skill edit",
        "scan_paths": list(scan_paths),
        "summary": {
            "files_scanned": len(files),
            "instructions_found": len(instructions),
            "debt_items": len(debt_items),
            "by_severity": dict(sorted(by_severity.items())),
            "by_issue": dict(sorted(by_issue.items())),
        },
        "files": dict(sorted(by_file.items())),
        "debt_items": [serialize_instruction(item) for item in debt_items[:80]],
        "recommendations": build_recommendations(debt_items),
    }


def severity_rank(severity: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(severity, 0)


def serialize_instruction(item: Instruction) -> dict[str, Any]:
    return {
        "file": item.file,
        "line": item.line,
        "text": item.text,
        "severity": item.severity,
        "issues": item.issue_codes,
        "has_rationale": item.rationale,
        "has_scope": item.scope,
        "has_retirement_condition": item.retirement,
        "duplicate_count": item.duplicate_count,
    }


def build_recommendations(debt_items: list[Instruction]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    missing_rationale = [item for item in debt_items if "missing_rationale" in item.issue_codes]
    missing_retirement = [item for item in debt_items if "missing_retirement_condition" in item.issue_codes]
    duplicates = [item for item in debt_items if "possible_duplicate" in item.issue_codes]
    if missing_rationale:
        recommendations.append(
            {
                "action": "add_rationale_comments",
                "reason": "instructions without reasons are hard to delete safely later",
                "count": len(missing_rationale),
            }
        )
    if missing_retirement:
        recommendations.append(
            {
                "action": "add_retirement_conditions",
                "reason": "temporary or context-specific rules need an explicit condition for removal",
                "count": len(missing_retirement),
            }
        )
    if duplicates:
        recommendations.append(
            {
                "action": "review_possible_duplicates",
                "reason": "similar instructions should be merged or routed to one source of truth",
                "count": len(duplicates),
            }
        )
    return recommendations


def build_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines: list[str] = []
    lines.append("# Instruction Debt Report")
    lines.append("")
    lines.append(f"- Generated at: `{report.get('generated_at')}`")
    lines.append(f"- Mode: `{report.get('mode')}`")
    lines.append(f"- Guardrail: {report.get('guardrail')}")
    lines.append(f"- Files scanned: {summary.get('files_scanned', 0)}")
    lines.append(f"- Instructions found: {summary.get('instructions_found', 0)}")
    lines.append(f"- Debt items: {summary.get('debt_items', 0)}")
    lines.append("")
    lines.append("## Issue Summary")
    for key, count in (summary.get("by_issue") or {}).items():
        lines.append(f"- {key}: {count}")
    lines.append("")
    guidance_items = [item for item in report.get("debt_items") or [] if item.get("file") != "MEMORY.md"]
    memory_items = [item for item in report.get("debt_items") or [] if item.get("file") == "MEMORY.md"]
    lines.append("## Guidance Debt Items")
    for item in guidance_items:
        location = f"{item['file']}:{item['line']}"
        issues = ", ".join(item.get("issues") or [])
        lines.append(f"- `{location}` [{item['severity']}] {issues}: {item['text']}")
    if not guidance_items:
        lines.append("- No guidance instruction debt detected.")
    lines.append("")
    lines.append("## Memory Policy Candidates")
    for item in memory_items[:30]:
        location = f"{item['file']}:{item['line']}"
        issues = ", ".join(item.get("issues") or [])
        lines.append(f"- `{location}` [{item['severity']}] {issues}: {item['text']}")
    if not memory_items:
        lines.append("- No memory policy candidates detected.")
    lines.append("")
    lines.append("## Recommendations")
    for item in report.get("recommendations") or []:
        lines.append(f"- {item['action']}: {item['count']} ({item['reason']})")
    if not report.get("recommendations"):
        lines.append("- No action recommended.")
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(build_markdown(report), encoding="utf-8")


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--scan-path", action="append", dest="scan_paths", help="Relative file or directory to scan. Repeatable.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_instruction_debt_report(
        root=args.root.resolve(),
        scan_paths=args.scan_paths or DEFAULT_SCAN_PATHS,
    )
    write_outputs(report, output_json=args.output_json, output_md=args.output_md)
    summary = report["summary"]
    print(
        "instruction_debt "
        f"files={summary['files_scanned']} "
        f"instructions={summary['instructions_found']} "
        f"debt_items={summary['debt_items']}"
    )


if __name__ == "__main__":
    main()
