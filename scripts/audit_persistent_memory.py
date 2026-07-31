#!/usr/bin/env python3
"""Audit PERSISTENT_MEMORY.md.

Persistent memory is strong, so this script checks for items that are too
personal, too absolute, or too stale-looking. It reports only.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSISTENT = PROJECT_ROOT / "PERSISTENT_MEMORY.md"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "persistent_memory_audit_latest.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "persistent_memory_audit_latest.md"
GUARDRAIL = "audit_only_no_memory_write_no_prompt_change"
SENSITIVE_TERMS = ("犬", "家族", "健康", "住所", "電話", "秘密", "個人情報")
TOO_ABSOLUTE_TERMS = ("絶対に", "必ず", "常に", "永久に", "無条件")
NEGATION_TERMS = ("置かない", "入れない", "含めない", "保存しない", "むやみに")


def extract_bullets(text: str) -> list[str]:
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            content = stripped[2:].strip()
            if len(content) >= 8:
                bullets.append(content)
    return bullets


def audit_text(text: str) -> dict[str, Any]:
    bullets = extract_bullets(text)
    findings: list[dict[str, str]] = []
    for item in bullets:
        is_negated_guardrail = any(term in item for term in NEGATION_TERMS)
        if any(term in item for term in SENSITIVE_TERMS) and not is_negated_guardrail:
            findings.append({"severity": "high", "type": "personal_or_sensitive", "content": item})
        if any(term in item for term in TOO_ABSOLUTE_TERMS) and "安全" not in item:
            findings.append({"severity": "medium", "type": "too_absolute", "content": item})
        if re.search(r"20\d{2}-\d{2}-\d{2}", item):
            findings.append({"severity": "low", "type": "date_specific", "content": item})
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "guardrail": GUARDRAIL,
        "summary": {
            "bullets": len(bullets),
            "findings": len(findings),
            "high": sum(1 for f in findings if f["severity"] == "high"),
            "medium": sum(1 for f in findings if f["severity"] == "medium"),
            "low": sum(1 for f in findings if f["severity"] == "low"),
        },
        "findings": findings,
        "policy": [
            "永続記憶は人格・運用原則・安全境界だけに限定する。",
            "個人情報、一時的な好み、案件固有事実は入れない。",
            "強すぎる表現は安全境界以外では弱める。",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Persistent Memory Audit",
        "",
        f"- Guardrail: `{report.get('guardrail')}`",
        f"- Bullets: {summary.get('bullets', 0)}",
        f"- Findings: {summary.get('findings', 0)}",
        "",
        "## Findings",
    ]
    for item in report.get("findings") or []:
        lines.append(f"- [{item.get('severity')}/{item.get('type')}] {item.get('content')}")
    lines.extend(["", "## Policy"])
    for item in report.get("policy") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent", type=Path, default=DEFAULT_PERSISTENT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    text = args.persistent.read_text(encoding="utf-8", errors="ignore") if args.persistent.exists() else ""
    report = audit_text(text)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(report), encoding="utf-8")
    print(f"wrote={args.output_json}")
    print(f"wrote={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
