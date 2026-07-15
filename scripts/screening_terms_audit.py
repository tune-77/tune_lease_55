"""Read-only audit for screening terminology drift.

This script scans code-facing files for terms that easily get mixed up in the
screening UI: actual PD, high-risk financial-pattern similarity, Q_risk, and
score labels. It does not change app logic or data.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = PROJECT_ROOT / "reports" / "screening_terms_audit_latest.json"
DEFAULT_MD = PROJECT_ROOT / "reports" / "screening_terms_audit_latest.md"

DEFAULT_SCAN_TARGETS = [
    PROJECT_ROOT / "frontend" / "src",
    PROJECT_ROOT / "api",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "report_generator.py",
    PROJECT_ROOT / "scoring_core.py",
    PROJECT_ROOT / "shinsa_gunshi_logic.py",
]

SKIP_PARTS = {
    "__pycache__",
    ".git",
    ".next",
    "node_modules",
    "reports",
    "data",
}

TEXT_SUFFIXES = {
    ".py",
    ".tsx",
    ".ts",
    ".jsx",
    ".js",
    ".md",
    ".json",
    ".sh",
}

TERM_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "actual_pd": [
        re.compile(r"PD(?!CA|F)"),
        re.compile(r"result\.pd\b"),
        re.compile(r"pd_percent"),
        re.compile(r"default_prob"),
        re.compile(r"デフォルト確率"),
        re.compile(r"デフォルト率"),
    ],
    "high_risk_similarity": [
        re.compile(r"default_warnings"),
        re.compile(r"高リスク財務パターン"),
        re.compile(r"類似度"),
        re.compile(r"高リスク格付先"),
    ],
    "q_risk": [
        re.compile(r"Q[_-]?risk", re.IGNORECASE),
        re.compile(r"quantum_risk"),
        re.compile(r"量子干渉"),
        re.compile(r"量子リスク"),
    ],
    "score": [
        re.compile(r"\bscore\b"),
        re.compile(r"スコア"),
    ],
}

DANGER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"result\.pd\b"),
    re.compile(r"デフォルト率モデル"),
    re.compile(r"推定デフォルト率"),
    re.compile(r"平均PD"),
    re.compile(r"PD値"),
    re.compile(r"PD\s*\d+%"),
    re.compile(r"高PD"),
    re.compile(r"今のPD"),
    re.compile(r"PD×"),
    re.compile(r"PD\s*\*"),
    re.compile(r"倒産確率.*実質ゼロ"),
    re.compile(r"default_prob\s*=\s*0\.\d+"),
]

SAFE_HINTS = (
    "算出済みPD",
    "PDが算出",
    "PD算出時",
    "PDは算出",
    "PDがある場合",
    "pd_percent",
    "未算出",
    "実PDでは",
    "延滞/デフォルト率",
    "スコアからPD",
    "高リスク財務パターン",
    "類似度",
    "Q_risk",
    "quantum_risk",
    "減点ではなく",
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    category: str
    severity: str
    text: str
    reason: str


def _iter_files(targets: Iterable[Path]) -> Iterable[Path]:
    for target in targets:
        target = target.expanduser()
        if not target.exists():
            continue
        if target.is_file():
            if target.suffix in TEXT_SUFFIXES:
                yield target
            continue
        for path in target.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            if path.name == "screening_terms_audit.py":
                continue
            rel_parts = set(path.relative_to(PROJECT_ROOT).parts)
            if rel_parts & SKIP_PARTS:
                continue
            yield path


def _categories_for(line: str) -> list[str]:
    categories: list[str] = []
    for category, patterns in TERM_PATTERNS.items():
        if any(pattern.search(line) for pattern in patterns):
            categories.append(category)
    return categories


def _classify(line: str, categories: list[str]) -> tuple[str, str]:
    if any(pattern.search(line) for pattern in DANGER_PATTERNS):
        return "warn", "PD/デフォルト確率を断定・試算・旧フォールバックとして見せる恐れ"
    if any(hint in line for hint in SAFE_HINTS):
        return "ok", "算出済み/未算出/補助指標の区別が明示されている"
    if "actual_pd" in categories:
        return "review", "PD表記だが、算出済みか補助指標かの区別が読み取りにくい"
    return "ok", "監査対象語だが危険表現ではない"


def build_audit(targets: Iterable[Path] = DEFAULT_SCAN_TARGETS) -> dict:
    findings: list[Finding] = []
    scanned_files = 0

    for path in sorted(set(_iter_files(targets))):
        scanned_files += 1
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for lineno, line in enumerate(lines, start=1):
            categories = _categories_for(line)
            if not categories:
                continue
            severity, reason = _classify(line, categories)
            try:
                rel = str(path.relative_to(PROJECT_ROOT))
            except ValueError:
                rel = str(path)
            for category in categories:
                findings.append(
                    Finding(
                        path=rel,
                        line=lineno,
                        category=category,
                        severity=severity,
                        text=line.strip()[:240],
                        reason=reason,
                    )
                )

    counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
        category_counts[finding.category] = category_counts.get(finding.category, 0) + 1

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "warn" if counts.get("warn", 0) else "ok",
        "scanned_files": scanned_files,
        "counts": counts,
        "category_counts": category_counts,
        "guardrail": "read_only_terms_audit_no_scoring_or_db_change",
        "glossary": {
            "actual_pd": "pd_percent が明示される場合だけPDとして扱う。欠落・0は未算出扱い。",
            "high_risk_similarity": "default_warnings は高リスク格付先との財務類似度。実PDではない。",
            "q_risk": "Q_risk / quantum_risk は財務・入力整合性の論点分解センサー。自動減点ではない。",
            "score": "スコアは総合判断の入口。PDやQ_riskと同一視しない。",
        },
        "findings": [asdict(finding) for finding in findings],
    }


def render_markdown(report: dict, *, max_rows: int = 80) -> str:
    lines = [
        "# Screening Terms Audit",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- status: `{report['status']}`",
        f"- scanned_files: `{report['scanned_files']}`",
        f"- guardrail: `{report['guardrail']}`",
        "",
        "## Counts",
        "",
    ]
    for key in ("warn", "review", "ok"):
        lines.append(f"- {key}: `{report.get('counts', {}).get(key, 0)}`")
    lines.extend(["", "## Glossary", ""])
    for key, value in report.get("glossary", {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Warn / Review Findings", ""])
    rows = [
        item
        for item in report.get("findings", [])
        if item.get("severity") in {"warn", "review"}
    ][:max_rows]
    if not rows:
        lines.append("- none")
    for item in rows:
        lines.append(
            f"- `{item['severity']}` `{item['category']}` "
            f"`{item['path']}:{item['line']}` — {item['reason']}"
        )
        lines.append(f"  - `{item['text']}`")
    return "\n".join(lines) + "\n"


def write_outputs(report: dict, *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit screening terminology drift.")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_MD)
    parser.add_argument("--target", type=Path, action="append")
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    report = build_audit(args.target or DEFAULT_SCAN_TARGETS)
    write_outputs(report, output_json=args.json, output_md=args.report)
    if args.print:
        print(render_markdown(report))
    else:
        print(f"report={args.report}")
        print(f"status={report['status']}")
        print(f"warn={report.get('counts', {}).get('warn', 0)}")
        print(f"review={report.get('counts', {}).get('review', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
