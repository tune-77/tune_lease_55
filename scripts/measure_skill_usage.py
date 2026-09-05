#!/usr/bin/env python3
"""Measure local Claude/Codex Skill usage without changing any Skill."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLAUDE_STATE = Path.home() / ".claude.json"
CODEX_SESSIONS = Path.home() / ".codex" / "sessions"
SKILL_ROOTS = (ROOT / ".claude" / "skills", ROOT / ".agents" / "skills")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def discover_local_skills() -> list[str]:
    names: set[str] = set()
    for skill_root in SKILL_ROOTS:
        if not skill_root.exists():
            continue
        for child in skill_root.iterdir():
            if child.is_dir() and (child / "SKILL.md").exists():
                names.add(child.name)
    return sorted(names)


def claude_counts(skill_names: list[str]) -> dict[str, int]:
    try:
        state = json.loads(CLAUDE_STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}
    usage = state.get("skillUsage") if isinstance(state, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    return {
        name: int((usage.get(name) or {}).get("usageCount") or 0)
        for name in skill_names
    }


def codex_skill_file_reads(
    skill_names: list[str], start: datetime, end: datetime
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if not CODEX_SESSIONS.exists():
        return {name: 0 for name in skill_names}
    patterns = {
        name: re.compile(rf"(?:^|[/\\]){re.escape(name)}[/\\]SKILL\.md")
        for name in skill_names
    }
    for path in CODEX_SESSIONS.rglob("*.jsonl"):
        try:
            lines = path.open(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        with lines:
            for line in lines:
                try:
                    row = json.loads(line)
                    timestamp = datetime.fromisoformat(str(row.get("timestamp", "")).replace("Z", "+00:00"))
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                if not start <= timestamp.astimezone() <= end:
                    continue
                payload = row.get("payload")
                if row.get("type") != "response_item" or not isinstance(payload, dict):
                    continue
                if payload.get("type") not in {"custom_tool_call", "function_call"}:
                    continue
                raw = payload.get("input", payload.get("arguments", ""))
                text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                for name, pattern in patterns.items():
                    if pattern.search(text):
                        counts[name] += 1
    return {name: counts[name] for name in skill_names}


def capture_baseline(path: Path) -> None:
    skills = discover_local_skills()
    payload = {
        "captured_at": now_iso(),
        "skills": skills,
        "claude_usage_counts": claude_counts(skills),
        "codex_metric": "tool-call inputs that read a matching SKILL.md; proxy only",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(baseline_path: Path, output_path: Path, end: datetime) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    start = datetime.fromisoformat(baseline["captured_at"])
    skills = sorted(set(baseline.get("skills", [])) | set(discover_local_skills()))
    before = baseline.get("claude_usage_counts", {})
    current = claude_counts(skills)
    codex_reads = codex_skill_file_reads(skills, start, end)
    rows = []
    for name in skills:
        claude_delta = max(0, current.get(name, 0) - int(before.get(name, 0)))
        codex_proxy = codex_reads.get(name, 0)
        status = "active" if claude_delta or codex_proxy else "unobserved"
        rows.append((name, claude_delta, codex_proxy, status))

    lines = [
        f"# Skill Usage — {end.date().isoformat()}",
        "",
        f"- window: `{start.isoformat(timespec='seconds')}` → `{end.isoformat(timespec='seconds')}`",
        "- Claude: `.claude.json` の `skillUsage.usageCount` 差分",
        "- Codex: tool call内で実際に読まれた `SKILL.md` の回数（代理指標）",
        "- `unobserved` は削除決定ではなく、人間レビュー対象",
        "",
        "| Skill | Claude delta | Codex read proxy | Status |",
        "|---|---:|---:|---|",
    ]
    lines.extend(f"| `{name}` | {claude} | {codex} | {status} |" for name, claude, codex, status in rows)
    lines.extend(
        [
            "",
            "## Review candidates",
            "",
            *([f"- `{name}`" for name, _, _, status in rows if status == "unobserved"] or ["- なし"]),
            "",
            "安全・事故防止Skillは利用ゼロだけで休眠させない。発火条件の狭小化を先に検討する。",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--capture-baseline", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.capture_baseline:
        capture_baseline(args.baseline)
        return
    if not args.output:
        parser.error("--output is required unless --capture-baseline is used")
    write_report(args.baseline, args.output, datetime.now().astimezone())


if __name__ == "__main__":
    main()
