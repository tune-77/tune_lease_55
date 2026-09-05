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
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
CODEX_SESSIONS = Path.home() / ".codex" / "sessions"
SKILL_ROOTS = (ROOT / ".claude" / "skills", ROOT / ".agents" / "skills")
REWORK_RE = re.compile(r"やり直|修正|違う|別(?:の)?skill|redo|retry|fix|instead", re.IGNORECASE)


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


def _timestamp(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _text_blocks(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") in {"text", "input_text"}
    )


def _invocation_labels(skill_name: str, user_text: str) -> tuple[str, str]:
    slash = re.search(rf"(?:^|\s)/{re.escape(skill_name)}(?:\s|$)", user_text, re.IGNORECASE)
    normalized_skill = re.sub(r"[-_\s]", "", skill_name).lower()
    normalized_text = re.sub(r"[-_\s]", "", user_text).lower()
    explicit = slash or normalized_skill in normalized_text
    return ("slash_command" if slash else "skill_tool", "explicit" if explicit else "auto")


def _mark_rework(records: list[dict[str, object]], pending: list[int], user_text: str) -> None:
    for index in pending:
        records[index]["user_rework_signal"] = bool(REWORK_RE.search(user_text))
    pending.clear()


def claude_skill_invocations(
    skill_names: list[str], start: datetime, end: datetime
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not CLAUDE_PROJECTS.exists():
        return records
    known = set(skill_names)
    for path in CLAUDE_PROJECTS.rglob("*.jsonl"):
        last_user_text = ""
        pending_rework: list[int] = []
        calls: dict[str, int] = {}
        try:
            lines = path.open(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        with lines:
            for line in lines:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = row.get("message") if isinstance(row.get("message"), dict) else {}
                content = message.get("content", [])
                human_text = "" if row.get("isMeta") or row.get("sourceToolUseID") else _text_blocks(content)
                if row.get("type") == "user" and human_text:
                    _mark_rework(records, pending_rework, human_text)
                    last_user_text = human_text
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use" and block.get("name") == "Skill":
                        skill = str((block.get("input") or {}).get("skill", ""))
                        invoked_at = _timestamp(row.get("timestamp"))
                        if skill not in known or invoked_at is None or not start <= invoked_at <= end:
                            continue
                        invocation_type, explicit_or_auto = _invocation_labels(skill, last_user_text)
                        calls[str(block.get("id", ""))] = len(records)
                        records.append(
                            {
                                "skill_name": skill,
                                "invoked_at": invoked_at.isoformat(timespec="seconds"),
                                "source": "claude",
                                "invocation_type": invocation_type,
                                "explicit_or_auto": explicit_or_auto,
                                "completed": False,
                                "user_rework_signal": False,
                            }
                        )
                    elif block.get("type") == "tool_result":
                        index = calls.get(str(block.get("tool_use_id", "")))
                        if index is not None:
                            records[index]["completed"] = not bool(block.get("is_error"))
                            pending_rework.append(index)
    return records


def codex_skill_invocations(
    skill_names: list[str], start: datetime, end: datetime
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not CODEX_SESSIONS.exists():
        return records
    patterns = {
        name: re.compile(rf"(?:^|[/\\]){re.escape(name)}[/\\]SKILL\.md")
        for name in skill_names
    }
    for path in CODEX_SESSIONS.rglob("*.jsonl"):
        last_user_text = ""
        pending_rework: list[int] = []
        calls: dict[str, list[int]] = {}
        try:
            lines = path.open(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        with lines:
            for line in lines:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
                if row.get("type") != "response_item":
                    continue
                if payload.get("type") == "message" and payload.get("role") == "user":
                    kinds = (payload.get("internal_chat_message_metadata_passthrough") or {}).get(
                        "content_item_kinds", []
                    )
                    human_text = _text_blocks(payload.get("content")) if not kinds or "user.text" in kinds else ""
                    if human_text:
                        _mark_rework(records, pending_rework, human_text)
                        last_user_text = human_text
                    continue
                if payload.get("type") in {"custom_tool_call", "function_call"}:
                    invoked_at = _timestamp(row.get("timestamp"))
                    if invoked_at is None or not start <= invoked_at <= end:
                        continue
                    raw = payload.get("input", payload.get("arguments", ""))
                    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                    for skill, pattern in patterns.items():
                        if not pattern.search(text):
                            continue
                        invocation_type, explicit_or_auto = _invocation_labels(skill, last_user_text)
                        index = len(records)
                        calls.setdefault(str(payload.get("call_id", "")), []).append(index)
                        records.append(
                            {
                                "skill_name": skill,
                                "invoked_at": invoked_at.isoformat(timespec="seconds"),
                                "source": "codex",
                                "invocation_type": invocation_type,
                                "explicit_or_auto": explicit_or_auto,
                                "completed": False,
                                "user_rework_signal": False,
                            }
                        )
                elif payload.get("type") in {"custom_tool_call_output", "function_call_output"}:
                    for index in calls.get(str(payload.get("call_id", "")), []):
                        records[index]["completed"] = True
                        pending_rework.append(index)
    return records


def codex_skill_file_reads(
    skill_names: list[str], start: datetime, end: datetime
) -> dict[str, int]:
    counts = Counter(record["skill_name"] for record in codex_skill_invocations(skill_names, start, end))
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
    invocations = sorted(
        claude_skill_invocations(skills, start, end) + codex_skill_invocations(skills, start, end),
        key=lambda item: str(item["invoked_at"]),
    )
    detailed_counts = Counter(record["skill_name"] for record in invocations)
    codex_reads = Counter(
        record["skill_name"] for record in invocations if record["source"] == "codex"
    )
    rows = []
    for name in skills:
        claude_delta = max(0, current.get(name, 0) - int(before.get(name, 0)))
        codex_proxy = codex_reads.get(name, 0)
        status = "active" if detailed_counts[name] else "aggregate_only" if claude_delta else "unobserved"
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
            *([f"- `{name}` ({status})" for name, _, _, status in rows if status != "active"] or ["- なし"]),
            "",
            "安全・事故防止Skillは利用ゼロだけで休眠させない。発火条件の狭小化を先に検討する。",
            "",
            "## Invocation audit",
            "",
            "| Skill | Invoked at | Source | Type | Explicit/auto | Completed | Rework |",
            "|---|---|---|---|---|---:|---:|",
            *(
                [
                    f"| `{item['skill_name']}` | `{item['invoked_at']}` | {item['source']} | "
                    f"{item['invocation_type']} | {item['explicit_or_auto']} | "
                    f"{str(item['completed']).lower()} | {str(item['user_rework_signal']).lower()} |"
                    for item in invocations
                ]
                or ["| - | - | - | - | - | - | - |"]
            ),
            "",
            "明示/自動と手戻りは直前・直後のユーザー入力から保守的に判定する。プロンプト本文は保存しない。",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "window": {"start": start.isoformat(), "end": end.isoformat()},
                "invocations": invocations,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


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
