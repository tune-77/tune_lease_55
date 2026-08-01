#!/usr/bin/env python3
"""Build a compact Shion-facing digest from Codex/Claude work logs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

# runtime_paths を import するためリポジトリルートを sys.path に載せる
import sys as _sys
from pathlib import Path as _Path

_RUNTIME_PATHS_ROOT = str(_Path(__file__).resolve().parents[1])
if _RUNTIME_PATHS_ROOT not in _sys.path:
    _sys.path.insert(0, _RUNTIME_PATHS_ROOT)

from runtime_paths import resolve_obsidian_vault  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_JSON = REPORTS_DIR / "agent_worklog_digest_latest.json"
DEFAULT_MD = REPORTS_DIR / "agent_worklog_digest_latest.md"
DEFAULT_VAULT = resolve_obsidian_vault()

WORKLOG_HEADING_RE = re.compile(r"^##\s+(?P<time>\d{2}:\d{2})\s+(?P<agent>Codex|Claude)\s+Work Log\s*$")
SECTION_RE = re.compile(r"^###\s+(?P<title>.+?)\s*$")
PUBLIC_SECTIONS = {"Summary", "Chat Summary", "Decisions", "Changes", "Verification", "Open Items"}
MAX_FIELD_CHARS = 420


def _clip(text: str, limit: int = MAX_FIELD_CHARS) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _daily_note_paths(vault: Path, days: int) -> list[Path]:
    today = dt.date.today()
    paths: list[Path] = []
    for offset in range(max(1, days)):
        day = today - dt.timedelta(days=offset)
        path = vault / "Daily" / f"{day.isoformat()}.md"
        if path.exists():
            paths.append(path)
    return paths


def _parse_bullets(lines: list[str]) -> list[str]:
    items: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line == "- none":
            continue
        line = re.sub(r"^[-*]\s+", "", line).strip()
        if line:
            items.append(_clip(line))
    return items


def parse_work_logs(note_path: Path) -> list[dict[str, Any]]:
    try:
        lines = note_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []

    logs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section = ""
    section_lines: list[str] = []

    def flush_section() -> None:
        nonlocal section_lines
        if current is None or not section:
            section_lines = []
            return
        if section in PUBLIC_SECTIONS:
            current["sections"][section] = _parse_bullets(section_lines)
        section_lines = []

    def flush_log() -> None:
        nonlocal current
        if current is None:
            return
        flush_section()
        logs.append(current)
        current = None

    for line in lines:
        heading = WORKLOG_HEADING_RE.match(line)
        if heading:
            flush_log()
            current = {
                "date": note_path.stem,
                "time": heading.group("time"),
                "agent": heading.group("agent"),
                "source_path": str(note_path),
                "sections": {},
            }
            section = ""
            section_lines = []
            continue

        if current is None:
            continue

        if line.startswith("## "):
            flush_log()
            section = ""
            section_lines = []
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            flush_section()
            section = section_match.group("title").strip()
            section_lines = []
            continue

        section_lines.append(line)

    flush_log()
    return logs


def _summarize_log(log: dict[str, Any]) -> dict[str, Any]:
    sections = log.get("sections") if isinstance(log.get("sections"), dict) else {}
    summary = list(sections.get("Summary") or [])[:2]
    chat = list(sections.get("Chat Summary") or [])[:2]
    decisions = list(sections.get("Decisions") or [])[:3]
    changes = list(sections.get("Changes") or [])[:3]
    verification = list(sections.get("Verification") or [])[:2]
    open_items = list(sections.get("Open Items") or [])[:2]
    return {
        "date": log.get("date"),
        "time": log.get("time"),
        "agent": log.get("agent"),
        "source_path": log.get("source_path"),
        "summary": summary,
        "chat_summary": chat,
        "decisions": decisions,
        "changes": changes,
        "verification": verification,
        "open_items": open_items,
        "shion_use": "Userの意図・判断変更・採用/保留理由を内政モードの補正情報として使う",
    }


def build_digest(vault: Path, days: int = 14, limit: int = 12) -> dict[str, Any]:
    paths = _daily_note_paths(vault, days)
    logs: list[dict[str, Any]] = []
    for path in paths:
        logs.extend(parse_work_logs(path))
    logs.sort(key=lambda item: (str(item.get("date") or ""), str(item.get("time") or "")), reverse=True)
    items = [_summarize_log(log) for log in logs[: max(0, limit)]]
    return {
        "label": "Codex/Claude 作業録ダイジェスト",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "vault": str(vault),
        "days": days,
        "count": len(items),
        "source_count": len(logs),
        "items": items,
        "policy": {
            "raw_chat_logs_excluded": True,
            "private_detail_excluded": True,
            "use_for": "紫苑の内政モードで、Userの意図・判断・制約・実装後の検証結果を理解する補助情報",
            "do_not_use_for": "顧客情報の推測、Private Reflection原文の引用、人間承認なしの判断資産昇格",
        },
    }


def write_outputs(digest: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(digest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {digest['label']}",
        "",
        f"- generated_at: {digest['generated_at']}",
        f"- source_count: {digest['source_count']}",
        f"- displayed: {digest['count']}",
        "",
        "## Shion Use Policy",
        f"- {digest['policy']['use_for']}",
        f"- 禁止: {digest['policy']['do_not_use_for']}",
        "",
        "## Items",
    ]
    for item in digest.get("items") or []:
        lines.extend([
            "",
            f"### {item.get('date')} {item.get('time')} {item.get('agent')}",
            f"- Summary: {' / '.join(item.get('summary') or []) or '-'}",
            f"- Chat Summary: {' / '.join(item.get('chat_summary') or []) or '-'}",
            f"- Decisions: {' / '.join(item.get('decisions') or []) or '-'}",
            f"- Changes: {' / '.join(item.get('changes') or []) or '-'}",
            f"- Verification: {' / '.join(item.get('verification') or []) or '-'}",
            f"- Open Items: {' / '.join(item.get('open_items') or []) or '-'}",
        ])
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    vault = args.vault.expanduser()
    digest = build_digest(vault, days=args.days, limit=args.limit)
    write_outputs(digest, args.json, args.md)
    print(f"agent_worklog_digest={digest['count']} source={digest['source_count']}")
    print(args.json)
    print(args.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
