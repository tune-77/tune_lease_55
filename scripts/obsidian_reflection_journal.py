#!/usr/bin/env python3
"""Turn obsidian_theme_radar's selected-theme angle into a Problem/Insight/
Solution outline about improving lease screening / this system, and write it
as a daily note in the vault.

Deliberately writes to a new ``System Improvement Reflection/`` folder at the
vault root, separate from ``Private Reflection`` (Shion's own dialogue
reflection log) — see the design discussion in this feature's planning:
mixing this material into Shion's internal reflection log would blur two
unrelated logs. This folder is meant as raw material a human can later feed
into the REV improvement pipeline (e.g. scripts/build_reflection_action_candidates.py-style
review) — it does not write to the improvement ledger itself.

Input is scripts/obsidian_theme_radar.py's latest report. If it found no
theme at all (empty vault) or Vertex text generation is unavailable, this
script writes nothing for the day and reports why.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_paths import resolve_obsidian_vault  # noqa: E402
from scripts._vertex_text_gen import generate_text  # noqa: E402

REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_THEME_RADAR_JSON = REPORTS_DIR / "obsidian_theme_radar_latest.json"
DEFAULT_OUTPUT_JSON = REPORTS_DIR / "obsidian_reflection_journal_latest.json"
JOURNAL_DIR_REL = Path("System Improvement Reflection")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_outline(text: str) -> dict[str, str]:
    fields = {}
    for key, pattern in (
        ("problem", r"PROBLEM\s*[:：]\s*(.+?)(?=\n[A-Z]+\s*[:：]|\Z)"),
        ("insight", r"INSIGHT\s*[:：]\s*(.+?)(?=\n[A-Z]+\s*[:：]|\Z)"),
        ("solution", r"SOLUTION\s*[:：]\s*(.+?)(?=\n[A-Z]+\s*[:：]|\Z)"),
    ):
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        fields[key] = match.group(1).strip() if match else ""
    return fields


def build_outline(theme: str, angle: str, buzz: dict[str, Any], *, generate_fn=generate_text) -> dict[str, Any]:
    summary = str(buzz.get("summary") or "")
    prompt = (
        "あなたはリース審査AIとこのシステム（tune_lease_55）の改善を考えるアシスタントです。"
        "以下の切り口を、「問題（困っていること）」「気づき（新しい見方）」"
        "「解決（具体的にどうすればいいか）」の3段構成の簡単なアウトラインにしてください。"
        "「解決」は、リース審査AIやこのシステムの改善につながる具体的なアクションにしてください。"
        "各項目は2〜3文で日本語で書いてください。\n\n"
        f"テーマ: {theme}\n"
        f"切り口: {angle}\n"
        f"世の中の反応の要約: {summary[:600] or '(なし)'}\n\n"
        "以下の形式**のみ**で回答してください（前置き不要）:\n"
        "PROBLEM: ...\n"
        "INSIGHT: ...\n"
        "SOLUTION: ..."
    )
    result = generate_fn(prompt)
    if not result.get("used"):
        return {"parsed": False, "status": result.get("status", "unavailable"), "problem": "", "insight": "", "solution": ""}
    outline = _parse_outline(result.get("text", ""))
    parsed = bool(outline["problem"] or outline["insight"] or outline["solution"])
    return {"parsed": parsed, "status": "ok" if parsed else "unparseable", **outline}


def render_note(*, today: dt.date, theme: str, angle: str, outline: dict[str, Any], buzz: dict[str, Any]) -> str:
    sources = buzz.get("sources") or []
    ref_lines = "\n".join(f"- [{s.get('title')}]({s.get('uri')})" for s in sources if s.get("uri")) or "- (参照なし)"
    return f"""---
date: {today.isoformat()}
source: obsidian_theme_radar
theme: "{theme}"
tags: ["system-improvement", "theme-radar"]
---
# {today.isoformat()} システム改善の内省

## テーマ
{theme}

## 切り口
{angle}

## 問題（困っていること）
{outline.get('problem') or '(生成なし)'}

## 気づき（新しい見方）
{outline.get('insight') or '(生成なし)'}

## 解決（具体的にどうすればいいか）
{outline.get('solution') or '(生成なし)'}

## 参考（世の中の反応）
{ref_lines}
"""


def build_journal_entry(
    vault: Path,
    theme_radar_report: dict[str, Any] | None,
    *,
    today: dt.date | None = None,
    generate_fn=generate_text,
    force: bool = False,
) -> dict[str, Any]:
    today = today or dt.date.today()
    journal_dir = vault / JOURNAL_DIR_REL
    note_path = journal_dir / f"{today.isoformat()}.md"

    if not theme_radar_report:
        return {"status": "skipped_no_theme_radar_report", "note_path": str(note_path), "written": False}

    theme = str(theme_radar_report.get("selected_theme") or "")
    angle_info = theme_radar_report.get("proposed_angle") or {}
    angle = str(angle_info.get("angle") or "")
    if not theme or not angle:
        return {"status": "skipped_no_selected_theme", "note_path": str(note_path), "written": False}

    if note_path.exists() and not force:
        return {"status": "skipped_already_exists", "note_path": str(note_path), "written": False}

    buzz = next(
        (b for b in theme_radar_report.get("buzz_checks") or [] if b.get("theme") == theme),
        {},
    )
    outline = build_outline(theme, angle, buzz, generate_fn=generate_fn)
    note_text = render_note(today=today, theme=theme, angle=angle, outline=outline, buzz=buzz)

    journal_dir.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_text, encoding="utf-8")

    return {
        "status": "written",
        "note_path": str(note_path),
        "written": True,
        "theme": theme,
        "angle": angle,
        "outline_status": outline.get("status"),
        "problem": outline.get("problem", ""),
        "insight": outline.get("insight", ""),
        "solution": outline.get("solution", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=None)
    parser.add_argument("--theme-radar-json", type=Path, default=DEFAULT_THEME_RADAR_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--force", action="store_true", help="Overwrite today's journal note if it already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print result without writing to the vault")
    args = parser.parse_args()

    vault = args.vault.expanduser() if args.vault else resolve_obsidian_vault()
    if not vault.exists():
        print(f"[ERROR] Vault not found: {vault}")
        return 1

    theme_radar_report = _read_json(args.theme_radar_json.expanduser())

    if args.dry_run:
        theme = str((theme_radar_report or {}).get("selected_theme") or "")
        angle = str(((theme_radar_report or {}).get("proposed_angle") or {}).get("angle") or "")
        print(json.dumps({"would_write_theme": theme, "would_write_angle": angle}, ensure_ascii=False, indent=2))
        return 0

    result = build_journal_entry(vault, theme_radar_report, force=args.force)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
