"""Obsidian-backed humor guidance for AI screening comments.

The app treats Obsidian as an editable upstream rulebook. This module reads a
small set of Markdown notes and turns them into compact prompt add-ons.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from obsidian_query import split_query_terms


DEFAULT_VAULT = Path("/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault")
HUMOR_NOTES = {
    "tone": Path("Humor/口調ルール.md"),
    "ng": Path("Humor/NG表現.md"),
    "risk": Path("Humor/リスク別ユーモア.md"),
    "industry": Path("Humor/業種別ユーモア.md"),
    "asset": Path("Humor/物件別ユーモア.md"),
}


def _vault_path(vault: str | os.PathLike[str] | None = None) -> Path | None:
    if vault:
        path = Path(vault).expanduser()
        return path if path.exists() and path.is_dir() else None

    candidates: list[Path] = []
    env = os.environ.get("OBSIDIAN_VAULT")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(DEFAULT_VAULT)
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def _read_note(vault: Path, rel_path: Path) -> str:
    path = vault / rel_path
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return text.strip()


def _risk_label(score: float) -> str:
    if score >= 75:
        return "低リスク"
    if score >= 55:
        return "中リスク"
    if score >= 40:
        return "高リスク"
    return "極高リスク"


def _keywords_from_res(res: dict[str, Any]) -> set[str]:
    raw = " ".join(
        str(res.get(k, "") or "")
        for k in ("industry_sub", "industry_major", "asset_name", "lease_asset_name")
    )
    keywords = {w for w in re.split(r"[\s　/・,、()（）]+", raw) if len(w) >= 2}
    keywords.update(split_query_terms(raw))
    for fixed in ("建設", "製造", "医療", "福祉", "運輸", "物流", "小売", "サービス", "IT", "OA", "車両"):
        if fixed in raw:
            keywords.add(fixed)
    return keywords


def _section_for_heading(text: str, heading: str, max_chars: int = 420) -> str:
    """Return bullets under a Markdown heading. Falls back to matching lines."""
    body = _strip_frontmatter(text)
    lines = body.splitlines()
    start = None
    heading_pattern = re.compile(r"^#{2,3}\s+" + re.escape(heading) + r"\s*$")
    for i, line in enumerate(lines):
        if heading_pattern.match(line.strip()):
            start = i + 1
            break
    if start is None:
        return ""
    out: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if stripped:
            out.append(stripped)
        if len("\n".join(out)) >= max_chars:
            break
    return "\n".join(out)[:max_chars].strip()


def _matching_bullets(text: str, keywords: set[str], max_lines: int = 4) -> str:
    body = _strip_frontmatter(text)
    selected: list[str] = []
    active = False
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            active = any(k in stripped for k in keywords)
            continue
        if active and stripped.startswith("- "):
            selected.append(stripped)
        elif stripped.startswith("- ") and any(k in stripped for k in keywords):
            selected.append(stripped)
        if len(selected) >= max_lines:
            break
    return "\n".join(selected)


def _first_bullets(text: str, max_lines: int = 6) -> str:
    body = _strip_frontmatter(text)
    bullets = [line.strip() for line in body.splitlines() if line.strip().startswith("- ")]
    return "\n".join(bullets[:max_lines])


def build_humor_prompt_addon(
    res: dict[str, Any] | None,
    *,
    style: str = "standard",
    vault: str | os.PathLike[str] | None = None,
    max_chars: int = 1800,
) -> str:
    """Build compact prompt guidance from Obsidian humor notes.

    Empty string means the Vault or notes were unavailable; callers can ignore it.
    """
    res = res or {}
    vault_path = _vault_path(vault)
    if not vault_path:
        return ""

    notes = {name: _read_note(vault_path, rel) for name, rel in HUMOR_NOTES.items()}
    if not any(notes.values()):
        return ""

    score = float(res.get("hantei_score", res.get("score", 0)) or 0)
    risk = _risk_label(score)
    keywords = _keywords_from_res(res)
    keywords.add(risk)

    tone = _first_bullets(notes.get("tone", ""), max_lines=7)
    ng = _first_bullets(notes.get("ng", ""), max_lines=6)
    risk_examples = _section_for_heading(notes.get("risk", ""), risk)
    industry_examples = _matching_bullets(notes.get("industry", ""), keywords, max_lines=3)
    asset_examples = _matching_bullets(notes.get("asset", ""), keywords, max_lines=3)

    style_note = (
        "現在のスタイルは標準モード。軽い職業ユーモアに限定する。"
        if style == "standard"
        else "現在のスタイルはキャラクター寄りモード。ただし審査本文は真面目に保ち、雑談は最後の1文だけにする。"
    )

    blocks = [
        "【Obsidianユーモア編集室】",
        style_note,
        "目的: 審査品質を壊さず、最後に読みやすいユーモアを1文だけ添える。",
        f"推定リスク帯: {risk}",
        "守るルール:",
        tone,
        "避ける表現:",
        ng,
        "参考例（必要に応じて言い換える。丸写しは不要）:",
        risk_examples,
        industry_examples,
        asset_examples,
        "出力制約: 審査判断は真面目に。最後に「ひとこと:」として1文だけユーモアを追加する。",
    ]
    addon = "\n".join(b for b in blocks if b and b.strip())
    return addon[:max_chars].strip()


__all__ = ["build_humor_prompt_addon"]
