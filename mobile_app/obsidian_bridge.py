"""Small Obsidian bridge for the mobile API chat.

The bridge stores only summarized notes selected by the AI chat layer. It does
not expose raw vault contents wholesale.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Iterable


def _home_candidates() -> list[Path]:
    home = Path.home()
    roots = [
        Path(os.getenv("OBSIDIAN_VAULT", "")).expanduser() if os.getenv("OBSIDIAN_VAULT") else None,
        home / "Documents" / "Obsidian Vault",
        home / "Documents",
        home / "Obsidian",
        home / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents",
        home / "Library" / "Mobile Documents" / "com~apple~CloudDocs",
    ]
    return [p for p in roots if p and p.exists()]


def find_vault() -> Path | None:
    vaults: list[Path] = []
    for root in _home_candidates():
        if (root / ".obsidian").exists():
            vaults.append(root.resolve())
            continue
        try:
            for marker in root.rglob(".obsidian"):
                if marker.is_dir():
                    vaults.append(marker.parent.resolve())
                    break
        except OSError:
            continue
    seen: set[str] = set()
    for vault in vaults:
        key = str(vault)
        if key not in seen:
            return vault
        seen.add(key)
    return None


def _safe_note_path(vault: Path, rel: str) -> Path:
    target = (vault / rel).resolve()
    if vault not in target.parents and target != vault:
        raise ValueError("refusing to write outside the Obsidian vault")
    if target.suffix.lower() != ".md":
        target = target.with_suffix(".md")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def search_notes(query: str, limit: int = 4, max_chars: int = 700) -> list[dict[str, str]]:
    vault = find_vault()
    if not vault:
        return []
    terms = _expand_query_terms(query)[:8]
    if not terms:
        return []

    hits: list[dict[str, str]] = []
    for path in vault.rglob("*.md"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        low = text.lower()
        name = path.name.lower()
        if not any(t in low or t in name for t in terms):
            continue
        first = min((low.find(t) for t in terms if t in low), default=0)
        start = max(0, first - 160)
        snippet = text[start:start + max_chars].strip()
        hits.append({
            "path": str(path.relative_to(vault)),
            "snippet": snippet,
        })
        if len(hits) >= limit:
            break
    return hits


def _expand_query_terms(query: str) -> list[str]:
    raw_terms = [t.lower() for t in query.replace("　", " ").split() if len(t.strip()) >= 2]
    if not raw_terms:
        raw_terms = [query.lower().strip()]
    expanded = list(raw_terms)
    joined = " ".join(raw_terms)
    if any(k in joined for k in ("条件", "承認", "承認条件", "条件付", "条件付き")):
        expanded.extend([
            "承認条件",
            "条件付き承認",
            "条件付承認",
            "再提出",
            "保証",
            "担保",
            "期間短縮",
            "前受",
            "稟議",
            "q_risk",
            "信用リスク",
        ])
    if any(k in joined for k in ("obsidian", "保存", "メモ", "案件")):
        expanded.extend(["Projects/tune_lease_55/AI Chat", "Daily"])
    seen: set[str] = set()
    result: list[str] = []
    for term in expanded:
        term = term.strip().lower()
        if len(term) < 2 or term in seen:
            continue
        result.append(term)
        seen.add(term)
    return result


def recent_notes(limit: int = 3, folders: Iterable[str] | None = None, max_chars: int = 700) -> list[dict[str, str]]:
    vault = find_vault()
    if not vault:
        return []
    folders = list(folders or ("Projects/tune_lease_55/AI Chat", "Daily"))
    candidates: list[Path] = []
    for folder in folders:
        base = vault / folder
        if base.exists():
            try:
                candidates.extend(base.rglob("*.md"))
            except OSError:
                continue
    if not candidates:
        try:
            candidates = sorted(vault.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            candidates = []
    seen: set[str] = set()
    hits: list[dict[str, str]] = []
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(vault))
        if rel in seen:
            continue
        seen.add(rel)
        snippet = text.strip()[:max_chars]
        hits.append({"path": rel, "snippet": snippet})
        if len(hits) >= limit:
            break
    return hits


def collect_obsidian_context(query: str, limit: int = 4) -> list[dict[str, str]]:
    hits = search_notes(query, limit=limit)
    if len(hits) < limit:
        recent = recent_notes(limit=limit - len(hits))
        seen = {item["path"] for item in hits}
        for item in recent:
            if item["path"] not in seen:
                hits.append(item)
    return hits


def append_chat_note(title: str, body: str) -> dict[str, str]:
    vault = find_vault()
    if not vault:
        return {"status": "skipped", "reason": "Obsidian vault not found"}
    day = dt.date.today().isoformat()
    rel = f"Projects/tune_lease_55/AI Chat/{day}.md"
    path = _safe_note_path(vault, rel)
    now = dt.datetime.now().strftime("%H:%M")
    clean_title = (title or "AIチャットメモ").strip()[:80]
    clean_body = (body or "").strip()
    if not clean_body:
        return {"status": "skipped", "reason": "empty note body"}
    section = f"## {now} {clean_title}\n\n{clean_body}\n"
    prefix = "\n" if path.exists() and path.read_text(encoding="utf-8", errors="ignore").strip() else ""
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + section)
    return {"status": "saved", "path": str(path)}


def append_improvement_note(title: str, body: str) -> dict[str, str]:
    vault = find_vault()
    if not vault:
        return {"status": "skipped", "reason": "Obsidian vault not found"}
    day = dt.date.today().isoformat()
    rel = f"Projects/tune_lease_55/AI Chat/Improvement Log/{day}.md"
    path = _safe_note_path(vault, rel)
    now = dt.datetime.now().strftime("%H:%M")
    clean_title = (title or "AI改善候補").strip()[:80]
    clean_body = (body or "").strip()
    if not clean_body:
        return {"status": "skipped", "reason": "empty note body"}
    section = f"## {now} {clean_title}\n\n{clean_body}\n"
    prefix = "\n" if path.exists() and path.read_text(encoding="utf-8", errors="ignore").strip() else ""
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + section)
    return {"status": "saved", "path": str(path)}
