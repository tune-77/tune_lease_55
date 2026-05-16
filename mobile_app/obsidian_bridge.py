"""Small Obsidian bridge for the mobile API chat.

The bridge stores only summarized notes selected by the AI chat layer. It does
not expose raw vault contents wholesale.
"""

from __future__ import annotations

import datetime as dt
import os
import re
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


def _to_wikilink(rel_path: str, alias: str | None = None) -> str:
    stem = str(rel_path or "").strip()
    if stem.lower().endswith(".md"):
        stem = stem[:-3]
    stem = stem.strip().replace("\\", "/")
    if not stem:
        return ""
    if alias:
        return f"[[{stem}|{alias}]]"
    return f"[[{stem}]]"


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


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("　", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_relevant_excerpt(snippet: str, terms: list[str], max_len: int = 240) -> str:
    text = (snippet or "").strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:max_len]
    for line in lines:
        low = line.lower()
        if any(term in low for term in terms):
            return line[:max_len]
    bullet_lines = [line for line in lines if line.startswith(("-", "・", "*", "1.", "2.", "3.", "4.", "5."))]
    if bullet_lines:
        return bullet_lines[0][:max_len]
    return lines[0][:max_len]


def build_obsidian_digest(query: str, hits: list[dict[str, str]], max_items: int = 4) -> dict[str, str]:
    terms = _expand_query_terms(query)[:8]
    if not terms:
        terms = [_normalize_text(query)]
    dedup_paths: list[str] = []
    dedup_excerpts: list[str] = []
    wikilinks: list[str] = []
    seen_paths: set[str] = set()
    seen_excerpts: set[str] = set()
    for hit in hits[:max_items]:
        path = str(hit.get("path", "")).strip()
        if not path or path in seen_paths:
            continue
        excerpt = _extract_relevant_excerpt(str(hit.get("snippet", "")), terms)
        if not excerpt:
            continue
        seen_paths.add(path)
        dedup_paths.append(path)
        wikilinks.append(_to_wikilink(path, Path(path).stem))
        if excerpt not in seen_excerpts:
            seen_excerpts.add(excerpt)
            dedup_excerpts.append(excerpt)

    if not dedup_excerpts:
        return {"digest": "", "title": "Obsidian統合要約", "source_count": "0"}

    lines = ["## Obsidian統合要約", ""]
    lines.append(f"- 関連ノート数: {len(dedup_paths)}")
    if dedup_paths:
        lines.append(f"- 対象: {', '.join(dedup_paths[:max_items])}")
    lines.append("")
    lines.append("### 共通して見える要点")
    for excerpt in dedup_excerpts[:max_items]:
        lines.append(f"- {excerpt}")
    if wikilinks:
        lines.append("")
        lines.append("### 関連ノート")
        for link in wikilinks[:max_items]:
            lines.append(f"- {link}")
    return {
        "digest": "\n".join(lines).strip(),
        "title": "Obsidian統合要約",
        "source_count": str(len(dedup_paths)),
        "links": "\n".join(wikilinks[:max_items]),
    }


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


def append_web_note(title: str, body: str) -> dict[str, str]:
    vault = find_vault()
    if not vault:
        return {"status": "skipped", "reason": "Obsidian vault not found"}
    day = dt.date.today().isoformat()
    rel = f"Projects/tune_lease_55/AI Chat/Web Research/{day}.md"
    path = _safe_note_path(vault, rel)
    now = dt.datetime.now().strftime("%H:%M")
    clean_title = (title or "Web参照メモ").strip()[:80]
    clean_body = (body or "").strip()
    if not clean_body:
        return {"status": "skipped", "reason": "empty note body"}
    section = f"## {now} {clean_title}\n\n{clean_body}\n"
    prefix = "\n" if path.exists() and path.read_text(encoding="utf-8", errors="ignore").strip() else ""
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + section)
    return {"status": "saved", "path": str(path)}


def append_wiki_note(title: str, body: str, related_paths: Iterable[str] | None = None) -> dict[str, str]:
    vault = find_vault()
    if not vault:
        return {"status": "skipped", "reason": "Obsidian vault not found"}
    rel = "Projects/tune_lease_55/tune_lease_55 Wiki.md"
    path = _safe_note_path(vault, rel)
    now = dt.datetime.now().strftime("%H:%M")
    clean_title = (title or "AI Wiki連携").strip()[:80]
    clean_body = (body or "").strip()
    related = []
    for item in related_paths or []:
        item = str(item or "").strip()
        if item:
            related.append(_to_wikilink(item, Path(item).stem))
    if not clean_body and not related:
        return {"status": "skipped", "reason": "empty note body"}
    section_lines = [f"## {now} {clean_title}", ""]
    if related:
        section_lines.append("### 関連ノート")
        for link in related[:8]:
            section_lines.append(f"- {link}")
        section_lines.append("")
    if clean_body:
        section_lines.append(clean_body)
    section = "\n".join(section_lines).rstrip() + "\n"
    prefix = "\n" if path.exists() and path.read_text(encoding="utf-8", errors="ignore").strip() else ""
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + section)
    return {"status": "saved", "path": str(path)}
