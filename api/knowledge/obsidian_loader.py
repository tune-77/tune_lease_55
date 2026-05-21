"""
Obsidian Vault の .md ファイルを再帰スキャンし、
H2見出し単位でチャンキングして返す。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Iterator

_VAULT_PATH = (
    "/Users/kobayashiisaoryou/Documents/Obsidian Vault/Projects/tune_lease_55/"
)

# YAML frontmatter パターン
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# H2 見出し（## ）
_H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    """Obsidian ナレッジの最小単位。"""

    file_path: str
    file_name: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)
    mtime: float = 0.0

    @property
    def doc_id(self) -> str:
        """ChromaDB document ID（ファイル名 + セクションのハッシュ）。"""
        import hashlib
        raw = f"{self.file_name}#{self.section}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def obsidian_ref(self) -> str:
        """Obsidian 内部リンク表記 [[ファイル名#セクション]]。"""
        base = os.path.splitext(self.file_name)[0]
        return f"[[{base}#{self.section}]]"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """frontmatter を解析して (metadict, 本文) を返す。"""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"').strip("'")
    body = text[m.end():]
    return meta, body


def _chunk_by_h2(body: str, file_path: str, file_name: str, meta: dict, mtime: float) -> list[Chunk]:
    """H2見出し単位でチャンキングする。H2がない場合はファイル全体を1チャンク。"""
    positions = [(m.start(), m.group(1)) for m in _H2_RE.finditer(body)]

    if not positions:
        text = body.strip()
        if not text:
            return []
        return [Chunk(
            file_path=file_path,
            file_name=file_name,
            section="概要",
            text=text,
            metadata=meta,
            mtime=mtime,
        )]

    chunks: list[Chunk] = []
    for i, (pos, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(body)
        text = body[pos:end].strip()
        # 見出し行自体は除いてテキストを取る
        text_without_heading = re.sub(r"^##\s+.+\n?", "", text, count=1).strip()
        if not text_without_heading:
            continue
        chunks.append(Chunk(
            file_path=file_path,
            file_name=file_name,
            section=heading.strip(),
            text=text_without_heading,
            metadata={**meta, "section": heading.strip()},
            mtime=mtime,
        ))
    return chunks


def scan_vault(vault_path: str = _VAULT_PATH) -> Iterator[Chunk]:
    """
    Vault を再帰スキャンして Chunk を yield する。
    vault_path が存在しない場合は空イテレータを返す（静かに失敗）。
    """
    if not os.path.isdir(vault_path):
        return

    for root, _dirs, files in os.walk(vault_path):
        for fname in files:
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(root, fname)
            try:
                mtime = os.path.getmtime(fpath)
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    raw = f.read()
                meta, body = _parse_frontmatter(raw)
                for chunk in _chunk_by_h2(body, fpath, fname, meta, mtime):
                    yield chunk
            except Exception:
                continue
