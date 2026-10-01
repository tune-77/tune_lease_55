"""
Obsidian Vault の .md ファイルを再帰スキャンし、
H2見出し単位でチャンキングして返す。
wikiリンク（[[ノート名]]）をメタデータに含めてRAGのグラフ拡張に使う。
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Iterator

from runtime_paths import get_obsidian_vault_path

logger = logging.getLogger(__name__)

_VAULT_PATH = get_obsidian_vault_path()

# scan_vault が必ず対象に含めるべき Vault 直下サブディレクトリ（明示ドキュメント）
_REQUIRED_SUBDIRS: tuple[str, ...] = (
    "業界リスクニュース",
    "リースニュース",
    "リース知識",
    "Projects/tune_lease_55",
)

# ユーザーは読めるが、あらゆるAI検索経路（ChromaDB埋め込みを含む）から除外するディレクトリ。
# mobile_app/obsidian_bridge.py の _PRIVATE_NOTE_DIRS と揃える。
_PRIVATE_NOTE_DIRS: tuple[str, ...] = ("Private Reflection",)

# YAML frontmatter パターン
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# H2 見出し（## ）
_H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
# wikiリンク [[ノート名]] または [[ノート名|表示名]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
# 文として読める記号（句読点・括弧・表/見出し記号）。1つも無い1行の語の羅列は本文を持たない。
_SENTENCE_MARK_RE = re.compile(r"[。、，,．.:：;；!?！？「」『』()（）\[\]【】|*#`]")
_SPACE_RE = re.compile(r"\s+")


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# Vertex検索ワークフローのノート（api/vertex_distillation.py）で本文を持たない節。
_VERTEX_NOTE_SOURCE = "vertex_ai_search_workflow"
_VERTEX_METADATA_SECTIONS = frozenset({"Topic", "Query", "Mode"})


def normalize_chunk_text(text: str) -> str:
    """重複判定用の正規化。空白・箇条書き記号の揺れと数値だけの差
    （日次レポートの「win_pct(58.5%)」と「(58.7%)」など）を吸収する。"""
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text or "")
    return _SPACE_RE.sub(" ", _NUMBER_RE.sub("0", text)).strip()


def is_vertex_metadata_section(section: str, meta: dict) -> bool:
    """Vertex検索ノートの Topic/Query/Mode 節（検索語やモード名だけで本文が無い）。"""
    return (
        str((meta or {}).get("source") or "") == _VERTEX_NOTE_SOURCE
        and str(section or "").strip() in _VERTEX_METADATA_SECTIONS
    )


def is_keyword_stub(text: str) -> bool:
    """検索クエリ文字列のような「語を空白で並べただけ」の1行チャンクか。

    例: 「物件別の中古流動性と残価リスク 審査 根拠 確認論点 条件 リスク 出典 稟議 コメント」
    （Vertex Distilled ノートの ## Query 節）。語の一致だけで上位に来て本文を押し出す。
    """
    stripped = (text or "").strip()
    # 全角スペース区切りは人が書いた口語文（例:「違う　中古車だから　車検は2年」）なので対象外。
    if not stripped or "\n" in stripped or "　" in stripped or len(stripped) > 200:
        return False
    if _SENTENCE_MARK_RE.search(stripped):
        return False
    return len(stripped.split()) >= 4


def extract_wikilinks(text: str) -> list[str]:
    """[[リンク先]] を抽出してユニークなリストで返す。"""
    seen = set()
    result = []
    for m in _WIKILINK_RE.finditer(text):
        name = m.group(1).strip()
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


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
        """ChromaDB document ID（フルパス + セクションのハッシュ）。"""
        import hashlib
        raw = f"{self.file_path}#{self.section}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def obsidian_ref(self) -> str:
        """Obsidian 内部リンク表記 [[ファイル名#セクション]]。"""
        base = os.path.splitext(self.file_name)[0]
        return f"[[{base}#{self.section}]]"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """frontmatter を解析して (metadict, 本文) を返す。yaml.safe_load でパース。"""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        import yaml
        meta = yaml.safe_load(m.group(1)) or {}
        if not isinstance(meta, dict):
            meta = {}
    except Exception:
        meta = {}
    body = text[m.end():]
    return meta, body


def _chunk_by_h2(body: str, file_path: str, file_name: str, meta: dict, mtime: float) -> list[Chunk]:
    """H2見出し単位でチャンキングする。H2がない場合はファイル全体を1チャンク。"""
    positions = [(m.start(), m.group(1)) for m in _H2_RE.finditer(body)]

    # ファイル全体のwikilinksを抽出（ファイル単位で共有）
    file_wikilinks = extract_wikilinks(body)
    wikilinks_str = ",".join(file_wikilinks[:20])  # 最大20件

    if not positions:
        text = body.strip()
        if not text or is_keyword_stub(text):
            return []
        return [Chunk(
            file_path=file_path,
            file_name=file_name,
            section="概要",
            text=text,
            metadata={**meta, "wikilinks": wikilinks_str},
            mtime=mtime,
        )]

    chunks: list[Chunk] = []
    for i, (pos, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(body)
        text = body[pos:end].strip()
        # 見出し行自体は除いてテキストを取る
        text_without_heading = re.sub(r"^##\s+.+\n?", "", text, count=1).strip()
        if (
            not text_without_heading
            or is_keyword_stub(text_without_heading)
            or is_vertex_metadata_section(heading, meta)
        ):
            continue
        chunks.append(Chunk(
            file_path=file_path,
            file_name=file_name,
            section=heading.strip(),
            text=text_without_heading,
            metadata={**meta, "section": heading.strip(), "wikilinks": wikilinks_str},
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
            rel_parts = os.path.relpath(fpath, vault_path).split(os.sep)
            if any(part in _PRIVATE_NOTE_DIRS for part in rel_parts):
                continue
            try:
                mtime = os.path.getmtime(fpath)
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    raw = f.read()
                meta, body = _parse_frontmatter(raw)
                for chunk in _chunk_by_h2(body, fpath, fname, meta, mtime):
                    yield chunk
            except Exception as e:
                logger.debug("obsidian_loader: skip %s (%s)", fpath, e)
                continue
