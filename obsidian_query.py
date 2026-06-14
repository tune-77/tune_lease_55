"""Query normalization helpers for Obsidian-backed AI features."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterator


QUERY_FILLER_PATTERNS = (
    "について",
    "に関して",
    "に関する",
    "を教えて",
    "教えて",
    "とは",
    "って何",
    "ってなに",
    "知りたい",
    "説明して",
    "まとめて",
    "ください",
    "確認したい",
    "したい",
    "すべき",
    "使える",
    "でも",
    "ですか",
    "ますか",
)
QUERY_CONNECTOR_PATTERN = re.compile(r"(?<=[ぁ-んァ-ン一-龥A-Za-z0-9])(?:と|や|から|まで)(?=[ぁ-んァ-ン一-龥A-Za-z0-9])")
QUERY_TOPIC_SUFFIX_PATTERN = re.compile(r"の(?=関係|違い|意味|見方|使い方|使い分け|条件|方法|理由|原因|対策|手順)")
QUERY_PARTICLE_PATTERN = re.compile(r"(?<=[ぁ-んァ-ン一-龥A-Za-z0-9])(?:の|を|に|へ|で|が|は)(?=[ァ-ン一-龥A-Za-z0-9])")


def split_query_terms(query: str) -> list[str]:
    """Split a Japanese chat-style question into stable Obsidian search terms."""
    text = unicodedata.normalize("NFKC", query or "").lower().replace("　", " ")
    text = re.sub(r"[‐‑‒–—―−]+", "-", text)
    for pattern in QUERY_FILLER_PATTERNS:
        text = text.replace(pattern, " ")
    text = QUERY_CONNECTOR_PATTERN.sub(" ", text)
    text = QUERY_TOPIC_SUFFIX_PATTERN.sub(" ", text)
    text = QUERY_PARTICLE_PATTERN.sub(" ", text)
    chunks = re.split(r"[\s,、。.!！?？:：;；/／\\|()\[\]{}「」『』【】<>＜＞]+", text)
    terms: list[str] = []
    for chunk in chunks:
        term = chunk.strip(" -_・")
        term = re.sub(r"(?:は|を|に|へ|で|が|の)$", "", term)
        if len(term) >= 2:
            terms.append(term)
    return terms


def iter_vault_md_files(
    vault: Path,
    folders: tuple[str, ...],
    excluded_parts: tuple[str, ...],
) -> Iterator[Path]:
    """指定フォルダ配下の .md ファイルをすべて yield する。

    excluded_parts に含まれるパス要素を持つファイルはスキップする。
    """
    for folder in folders:
        base = vault / folder
        if not base.is_dir():
            continue
        for path in base.rglob("*.md"):
            if any(part in excluded_parts for part in path.parts):
                continue
            yield path


__all__ = ["split_query_terms", "iter_vault_md_files"]
