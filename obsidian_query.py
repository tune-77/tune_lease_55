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


def list_vault_md_files(root: Path, *, max_scan: int | None = None) -> list[Path]:
    """root配下（サブフォルダ含む）の.mdファイル一覧を返す共通ヘルパー。

    Obsidian Vault全体やGCS同期先ディレクトリなど、AI検索用途に限らない
    「.mdの列挙・件数集計」はここを通す。api/main.py, system_misc.py,
    lease_intelligence_connection.py などに散らばっていた直接rglob実装を
    統一するために追加。
    """
    if not root.exists() or not root.is_dir():
        return []
    files: list[Path] = []
    try:
        for path in root.rglob("*.md"):
            if path.is_file():
                files.append(path)
                if max_scan is not None and len(files) >= max_scan:
                    break
    except Exception:
        pass
    return files


def count_vault_notes(root: Path, *, max_scan: int = 2000) -> int:
    """root配下の.md件数を返す（診断・ステータス表示用の共通ヘルパー）。"""
    return len(list_vault_md_files(root, max_scan=max_scan))


__all__ = [
    "split_query_terms",
    "iter_vault_md_files",
    "list_vault_md_files",
    "count_vault_notes",
]
