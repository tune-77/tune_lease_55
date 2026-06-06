"""Query normalization helpers for Obsidian-backed AI features."""

from __future__ import annotations

import re
import unicodedata


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


__all__ = ["split_query_terms"]
