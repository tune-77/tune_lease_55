"""リース業界ドメイン辞書ローダー（P7-001）。

`static_data/lease_domain_glossary.json` を読み込み、シノニム引き・語彙集合を提供する。
検索（RAG）専用であり、スコアリングからは参照しないこと（BR-701）。
ファイル不在・破損時は空辞書で継続し、例外を上位に投げない（BR-703）。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import unicodedata

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GLOSSARY_PATH = os.path.join(_REPO_ROOT, "static_data", "lease_domain_glossary.json")

_EMPTY_GLOSSARY: dict = {"version": 0, "updated": "", "synonym_groups": [], "industry_terms": []}

# industry_terms の term↔aliases 相互引きに使う weight（表記ゆれは同義とみなすが控えめに）
_ALIAS_WEIGHT = 0.9

_lock = threading.Lock()
# path -> {"mtime": float, "glossary": dict, "index": dict, "terms": frozenset}
_cache: dict[str, dict] = {}


def _normalize(term: str) -> str:
    return unicodedata.normalize("NFKC", str(term or "")).strip().lower()


def _load_raw(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("[DomainGlossary] load failed (%s): %s", path, exc)
        return dict(_EMPTY_GLOSSARY)
    if not isinstance(raw, dict):
        logger.warning("[DomainGlossary] unexpected schema (not a dict): %s", path)
        return dict(_EMPTY_GLOSSARY)

    groups: list[dict] = []
    for group in raw.get("synonym_groups") or []:
        if not isinstance(group, dict):
            continue
        canonical = str(group.get("canonical") or "").strip()
        synonyms = [str(s).strip() for s in (group.get("synonyms") or []) if str(s).strip()]
        if not canonical or not synonyms:
            logger.warning("[DomainGlossary] skipping invalid group: %r", group.get("canonical"))
            continue
        try:
            weight = float(group.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        clamped = max(0.0, min(1.0, weight))
        if clamped != weight:
            logger.warning("[DomainGlossary] weight clamped for %s: %s -> %s", canonical, weight, clamped)
        groups.append({
            "canonical": canonical,
            "synonyms": synonyms,
            "weight": clamped,
            "source": str(group.get("source") or "業務知識(要確認)"),
        })

    terms: list[dict] = []
    for item in raw.get("industry_terms") or []:
        if not isinstance(item, dict) or not str(item.get("term") or "").strip():
            continue
        terms.append({
            "term": str(item.get("term")).strip(),
            "meaning": str(item.get("meaning") or ""),
            "context": str(item.get("context") or ""),
            "aliases": [str(a).strip() for a in (item.get("aliases") or []) if str(a).strip()],
        })

    return {
        "version": int(raw.get("version") or 0),
        "updated": str(raw.get("updated") or ""),
        "synonym_groups": groups,
        "industry_terms": terms,
    }


def _build_entry(path: str) -> dict:
    glossary = _load_raw(path)
    index: dict[str, list[tuple[str, float]]] = {}
    vocabulary: set[str] = set()
    for group in glossary["synonym_groups"]:
        members = [group["canonical"], *group["synonyms"]]
        weight = group["weight"]
        for member in members:
            vocabulary.add(member)
            key = _normalize(member)
            others = [(other, weight) for other in members if other != member]
            index.setdefault(key, []).extend(others)
    for item in glossary["industry_terms"]:
        vocabulary.add(item["term"])
        vocabulary.update(item["aliases"])
        members = [item["term"], *item["aliases"]]
        if len(members) > 1:
            for member in members:
                key = _normalize(member)
                others = [(other, _ALIAS_WEIGHT) for other in members if other != member]
                index.setdefault(key, []).extend(others)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return {
        "mtime": mtime,
        "glossary": glossary,
        "index": index,
        "terms": frozenset(vocabulary),
    }


def _entry(path: str) -> dict:
    with _lock:
        cached = _cache.get(path)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        if cached is not None and cached["mtime"] == mtime:
            return cached
        entry = _build_entry(path)
        _cache[path] = entry
        return entry


def get_glossary(path: str = GLOSSARY_PATH) -> dict:
    """検証済み辞書を返す。不在・破損時は空構造（synonym_groups/industry_terms が空リスト）。"""
    return _entry(path)["glossary"]


def synonyms_for(term: str, path: str = GLOSSARY_PATH) -> list[tuple[str, float]]:
    """term（canonical でも synonym でも可）と同グループの他の語を (語, weight) で返す。"""
    if not term:
        return []
    return list(_entry(path)["index"].get(_normalize(term), []))


def known_terms(path: str = GLOSSARY_PATH) -> frozenset[str]:
    """canonical + synonyms + industry_terms（term/aliases）の全語彙。"""
    return _entry(path)["terms"]
