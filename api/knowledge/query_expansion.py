"""RAG クエリ拡張（P7-002）。

ドメイン辞書（P7-001）のシノニムでクエリを展開する。元クエリを必ず先頭
（weight=1.0）に置き、拡張は decay × グループ weight で減衰させる（BR-711）。
辞書が読めない・該当語がない場合は元クエリのみを返す無害な no-op になる。
"""
from __future__ import annotations

import logging

from api.knowledge.domain_glossary import GLOSSARY_PATH, known_terms, synonyms_for

logger = logging.getLogger(__name__)

DEFAULT_MAX_VARIANTS = 4
DEFAULT_DECAY = 0.4


def expand_query(
    query: str,
    max_variants: int = DEFAULT_MAX_VARIANTS,
    decay: float = DEFAULT_DECAY,
    path: str = GLOSSARY_PATH,
) -> list[dict]:
    """クエリをシノニム展開する。

    Returns:
        [{"query": str, "weight": float, "replaced": str}, ...]
        先頭は必ず元クエリ（weight=1.0, replaced=""）。
    """
    original = {"query": query, "weight": 1.0, "replaced": ""}
    if not query or max_variants <= 0:
        return [original]

    try:
        # クエリに現れる辞書語を長い順に抽出（部分文字列の重複マッチを避ける）
        matched_terms = sorted(
            (term for term in known_terms(path=path) if term and term in query),
            key=len,
            reverse=True,
        )
    except Exception as exc:
        logger.warning("[QueryExpansion] glossary lookup failed: %s", exc)
        return [original]

    if not matched_terms:
        return [original]

    # より長いマッチ語に包含される語は除外する
    # （「赤字企業」がマッチしたクエリで「赤字」を置換すると壊れた文字列になるため）
    filtered_terms: list[str] = []
    for term in matched_terms:
        if not any(term in longer for longer in filtered_terms):
            filtered_terms.append(term)
    matched_terms = filtered_terms

    # 各語の置換候補を weight 降順で用意する
    candidates_per_term: list[tuple[str, list[tuple[str, float]]]] = []
    for term in matched_terms:
        alternatives = sorted(synonyms_for(term, path=path), key=lambda item: -item[1])
        if alternatives:
            candidates_per_term.append((term, alternatives))

    # 複数語がマッチしたとき1グループが枠を独占しないよう、語をまたいでラウンドロビンで採用する
    variants: list[dict] = []
    seen: set[str] = {query}
    max_rank = max((len(alts) for _t, alts in candidates_per_term), default=0)
    for rank in range(max_rank):
        for term, alternatives in candidates_per_term:
            if len(variants) >= max_variants:
                break
            if rank >= len(alternatives):
                continue
            alternative, group_weight = alternatives[rank]
            expanded = query.replace(term, alternative)
            if expanded in seen:
                continue
            seen.add(expanded)
            variants.append({
                "query": expanded,
                "weight": round(max(0.0, min(1.0, decay * group_weight)), 4),
                "replaced": term,
            })
        if len(variants) >= max_variants:
            break

    return [original, *variants]
