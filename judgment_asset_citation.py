"""紫苑レビュー本文の判断資産citationを抽出する共通ヘルパー.

紫苑レビューのプロンプト（frontend/src/lib/shionReview.ts::buildShionReviewPrompt）は
LLMに「判断資産出典: 正規 JA-<candidate.id先頭8文字>」という形式で出典を明記させる。
正規判断資産の candidate.id は "cr-" + canonical_judgment_rules.json の rule_id なので、
本文中の citation は "JA-cr-<rule_idの先頭5文字程度>" という短縮形になる。

read-only の抽出専用モジュール。プロンプト・スコアリング・RAGには一切接続しない。
"""

from __future__ import annotations

import re
from typing import Iterable

CITATION_PATTERN = re.compile(r"JA-cr-([0-9a-fA-F]{3,})")


def extract_cited_rule_id_prefixes(review_text: str) -> list[str]:
    """review_text 中の JA-cr-<prefix> citation から prefix 一覧を返す（順序維持・重複除去）."""
    seen: list[str] = []
    for match in CITATION_PATTERN.finditer(review_text or ""):
        prefix = match.group(1).lower()
        if prefix not in seen:
            seen.append(prefix)
    return seen


def classify_citations(review_text: str, active_rule_ids: Iterable[str]) -> dict[str, list[str]]:
    """citationを能動判断資産のrule_idへ解決し、resolved/ambiguous/unresolvedに分類する.

    - resolved: prefixが能動ルールに一意に一致した rule_id
    - ambiguous: 複数の能動ルールに一致し得た prefix（短縮citationの原因調査用）
    - unresolved: どの能動ルールにも一致しなかった prefix（失効・非公開ルールの可能性）
    """
    normalized_ids = [str(rule_id).strip().lower() for rule_id in active_rule_ids if rule_id]
    resolved: list[str] = []
    ambiguous: list[str] = []
    unresolved: list[str] = []
    for prefix in extract_cited_rule_id_prefixes(review_text):
        matches = [rule_id for rule_id in normalized_ids if rule_id.startswith(prefix)]
        if len(matches) == 1:
            if matches[0] not in resolved:
                resolved.append(matches[0])
        elif len(matches) > 1:
            ambiguous.append(prefix)
        else:
            unresolved.append(prefix)
    return {"resolved": resolved, "ambiguous": ambiguous, "unresolved": unresolved}


def resolve_rule_ids_from_citations(review_text: str, active_rule_ids: Iterable[str]) -> list[str]:
    """フィードバック記録用: 一意に解決できたrule_idだけを返す（曖昧一致は採用しない）."""
    return classify_citations(review_text, active_rule_ids)["resolved"]
