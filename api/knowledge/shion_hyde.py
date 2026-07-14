"""Shion-specific HyDE query generation for RAG experiments.

This module is intentionally deterministic and side-effect free. It does not
call Gemini, does not alter the live chat prompt, and does not write to the RAG
index. The first use is offline/debug comparison after the hackathon freeze.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Any


_FILLER_PATTERNS = (
    "なんか",
    "ちょっと",
    "どう",
    "どうかな",
    "どうすれば",
    "教えて",
    "ください",
    "不安",
    "気になる",
)

_INTENT_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...], str], ...] = (
    (
        "repayment_source",
        ("返済", "返済原資", "資金繰り", "キャッシュ", "利益", "赤字", "償還", "支払"),
        ("返済原資", "資金繰り", "営業キャッシュフロー", "償還力", "返済余力"),
        "返済原資と資金繰りの確認",
    ),
    (
        "competition_risk",
        ("競合", "相見積", "他社", "成約", "失注", "見積", "銀行紹介", "案件化"),
        ("競合・成約リスク", "相見積", "銀行紹介", "案件発生経路", "営業温度感"),
        "競合・成約リスクの確認",
    ),
    (
        "asset_liquidity",
        ("物件", "残価", "中古", "売却", "換金", "処分", "再販", "担保", "設備"),
        ("物件換金性", "中古市場", "残価", "再販可能性", "処分リスク"),
        "物件換金性と残価妥当性の確認",
    ),
    (
        "business_flow",
        ("商流", "取引", "使途", "用途", "導入", "設備投資", "本業", "実態"),
        ("商流", "資金使途", "設備導入目的", "本業関連性", "取引実態"),
        "商流と導入目的の確認",
    ),
    (
        "new_customer",
        ("新規", "初取引", "初めて", "情報不足", "決算書", "実績なし"),
        ("新規先", "情報不足", "取引履歴", "外部信用", "代表者確認"),
        "新規先・情報不足案件の確認",
    ),
    (
        "condition_signal",
        ("条件", "保証", "頭金", "前払", "短縮", "期間", "承認", "要審議"),
        ("条件付き承認", "保証", "頭金", "リース期間", "保全条件", "承認条件"),
        "条件付き承認に必要な保全条件の確認",
    ),
    (
        "q_risk",
        ("qrisk", "q-risk", "q_risk", "量子", "違和感", "矛盾", "整合", "異常"),
        ("Q_risk", "違和感", "入力整合性", "探索シグナル", "リスク起点分離"),
        "Q_riskの違和感とリスク起点の確認",
    ),
    (
        "statutory_useful_life",
        ("耐用年数", "法定耐用", "償却", "トラック", "コンテナ", "フォークリフト"),
        ("法定耐用年数", "リース期間設定", "償却資産", "物件分類", "基本QA"),
        "法定耐用年数とリース期間設定の確認",
    ),
)

_DEFAULT_TERMS = (
    "審査確認点",
    "返済原資",
    "競合・成約リスク",
    "物件換金性",
    "商流",
    "条件付き承認",
)

_GENERIC_STOP_TERMS = {"案件", "確認", "不安", "審査", "リース", "会社", "今回"}


@dataclass(frozen=True)
class ShionHydeQuery:
    """Generated virtual screening memo used as a RAG query."""

    original_query: str
    hyde_query: str
    intent_tags: list[str] = field(default_factory=list)
    search_terms: list[str] = field(default_factory=list)
    should_search: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "hyde_query": self.hyde_query,
            "intent_tags": self.intent_tags,
            "search_terms": self.search_terms,
            "should_search": self.should_search,
            "reason": self.reason,
        }


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").lower().replace("　", " ")
    normalized = re.sub(r"[‐‑‒–—―−]+", "-", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _tokenize(text: str) -> list[str]:
    chunks = re.split(r"[\s,、。.!！?？:：;；/／\\|()\[\]{}「」『』【】<>＜＞]+", text)
    tokens: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        token = chunk.strip(" -_・")
        for filler in _FILLER_PATTERNS:
            token = token.replace(filler, "")
        token = re.sub(r"(?:は|を|に|へ|で|が|の|です|ます)$", "", token)
        if len(token) < 2 or token in _GENERIC_STOP_TERMS or token in seen:
            continue
        tokens.append(token)
        seen.add(token)
    return tokens


def _unique(items: list[str] | tuple[str, ...], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        result.append(value)
        seen.add(value)
        if limit is not None and len(result) >= limit:
            break
    return result


def build_shion_hyde_query(
    message: str,
    *,
    industry: str = "",
    asset_name: str = "",
    score_band: str = "",
    known_risk_tags: list[str] | None = None,
    max_terms: int = 16,
) -> ShionHydeQuery:
    """Convert a natural chat question into a virtual screening memo query.

    The output is optimized for retrieval, not for display to users.
    """
    original = (message or "").strip()
    normalized = _normalize(original)
    tokens = _tokenize(normalized)
    risk_tags = _unique(known_risk_tags or [], limit=6)

    matched_tags: list[str] = []
    expanded_terms: list[str] = []
    confirmation_points: list[str] = []
    for tag, triggers, terms, point in _INTENT_RULES:
        haystack = " ".join([normalized, " ".join(risk_tags).lower()])
        if any(trigger.lower() in haystack for trigger in triggers):
            matched_tags.append(tag)
            expanded_terms.extend(terms)
            confirmation_points.append(point)

    if not matched_tags:
        matched_tags = ["general_screening"]
        expanded_terms.extend(_DEFAULT_TERMS)
        confirmation_points.extend([
            "返済原資・競合・物件換金性・商流のうち、今回の不足情報を確認",
        ])

    context_terms = _unique(
        [
            industry.strip(),
            asset_name.strip(),
            score_band.strip(),
            *risk_tags,
            *tokens,
            *expanded_terms,
        ],
        limit=max_terms,
    )

    should_search = bool(original and len(original) >= 3)
    if not should_search:
        return ShionHydeQuery(
            original_query=original,
            hyde_query="",
            intent_tags=[],
            search_terms=[],
            should_search=False,
            reason="empty_or_too_short",
        )

    memo_lines = [
        "紫苑の審査RAG検索用の仮想審査メモ。",
        "今回の相談に対して、過去の判断資産・Obsidian知識・類似確認点から参照すべき材料を探す。",
        f"確認テーマ: {'、'.join(_unique(confirmation_points, limit=4))}。",
        f"検索語: {'、'.join(context_terms)}。",
        "出力で使う場合は、一般論ではなく今回の案件で見るべき確認点を最大3つに絞る。",
    ]
    return ShionHydeQuery(
        original_query=original,
        hyde_query="\n".join(memo_lines),
        intent_tags=_unique(matched_tags),
        search_terms=context_terms,
        should_search=True,
        reason="deterministic_template",
    )


def build_combined_search_query(query: ShionHydeQuery) -> str:
    """Return a compact single string suitable for existing vector_store.search."""
    if not query.should_search:
        return query.original_query
    parts = [
        query.original_query,
        " ".join(query.intent_tags),
        " ".join(query.search_terms),
        query.hyde_query,
    ]
    return "\n".join(part for part in parts if part).strip()


__all__ = ["ShionHydeQuery", "build_combined_search_query", "build_shion_hyde_query"]
