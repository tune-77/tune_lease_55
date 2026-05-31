"""改善案を意味単位で統合する軽量Agent."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from improvement_identity import canonical_key


_DOMAIN_PATTERNS: list[tuple[str, list[str]]] = [
    ("edinet", ["edinet"]),
    ("tdb", ["帝国データバンク", "tdb"]),
    ("ocr_mobile", ["ocr", "モバイル入力"]),
    ("portfolio_risk", ["ポートフォリオ"]),
    ("fairness_audit", ["公平性", "バイアス"]),
    ("industry_question", ["業界情報", "質問", "具体化"]),
    ("contract_guidance", ["契約条件", "ガイダンス"]),
    ("data_source", ["情報源", "source"]),
    ("out_of_scope_chat", ["審査外", "リース審査外"]),
    ("repeat_query", ["同一クエリ", "繰り返し"]),
    ("web_search_detail", ["web検索", "検索結果"]),
    ("detail_request", ["詳細情報", "詳細"]),
    ("asset_industry_inference", ["物件名", "業種", "推測"]),
    ("answer_truncation", ["回答途切れ", "途切れ"]),
    ("news_usage", ["ニュース", "審査"]),
    ("leaseback_learning", ["リースバック", "勉強会"]),
    ("ai_chat_db", ["ai chat", "db連携"]),
    ("lease_info_usage", ["リース情報", "活用"]),
    ("bulk_change_notification", ["大量修正", "通知"]),
    ("business_industry_update", ["事業内容", "業種情報"]),
    ("cross_industry_entry", ["異業種参入"]),
    ("reference_buttons", ["情報参照ボタン", "ナレッジ", "faqボタン"]),
    ("home_customize", ["ホーム画面", "カスタマイズ"]),
    ("staffing_industry_split", ["職業紹介", "労働者派遣"]),
    ("kubernetes", ["kubernetes", "k8s"]),
]


def _item_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key, ""))
        for key in ("title", "description", "detail", "reason", "target_module")
        if item.get(key)
    ).lower()


def _normalize_title(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[\s　・/（）()【】\[\]「」:：,，.。_-]+", "", normalized)
    normalized = re.sub(r"phase\d+", "", normalized)
    normalized = normalized.replace("api連携", "連携")
    return normalized


def _domain_key(item: dict[str, Any]) -> str:
    title = str(item.get("title", ""))
    return canonical_key(title, str(item.get("description") or item.get("detail") or ""))


def _merge_group(group_key: str, items: list[dict[str, Any]], index: int) -> dict[str, Any]:
    canonical = items[0]
    original_ids = [str(item.get("id", "")) for item in items if item.get("id")]
    titles = [str(item.get("title", "")).strip() for item in items if item.get("title")]
    duplicate_count = max(0, len(items) - 1)

    return {
        "group_id": f"GRP-{index:03d}",
        "group_key": group_key,
        "canonical_key": group_key,
        "canonical_id": canonical.get("id"),
        "merged_title": titles[0] if titles else group_key,
        "original_ids": original_ids,
        "original_titles": titles,
        "duplicate_count": duplicate_count,
        "merge_reason": (
            "同一テーマとして統合"
            if duplicate_count
            else "単独改善案"
        ),
    }


def deduplicate_improvements(
    improvements: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    改善案を意味単位でまとめる。

    Returns:
        canonical_improvements: 各グループ代表の改善案。既存IDは代表IDを維持する。
        groups: レポート用の統合結果。
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in improvements:
        buckets[_domain_key(item)].append(item)

    canonical_improvements: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []

    for index, (group_key, items) in enumerate(buckets.items(), 1):
        group = _merge_group(group_key, items, index)
        groups.append(group)

        canonical = dict(items[0])
        canonical["group_id"] = group["group_id"]
        canonical["canonical_key"] = group["canonical_key"]
        canonical["duplicate_ids"] = [
            item_id for item_id in group["original_ids"] if item_id != canonical.get("id")
        ]
        canonical["duplicate_count"] = group["duplicate_count"]
        canonical["original_titles"] = group["original_titles"]
        if group["duplicate_count"]:
            canonical["description"] = (
                str(canonical.get("description") or canonical.get("detail") or "")
                + "\n\n統合元: "
                + " / ".join(group["original_titles"])
            ).strip()
        canonical_improvements.append(canonical)

    return canonical_improvements, groups


if __name__ == "__main__":
    sample = [
        {"id": "REV-001", "title": "EDINET連携（Phase2）"},
        {"id": "REV-003", "title": "EDINET API連携"},
        {"id": "REV-030", "title": "審査・分析画面に情報参照ボタン追加"},
        {"id": "REV-032", "title": "審査・分析画面へのナレッジ・FAQボタン追加"},
    ]
    _, grouped = deduplicate_improvements(sample)
    for group in grouped:
        print(group)
