"""改善案の表記ゆれを吸収する canonical_key 生成ユーティリティ."""

from __future__ import annotations

import hashlib
import re


_DOMAIN_RULES: list[tuple[str, list[str]]] = [
    ("answer_truncation", ["回答", "途切"]),
    ("answer_truncation", ["回答", "切れ"]),
    ("answer_truncation", ["途中", "切れ"]),
    ("industry_question_clarification", ["業界情報", "質問"]),
    ("out_of_scope_chat", ["審査外"]),
    ("repeat_query", ["同一クエリ"]),
    ("detail_request", ["詳細情報"]),
    ("web_search_detail", ["web検索"]),
    ("asset_industry_inference", ["物件名", "業種"]),
    ("business_industry_update", ["事業内容", "業種"]),
    ("cross_industry_entry", ["異業種参入"]),
    ("staffing_industry_split", ["職業紹介", "労働者派遣"]),
    ("reference_buttons", ["審査", "情報参照"]),
    ("reference_buttons", ["審査", "ナレッジ"]),
    ("home_customize", ["ホーム画面", "カスタマイズ"]),
    ("contract_guidance", ["契約条件", "ガイダンス"]),
    ("data_source_clarity", ["情報源", "明確"]),
    ("news_screening_usage", ["ニュース", "審査"]),
    ("ai_chat_db", ["ai chat", "db"]),
    ("lease_info_usage", ["リース情報", "活用"]),
    ("bulk_change_notification", ["大量修正", "通知"]),
    ("leaseback_learning", ["リースバック", "勉強会"]),
    ("edinet_integration", ["edinet"]),
    ("tdb_integration", ["帝国データバンク"]),
    ("ocr_mobile_input", ["ocr", "モバイル"]),
    ("portfolio_risk", ["ポートフォリオ"]),
    ("fairness_audit", ["公平性", "バイアス"]),
    ("kubernetes_migration", ["kubernetes"]),
]

_NOISE_PATTERNS = [
    r"<!--.*?-->",
    r"✅.*$",
    r"実装済.*$",
    r"改善済.*$",
    r"対応済.*$",
    r"完了.*$",
    r"phase\s*\d+",
    r"api",
    r"機能",
    r"追加",
    r"強化",
    r"改善",
    r"対応",
    r"自動",
    r"検討",
    r"再検討",
]


def normalize_title(title: str) -> str:
    """比較用にタイトルを正規化する."""
    text = title.lower()
    for pattern in _NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\s　・/（）()【】\[\]「」:：,，.。_-]+", "", text)
    return text.strip()


def canonical_key(title: str, description: str = "") -> str:
    """改善案の安定キーを返す。既知ドメインは人間可読キー、それ以外は短いhash。"""
    source = f"{title} {description}".lower()
    normalized_source = normalize_title(source)

    for key, keywords in _DOMAIN_RULES:
        if all(keyword.lower() in source for keyword in keywords):
            return key

    if not normalized_source:
        normalized_source = normalize_title(title)

    digest = hashlib.sha1(normalized_source.encode("utf-8")).hexdigest()[:12]
    return f"misc_{digest}"


if __name__ == "__main__":
    samples = [
        "回答途切れの改善",
        "AI回答が途中で切れる問題の修正",
        "EDINET連携（Phase2）",
        "EDINET API連携",
    ]
    for sample in samples:
        print(sample, "=>", canonical_key(sample))
