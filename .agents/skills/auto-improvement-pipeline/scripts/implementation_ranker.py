"""改善案の実装可能性と優先順を判定するAgent."""

from __future__ import annotations

from typing import Any

from improvement_identity import canonical_key


_CATEGORY_ORDER = {
    "quick_ui": 10,
    "obsidian_chat": 20,
    "logic_light": 30,
    "data_quality": 40,
    "db_api": 50,
    "external": 60,
    "infra": 70,
    "planning": 80,
}


def _text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "description", "detail", "target_module"):
        value = str(item.get(key, ""))
        if not value:
            continue
        # Obsidian抽出由来の "理由：AI Chat 改善ログ..." は出所であり分類根拠ではない。
        value = value.split("理由：", 1)[0].split("理由:", 1)[0]
        parts.append(value)
    return " ".join(parts).lower()


def classify_implementation(item: dict[str, Any]) -> dict[str, Any]:
    text = _text(item)

    category = "planning"
    effort = 4
    risk = 3
    impact = 3
    reason = "仕様整理が必要"

    if any(kw in text for kw in ["kubernetes", "k8s", "インフラ"]):
        category, effort, risk, impact, reason = (
            "infra", 5, 4, 2, "インフラ変更でアプリ改善より後回し",
        )
    elif any(kw in text for kw in ["edinet", "帝国データバンク", "外部api", "外部 api"]):
        category, effort, risk, impact, reason = (
            "external", 5, 4, 4, "外部API・契約・認証情報に依存",
        )
    elif any(kw in text for kw in ["db連携", "データベース", "sqlite", "api連携"]):
        category, effort, risk, impact, reason = (
            "db_api", 4, 4, 4, "DB/API境界の設計とテストが必要",
        )
    elif any(kw in text for kw in ["回答途切れ", "同一クエリ", "審査外", "詳細情報", "業界情報", "ai chat"]):
        category, effort, risk, impact, reason = (
            "obsidian_chat", 2, 2, 5, "AI Chat/Obsidian文脈の改善で効果が出やすい",
        )
    elif any(kw in text for kw in ["情報参照ボタン", "faqボタン", "ナレッジ", "ホーム画面", "表示", "ボタン"]):
        category, effort, risk, impact, reason = (
            "quick_ui", 2, 2, 4, "UI導線中心で小さく実装できる",
        )
    elif any(kw in text for kw in ["業種", "異業種", "分類", "物件名", "契約条件", "ニュース"]):
        category, effort, risk, impact, reason = (
            "logic_light", 3, 3, 4, "警告・候補提示から段階導入できる",
        )
    elif any(kw in text for kw in ["情報源", "大量修正", "通知"]):
        category, effort, risk, impact, reason = (
            "data_quality", 3, 2, 3, "監査性・運用品質の改善",
        )
    elif any(kw in text for kw in ["ocr", "ポートフォリオ", "公平性", "バイアス"]):
        category, effort, risk, impact, reason = (
            "db_api", 4, 4, 5, "価値は高いが設計面の影響が大きい",
        )

    priority_score = (
        impact * 8
        - effort * 5
        - risk * 4
        - _CATEGORY_ORDER.get(category, 99) / 10
    )

    return {
        "category": category,
        "effort": effort,
        "risk": risk,
        "impact": impact,
        "priority_score": round(priority_score, 2),
        "rank_reason": reason,
    }


def rank_improvements(improvements: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked: list[dict[str, Any]] = []
    for item in improvements:
        enriched = dict(item)
        enriched["implementation"] = classify_implementation(item)
        ranked.append(enriched)

    ranked.sort(
        key=lambda item: (
            -item["implementation"]["priority_score"],
            _CATEGORY_ORDER.get(item["implementation"]["category"], 99),
            item.get("id", ""),
        )
    )

    recommended_order: list[dict[str, Any]] = []
    for order, item in enumerate(ranked, 1):
        impl = item["implementation"]
        item["recommended_order"] = order
        recommended_order.append({
            "order": order,
            "id": item.get("id"),
            "group_id": item.get("group_id"),
            "canonical_key": item.get("canonical_key")
            or canonical_key(str(item.get("title", "")), str(item.get("description", ""))),
            "title": item.get("title"),
            "category": impl["category"],
            "effort": impl["effort"],
            "risk": impl["risk"],
            "impact": impl["impact"],
            "priority_score": impl["priority_score"],
            "reason": impl["rank_reason"],
            "duplicate_ids": item.get("duplicate_ids", []),
        })

    return ranked, recommended_order


if __name__ == "__main__":
    sample = [
        {"id": "REV-020", "title": "回答途切れの改善"},
        {"id": "REV-008", "title": "Kubernetes移行検討"},
    ]
    _, order = rank_improvements(sample)
    for item in order:
        print(item)
