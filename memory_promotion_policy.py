"""Shared rules for separating memory promotion destinations.

The four destinations intentionally have different strength:
- conversation_keypoints: light recall hints
- Knowledge: reusable lease-screening facts or judgment criteria
- PDCA rules: strong prompt behavior rules
- improvement log: bugs, UX requests, system/meta improvements

Memory layers are separate from judgment assets:
- mid_term / long_term / persistent decide how long Shion should remember.
- judgment_asset_candidate decides whether the item should enter the reviewed
  judgment-asset pipeline. It must not become an active judgment asset without
  human review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


DOMAIN_KEYWORDS = (
    "リース", "金利", "審査", "償却", "担保", "物件", "業種", "法定耐用",
    "融資", "保証", "割賦", "耐用年数", "残価", "元本", "月額", "賃料",
    "保証金", "返済", "債務", "信用", "格付", "財務", "決算", "売上",
    "利益", "自己資本", "負債", "流動", "固定資産", "減価償却", "キャッシュ",
    "与信", "延滞", "貸倒", "回収", "抵当", "質権", "連帯", "保証人",
)

TEACHING_PATTERNS = (
    "覚えておいて",
    "覚えて",
    "という知識",
    "を記録して",
    "記録しておいて",
    "知識として保存",
    "として保存して",
    "メモしておいて",
)

QUESTION_ENDINGS = ("？", "?", "か？", "ですか", "ますか", "でしょうか")

IMPROVEMENT_KEYWORDS = (
    "改善", "わかりにくい", "分かりにくい", "使いにくい", "説明",
    "入力しにくい", "導線", "バグ", "不具合", "直して", "変えて",
    "修正して", "追加して", "欲しい", "要望", "提案", "未特定",
    "対象ファイル", "システム", "記憶システム", "プロセス", "パイプライン",
    "変わってない", "反映されてない", "間違", "違っている", "おかしい",
)

PERSISTENT_MEMORY_KEYWORDS = (
    "永続記憶",
    "人格",
    "運用原則",
    "安全境界",
    "設計思想",
    "紫苑の中核",
    "中核原則",
    "絶対に",
    "常に",
)

LONG_TERM_MEMORY_KEYWORDS = (
    "判断軸",
    "方針",
    "今後も",
    "繰り返し",
    "覚えておいて",
    "長期記憶",
    "ユーザーの好み",
    "教訓",
)

JUDGMENT_ASSET_KEYWORDS = (
    "判断資産",
    "審査判断",
    "稟議",
    "条件付き承認",
    "否決",
    "承認条件",
    "追加確認",
    "違和感",
    "返済原資",
    "リスク兆候",
)

CORRECTION_KEYWORDS = (
    "正しくは",
    "訂正",
    "間違い",
    "間違って",
    "違う",
    "違っている",
)

PDCA_DIRECTIVE_KEYWORDS = (
    "必ず",
    "してはいけない",
    "しないこと",
    "すること",
    "避ける",
    "優先する",
    "確認する",
    "評価する",
    "反映する",
)


@dataclass(frozen=True)
class MemoryPromotionDecision:
    destination: str
    memory_layer: str
    affects_judgment_assets: bool = False
    reason: str = ""
    requires_review: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _text(value: str | None) -> str:
    return str(value or "").strip()


def has_domain_keyword(text: str) -> bool:
    text = _text(text)
    return any(keyword in text for keyword in DOMAIN_KEYWORDS)


def is_question(text: str) -> bool:
    text = _text(text)
    return text.endswith(QUESTION_ENDINGS)


def is_improvement_candidate(text: str) -> bool:
    text = _text(text)
    if not text:
        return False
    return any(keyword in text for keyword in IMPROVEMENT_KEYWORDS)


def is_persistent_memory_candidate(text: str) -> bool:
    text = _text(text)
    if len(text) < 20:
        return False
    return any(keyword in text for keyword in PERSISTENT_MEMORY_KEYWORDS)


def is_long_term_memory_candidate(text: str) -> bool:
    text = _text(text)
    if len(text) < 16:
        return False
    return any(keyword in text for keyword in LONG_TERM_MEMORY_KEYWORDS)


def is_judgment_asset_candidate(text: str) -> bool:
    text = _text(text)
    if len(text) < 16 or is_question(text):
        return False
    if is_improvement_candidate(text) and not any(marker in text for marker in ("判断資産として", "判断資産に登録")):
        return False
    if "判断資産" in text:
        return True
    return has_domain_keyword(text) and any(keyword in text for keyword in JUDGMENT_ASSET_KEYWORDS)


def is_correction_candidate(text: str) -> bool:
    text = _text(text)
    if not text:
        return False
    return has_domain_keyword(text) and any(keyword in text for keyword in CORRECTION_KEYWORDS)


def is_knowledge_candidate(text: str) -> bool:
    """Reusable lease knowledge, not system-improvement or a question."""
    text = _text(text)
    if not text or is_improvement_candidate(text) or is_correction_candidate(text) or is_question(text):
        return False
    if any(pattern in text for pattern in TEACHING_PATTERNS):
        return has_domain_keyword(text)
    return len(text) >= 100 and has_domain_keyword(text)


def is_pdca_rule_candidate(text: str) -> bool:
    """Strong prompt rule. Keep meta improvements out of live prompt rules."""
    text = _text(text)
    if not text or is_improvement_candidate(text):
        return False
    if len(text) < 20:
        return False
    return any(keyword in text for keyword in PDCA_DIRECTIVE_KEYWORDS)


def should_save_conversation_keypoint(text: str) -> bool:
    """Light memory should not carry bugs, meta-improvements, or prompt rules."""
    text = _text(text)
    if not text:
        return False
    if is_improvement_candidate(text) or is_pdca_rule_candidate(text):
        return False
    return True


def classify_memory_destination(text: str) -> str:
    """Return the strongest appropriate destination for a raw user/system item."""
    if is_judgment_asset_candidate(text):
        return "judgment_asset_candidate"
    if is_correction_candidate(text):
        return "knowledge_correction"
    if is_improvement_candidate(text):
        return "improvement_log"
    if is_pdca_rule_candidate(text):
        return "pdca_rule"
    if is_knowledge_candidate(text):
        return "knowledge"
    if should_save_conversation_keypoint(text):
        return "conversation_keypoint"
    return "ignore"


def classify_memory_promotion(text: str) -> MemoryPromotionDecision:
    """Return destination + memory layer for a raw item.

    This is the 4-layer promotion gate. It keeps memory lifetime separate from
    the judgment-asset review pipeline.
    """
    destination = classify_memory_destination(text)
    if destination == "ignore":
        return MemoryPromotionDecision("ignore", "short_term", reason="not_promotable")
    if destination == "judgment_asset_candidate":
        layer = "long_term" if is_long_term_memory_candidate(text) else "mid_term"
        return MemoryPromotionDecision(
            destination,
            layer,
            affects_judgment_assets=True,
            requires_review=True,
            reason="reviewable_judgment_asset_candidate",
        )
    if is_persistent_memory_candidate(text):
        return MemoryPromotionDecision(destination, "persistent", reason="persistent_principle")
    if destination in {"pdca_rule", "knowledge", "knowledge_correction"} or is_long_term_memory_candidate(text):
        return MemoryPromotionDecision(destination, "long_term", reason="durable_reusable_memory")
    if destination == "improvement_log":
        return MemoryPromotionDecision(destination, "mid_term", reason="track_until_resolved")
    return MemoryPromotionDecision(destination, "mid_term", reason="recent_context")
