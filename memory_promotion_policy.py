"""Shared rules for separating memory promotion destinations.

The four destinations intentionally have different strength:
- conversation_keypoints: light recall hints
- Knowledge: reusable lease-screening facts or judgment criteria
- PDCA rules: strong prompt behavior rules
- improvement log: bugs, UX requests, system/meta improvements
"""

from __future__ import annotations


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
