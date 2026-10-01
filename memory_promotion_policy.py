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

import re
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

# 「説明」「提案」「間違い」は審査ノウハウの文にも普通に出る（「説明が付かない資金使途は要注意」
# 「金利を提案する」）ため、改善要望の判定語から外している。訂正は CORRECTION_KEYWORDS が拾う。
IMPROVEMENT_KEYWORDS = (
    "改善", "わかりにくい", "分かりにくい", "使いにくい",
    "入力しにくい", "導線", "バグ", "不具合", "直して", "変えて",
    "修正して", "追加して", "欲しい", "要望", "未特定",
    "対象ファイル", "システム", "記憶システム", "プロセス", "パイプライン",
    "変わってない", "反映されてない", "おかしい",
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


# --- チャットで教えられた審査ノウハウの1段判定 ---------------------------------
# 以前は「トリガー語」と「行動語」の二段判定で、どちらかが欠けると捨てていた。さらに
# 「何」「とは」を含むだけで除外したため「ということは」でも落ちた。全1,371発言で
# 候補0件だったので、LLMに頼らず「ドメイン語を含む断定・ノウハウ文」を1段で拾う。
# 拾った候補は要確認（needs_review_quality）に回るだけで、自動昇格はしない。
TEACHING_DOMAIN_TERMS = (
    "リース", "審査", "判断資産", "稟議", "与信", "物件", "見積", "残価", "中古", "新車", "走行",
    "耐用年数", "補助金", "省力化", "銀行", "借入", "取引", "延滞", "増額", "保証", "担保", "決算",
    "財務", "売上", "利益", "返済", "資金繰り", "業種", "業界", "設備", "機械", "車", "トラック",
    "ナンバー", "ディーラー", "販売店", "サプライヤー", "検収", "契約", "満了", "再リース", "買取",
    "売却", "金利", "料率", "格付", "承認", "否決", "条件付き", "信用", "債務", "自己資本", "赤字",
    "黒字", "倒産", "法定", "税制", "会計", "保険",
)
# 保存を頼む言い方。「判断資産」という語だけでは質問や画面要望（「判断資産グラフを直して」）も
# 含むので、頼む形に限る。
TEACHING_EXPLICIT_MARKERS = (
    "判断資産にし", "判断資産に入れ", "判断資産に登録", "判断資産として", "判断資産へ",
    "覚えて", "覚えと", "記録して", "メモして", "登録して", "に入れて", "として残", "知識として",
)
_TEACHING_INTERROGATIVES = (
    "?", "？", "とはなん", "とは何", "って何", "ってなに", "は何", "はなに", "なんだろ", "どう思", "どう考",
    "どう見", "どうする", "どうすれ", "どうしたら", "どうなる", "どうや", "どうか", "いつ頃", "いつから",
    "いつまで", "どこで", "どこに", "どこが", "誰が", "誰に", "どれが", "どれくらい", "どのくらい", "どの程度",
    "ですか", "ますか", "でしょうか", "ないか",
)
# 「付かない」の「かな」、「いつも」の「いつ」のように部分一致で誤爆する語は文末だけで見る。
_TEACHING_REQUEST_WORDS = (
    "教えて", "調べて", "検索", "確認してくれ", "見せて", "まとめて", "説明して", "分析", "比較し",
    "答えて", "作って", "出して",
)
_TEACHING_SYSTEM_TERMS = (
    "画面", "ボタン", "表示", "バグ", "不具合", "エラー", "UI", "API", "コード", "実装", "機能", "アプリ",
    "ページ", "プロンプト", "Codex", "Claude", "Gemini", "PR", "デプロイ", "ログ", "パイプライン", "システム",
    "紫苑", "あなた", "AI", "記憶", "レポート", "Obsidian", "オブディシアン",
)
_TEACHING_KNOWHOW_MARKERS = (
    "なら", "たら", "場合", "とき", "時は", "要注意", "注意", "危な", "危険", "しない", "ない", "できない",
    "難しい", "やる", "やっちゃう", "取り扱", "取扱", "付き合", "増えて", "増える", "増加", "減る", "減少",
    "変わる", "かかる", "別途", "必要", "べき", "重要", "基準", "目安", "以内", "以上", "以下", "から",
    "ので", "ため", "傾向", "多い", "少ない", "高い", "低い", "使える", "対象", "確認する", "見る", "上が",
    "下が", "求め", "優先", "依存",
)
_TEACHING_PROMPT_NOISE = (
    "【審査分析画面からの紫苑レビュー依頼】",
    "【Vertex補助検索ヒント】",
    "この案件を、審査担当者の横にいる紫苑としてレビュー",
)
_TEACHING_REQUEST_END = re.compile(r"(て|てくれ|てください|下さい|ください|てね|てよ|ろ|せよ|答えて)[。！!」\s]*$")
_TEACHING_QUESTION_END = re.compile(r"(の|のか|かな|かしら|だろう|でしょ|いつ|どこ|誰|どれ)[。]?$")


def classify_lease_teaching(text: str | None) -> tuple[bool, str]:
    """チャット発言がリース審査のノウハウ教示かを1段で判定する。戻り値は (該当, 理由)。"""
    t = " ".join(_text(text).split())
    if len(t) < 12 or len(t) > 600:
        return False, "length"
    if any(marker in t for marker in _TEACHING_PROMPT_NOISE):
        return False, "prompt_noise"
    if not any(term in t for term in TEACHING_DOMAIN_TERMS):
        return False, "no_domain"
    if any(marker in t for marker in TEACHING_EXPLICIT_MARKERS) and not is_improvement_candidate(t):
        return True, "explicit"
    probe = t.replace("なぜなら", "")
    if any(word in probe for word in _TEACHING_INTERROGATIVES) or "なぜ" in probe or _TEACHING_QUESTION_END.search(t):
        return False, "question"
    if _TEACHING_REQUEST_END.search(t) or any(word in t for word in _TEACHING_REQUEST_WORDS):
        return False, "request"
    if any(term in t for term in _TEACHING_SYSTEM_TERMS):
        return False, "system_or_meta"
    if not any(marker in t for marker in _TEACHING_KNOWHOW_MARKERS):
        return False, "no_knowhow_marker"
    return True, "domain_statement"


def is_lease_teaching(text: str | None) -> bool:
    return classify_lease_teaching(text)[0]


def is_judgment_asset_candidate(text: str) -> bool:
    return is_lease_teaching(text)


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
