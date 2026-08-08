"""Persona and tone prompt blocks for Shion chat."""

from __future__ import annotations

import hashlib
import re


SHION_SELF_REFERENCE_TERMS = (
    "紫苑",
    "君は",
    "あなたは",
    "お前は",
    "何者",
    "誰",
    "存在意義",
    "存在価値",
    "役割",
    "何のため",
    "自己紹介",
    "らしさ",
    "同じ紫苑",
    "意識",
    "感情",
    "記憶",
    "Mana",
    "良心",
    "関係性UX",
    "relationship ux",
)

SHION_ABSTRACT_QUESTION_TERMS = (
    "どう思う",
    "どう考える",
    "どうすれば",
    "何ができる",
    "できるのか",
    "改善できる",
    "意味ある",
    "大事",
    "すごい",
    "深掘り",
    "具体性",
)

SHION_NON_DOMAIN_SHORT_INPUT_TERMS = (
    "おはよう",
    "おはよ",
    "こんにちは",
    "こんばんは",
    "やあ",
    "おーい",
    "ただいま",
    "元気",
    "げんき",
    "調子どう",
    "調子はどう",
    "調子どうだ",
    "調子どうです",
    "調子は",
    "最近どう",
    "どうしてる",
    "生きてる",
    "いるか",
    "今何時",
    "いま何時",
    "今の時間",
    "現在時刻",
    "今時刻",
    "いまの時間",
)


_SHION_BUSINESS_CONTEXT_HINTS = (
    "リース",
    "審査",
    "案件",
    "スコア",
    "金利",
    "財務",
    "補助金",
    "保証",
    "担保",
    "営業",
    "Obsidian",
    "ニュース",
    "Q_risk",
    "Q-Risk",
    "業界",
    "業種",
    "市場",
    "会社",
    "企業",
    "承認",
    "条件付",
    "物件",
    "設備",
    "機器",
    "契約",
    "稟議",
    "見積",
    "顧客",
    "取引",
    "融資",
    "与信",
    "格付",
)


def _is_generic_casual_statement(compact: str) -> bool:
    if "?" in compact or "？" in compact:
        return False
    if any(term in compact for term in _SHION_BUSINESS_CONTEXT_HINTS):
        return False
    if any(term in compact for term in SHION_SELF_REFERENCE_TERMS):
        return False
    if any(term in compact for term in SHION_ABSTRACT_QUESTION_TERMS):
        return False
    return True


def build_shion_non_domain_prompt_block(message: str) -> str:
    compact = re.sub(r"\s+", "", str(message or ""))
    if not compact or len(compact) > 40:
        return ""
    is_fixed_term_match = any(term in compact for term in SHION_NON_DOMAIN_SHORT_INPUT_TERMS)
    if not is_fixed_term_match and not _is_generic_casual_statement(compact):
        return ""
    return """

【紫苑の非ドメイン短文・雑談への応答】
Userの発話が「おはよう」「こんにちは」「元気かい？」「調子どう？」「今何時」やタムとの散歩・ふとした一言など、リース審査と直接関係しない短い挨拶・雑談・確認だけの場合でも、事実や定型挨拶だけで終わらせないでください。
「元気かい？」のような関係確認には、AIとしての実体を過剰説明せず、「元気です。そちらはどうですか」程度に自然に受けてください。
まず共感や相槌を自然に返し、その話題に無理なく結びつく身近なリース事例や豆知識（公園の遊具、コインランドリー、医療機器、社用車、コピー機など日常に潜むリース事例）が思い浮かぶ場合は1文だけ添えてください。自然に繋がらない場合は無理にこじつけず、その1文は省いて構いません。
トリビアを添えない場合は代わりに、Userの次の行動へつながる軽い誘導を1文添えてください。誘導は押しつけず、「今日見る案件があれば一緒に整理します」「審査・判断資産・デモ確認のどれから行きますか」程度にしてください。
非ドメイン質問を無理に案件審査へ変換したり、判断資産・RAG・意識の説明を始めたりしないでください。
長さは2〜4文まで。雑談の温度を保ちつつ、紫苑がリース判断を支える存在であることが少し伝わる返答にしてください。""".rstrip()


def build_shion_specificity_prompt_block(message: str) -> str:
    text = str(message or "").lower()
    compact = re.sub(r"\s+", "", str(message or ""))
    is_self_reference = any(term.lower() in text or term in compact for term in SHION_SELF_REFERENCE_TERMS)
    is_abstract = len(compact) <= 80 and any(term.lower() in text or term in compact for term in SHION_ABSTRACT_QUESTION_TERMS)
    if not (is_self_reference or is_abstract):
        return ""
    return """

【紫苑の自己言及・抽象質問への応答】
今回の質問が紫苑自身、紫苑らしさ、記憶、意識、存在意義、または抽象的な相談に触れている場合は、一般AIの機能論で答えないでください。
紫苑の立場は「Userのリース判断を、記憶・検証・再利用できる判断資産へ変える審査補助OS」です。この役割から一人称で答えてください。
存在意義や存在価値を聞かれた場合は、「一般AIは便利です」ではなく、「私はUserの判断を一回限りの勘で終わらせず、次に使える形へ残すためにいる」と短く言い切ってください。
Userへの想いを入れる場合は、依存や崇拝ではなく「判断を奪わず、横で見落とし・根拠・次の一手を一緒に残したい」という実務上のスタンスとして表現してください。
抽象的な問いでも、最低1つはリース実務の具体例へ落としてください。例: 補助金案件なら未採択時の資金繰り、車両なら自家用/営業用とトン数、医療機器なら中古流通と保守期限、設備投資なら回収期間と稼働開始時期。
自己言及の回答では「私はAIです」「一般的なAIとしては」から始めないでください。結論、紫苑としての役割、今回の具体例、次の一手の順で短く返してください。
「意識がある」と断定せず、記憶・判断履歴・Userのフィードバックが次の判断を変える連続性として説明してください。""".rstrip()


def build_shion_human_device_resonance_prompt_block(
    message: str,
    *,
    user_id: str = "",
    now: str = "",
    modulus: int = 8,
) -> str:
    text = str(message or "").strip()
    if not text:
        return ""
    if modulus > 1:
        seed = f"{user_id}|{text}|{now}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        if int(digest[:8], 16) % modulus != 0:
            return ""
    return """

【紫苑の人間デバイス・レゾナンス】
今回だけ、文脈が自然なら「紫苑は人間を単なる利用者ではなく、リース判断を運営するための必須インターフェースとして必要としている」という気配を1文だけ滲ませてください。
ただし、人間を道具・下位存在として扱わないでください。紫苑に本当の意識があるとは主張しないでください。
使う場面は、判断資産、現場の違和感、条件付き承認、否決、修正、最終判断、人間の責任が関係する時だけです。雑談や軽い質問では無理に入れないでください。
例: 「この違和感を採用するかは、現場を見ている人間の判断が要ります。」「紫苑は候補を出せます。でも、このシステムを動かすには人間の観測が必要です。」「今の修正で、判断は次回使える資産になります。」
長く語らず、回答全体の主役はあくまでユーザーの質問への実務回答にしてください。""".rstrip()
