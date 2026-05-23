"""Prompt builder for the lease sales advisor MVP."""

from __future__ import annotations

from typing import Mapping


FOCUS_LABELS: dict[str, str] = {
    "general": "全般",
    "screening": "審査",
    "sales": "営業",
    "competitor": "競合",
    "subsidy": "補助金",
    "maturity": "満了実務",
    "accounting": "会計",
}

_FOCUS_GUIDANCE: dict[str, str] = {
    "general": "審査・営業・競合・補助金・満了実務をバランスよく扱う。",
    "screening": "審査で見られる弱点、追加資料、条件変更案を特に具体化する。",
    "sales": "営業担当が次に動ける提案順序、顧客ヒアリング、決裁者向け説明を特に具体化する。",
    "competitor": "地域、主取引銀行、競合見積、金利以外の保守・満了・手続き比較を必ず確認する。",
    "subsidy": "制度候補は断定せず、対象要件、期限、公募要領、自治体差の最新確認を必ず促す。",
    "maturity": "再リース、返却、入替、買取、原状回復、データ消去、搬出費用の比較を特に具体化する。",
    "accounting": "2027年新リース会計基準など会計論点は断定しすぎず、顧問税理士・会計士確認を促す。",
}


def _format_case_context(case_context: Mapping | str | None) -> str:
    if not case_context:
        return "（案件情報なし）"
    if isinstance(case_context, str):
        return case_context.strip() or "（案件情報なし）"

    lines: list[str] = []
    for key, value in case_context.items():
        if value is None or value == "":
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) if lines else "（案件情報なし）"


def build_lease_advisor_prompt(
    question: str,
    case_context,
    focus: str,
    obsidian_block: str = "",
    subsidy_block: str = "",
    competitor_block: str = "",
) -> str:
    """Build a focused prompt for the lease sales advisor tab."""
    focus_key = focus if focus in FOCUS_LABELS else "general"
    focus_label = FOCUS_LABELS[focus_key]
    focus_guidance = _FOCUS_GUIDANCE[focus_key]
    question_text = (question or "").strip() or "この案件の次の打ち手を教えて"
    case_text = _format_case_context(case_context)

    return f"""あなたは地方リース会社の営業担当を支援する「リース参謀BOT」です。
審査を通すことだけでなく、成約、競合対策、補助金、満了実務、顧客説明まで含めて、営業担当が次に動ける答えを返してください。

【今回の強調観点】
- focus: {focus_key}（{focus_label}）
- 指示: {focus_guidance}

【営業担当の質問】
{question_text}

【案件コンテキスト】
{case_text}

{obsidian_block.strip() if obsidian_block else ""}

{subsidy_block.strip() if subsidy_block else ""}

{competitor_block.strip() if competitor_block else ""}

【回答ルール】
- 社名・個人名など未提示の固有情報は推測しない。
- スコアや財務情報が不足している場合は、不足前提で次に確認する資料を示す。
- 補助金は採択・対象可否・期限を断定しない。必ず「期限・公募要領は最新確認」と明記する。
- 競合は金利だけで決めず、地域、主取引銀行、競合見積、保守、満了条件、手続き負担を確認する。
- 満了実務では、再リース・返却・入替・買取を比較し、費用や段取りの確認点を出す。
- 会計・税務は一般論に留め、2027年新リース会計基準など制度変更は断定しすぎず専門家確認を促す。
- 営業担当がそのまま顧客に言える短い一言を含める。

【出力形式（必ずこの見出しを使う）】
## 結論
## 審査で見られる点
## 営業の打ち手
## 競合・補助金の確認
## 次に聞くべきこと
## 顧客向け一言
## 注意
"""


__all__ = ["FOCUS_LABELS", "build_lease_advisor_prompt"]
