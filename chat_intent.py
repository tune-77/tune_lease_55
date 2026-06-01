"""Chat request classification helpers shared by API and mobile chat flows."""

from __future__ import annotations

from dataclasses import dataclass


_DETAIL_HINTS = (
    "詳しく",
    "詳細",
    "根拠",
    "理由",
    "具体",
    "比較",
    "違い",
    "手順",
    "表で",
    "一覧",
    "例",
    "補足",
)

_INDUSTRY_HINTS = (
    "業界情報",
    "業界",
    "業種",
    "業種別",
    "業界別",
    "市場",
    "業界動向",
    "industry",
)

_LEASE_SCOPE_HINTS = (
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
)

_INDUSTRY_SPECIFIC_HINTS = (
    "製造",
    "建設",
    "医療",
    "福祉",
    "運輸",
    "物流",
    "IT",
    "情報通信",
    "卸売",
    "小売",
    "サービス",
    "不動産",
    "飲食",
    "宿泊",
    "農業",
    "金融",
    "保険",
    "教育",
    "宿泊",
    "電気",
    "ガス",
)

_TODAY_SCOPE_HINTS = (
    "案件",
    "ニュース",
    "改善",
    "成約",
    "失注",
    "履歴",
    "要約",
    "分析",
    "レポート",
    "指標",
    "予定",
    "スケジュール",
    "会議",
    "審査",
    "結果",
    "データ",
    "売上",
    "ダッシュボード",
)


def normalize_chat_text(text: str) -> str:
    """比較用に空白と記号を寄せた文字列へ正規化する."""
    value = str(text or "").lower().strip()
    for ch in ("\n", "\r", "\t", " ", "　", "・", ":", "：", "，", ",", ".", "。", "!", "！", "?", "？", "(", ")", "（", "）", "[", "]", "【", "】", "/", "\\", "-", "_", "／"):
        value = value.replace(ch, "")
    return value


def _recent_user_messages(history: list[dict[str, str]] | None, limit: int = 5) -> list[str]:
    if not history:
        return []
    recent: list[str] = []
    for item in reversed(history):
        if str(item.get("role") or "") != "user":
            continue
        text = normalize_chat_text(item.get("content") or "")
        if text:
            recent.append(text)
        if len(recent) >= limit:
            break
    return recent


def is_repeated_query(message: str, history: list[dict[str, str]] | None = None) -> bool:
    """直近のユーザー質問と同じ内容かをざっくり判定する."""
    current = normalize_chat_text(message)
    if not current:
        return False
    return current in _recent_user_messages(history)


def is_detail_request(message: str) -> bool:
    text = str(message or "")
    return any(hint in text for hint in _DETAIL_HINTS)


def is_industry_clarification_needed(message: str) -> bool:
    text = str(message or "")
    if not any(hint in text for hint in _INDUSTRY_HINTS):
        return False
    if any(hint in text for hint in _INDUSTRY_SPECIFIC_HINTS):
        return False
    if any(token in text for token in ("大分類", "小分類", "比較", "業種コード", "業界名", "業種名", "何業", "どの業種")):
        return False
    return True


def is_today_scope_clarification_needed(message: str) -> bool:
    text = str(message or "").strip()
    if "今日の" not in text and "today's" not in text.lower():
        return False
    if any(token in text for token in _TODAY_SCOPE_HINTS):
        return False
    if any(token in text for token in ("日付", "何日", "何曜日", "date", "today")):
        return False
    return True


def is_out_of_scope(message: str) -> bool:
    text = str(message or "")
    if any(hint in text for hint in _LEASE_SCOPE_HINTS):
        return False
    if any(hint in text for hint in _INDUSTRY_HINTS):
        return False
    # まったく文脈がない短文は話題外として扱う
    stripped = normalize_chat_text(text)
    return len(stripped) <= 8


@dataclass(frozen=True)
class ChatGuidance:
    repeated_query: bool = False
    detail_request: bool = False
    industry_clarification_needed: bool = False
    today_scope_clarification_needed: bool = False
    out_of_scope: bool = False

    @property
    def prompt_lines(self) -> tuple[str, ...]:
        lines: list[str] = []
        if self.today_scope_clarification_needed:
            lines.append(
                "『今日の』だけでは対象が曖昧なので、今日の何について知りたいかを1文で確認する。"
            )
        if self.industry_clarification_needed:
            lines.append(
                "業界情報の質問は、業種大分類/小分類、比較対象、見たい観点（成約率・金利・財務・ニュース）を先に確認する。"
            )
        if self.repeated_query:
            lines.append(
                "同じ質問が続いている場合は、前回の要点を短く再掲し、追加で必要な確認事項だけを示す。"
            )
        if self.detail_request:
            lines.append(
                "詳細要求があるので、結論→根拠→前提→注意点の順で、通常より少し厚めに答える。"
            )
        if self.out_of_scope:
            lines.append(
                "審査外の質問は、答えられる範囲を短く返し、必要ならリース審査の文脈に戻す。"
            )
        return tuple(lines)

    @property
    def prompt_suffix(self) -> str:
        lines = self.prompt_lines
        if not lines:
            return ""
        return "\n\n【会話ガイダンス】\n" + "\n".join(f"- {line}" for line in lines)


def build_chat_guidance(message: str, history: list[dict[str, str]] | None = None) -> ChatGuidance:
    """質問文と直近履歴から、応答の強め方を返す."""
    return ChatGuidance(
        repeated_query=is_repeated_query(message, history),
        detail_request=is_detail_request(message),
        industry_clarification_needed=is_industry_clarification_needed(message),
        today_scope_clarification_needed=is_today_scope_clarification_needed(message),
        out_of_scope=is_out_of_scope(message),
    )
