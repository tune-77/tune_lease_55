"""ユーザーが教えた判断資産・ノウハウを「方針」と「知見」に分け、方針を回答の結論として効かせる。

- 方針（policy）: 社内ルールとして言い切ったもの。「〜とは付き合わない」「取り扱わない」「〜が増えないようにする」
  「〜してはいけない」「必ず〜する」「当社の方針」など、行為を禁止・義務づける明示的な表現がある時だけ。
- 知見（insight）: 傾向・数値・手順・確認事項。迷うものはすべて知見（方針に誤分類して答えを固めない）。

決定的な規則だけで分類する（LLM を使わない）。保存時・判断資産の正本の生成時・記憶索引の生成時に付与し、
付与されていない古い資産は想起時にこの関数で分類する。

2026-10-03 の前後比較（experiments/chat_before_after/）で、数字・手順型の教示は答えにそのまま使われた一方、
方針型の「銀行と取引のない企業とは付き合わない」は「高リスクなので動機と資金繰りを確認」に弱められた。
"""

from __future__ import annotations

import re
from typing import Any

POLICY = "policy"
INSIGHT = "insight"

# 当社の行為を否定する言い切り（「陳腐化しない」等の物件の性質は対象外にするため、行為の動詞に限る）
_POLICY_PATTERNS = (
    re.compile(r"(付き合わ|取り?扱わ|取扱わ|貸さ|通さ|増やさ|出さ)ない|(取引|契約|承認|増額)しない|(受け付け|引き受け|応じ)ない"),
    re.compile(r"取り?扱いしない|取扱いしない"),
    re.compile(r"ないようにする|ないようにしている|しないこと|しないように"),
    re.compile(r"してはいけない|してはならない|べきではない|べきでない|禁止|厳禁|認めない|お断りする|お断りしている|断っている"),
    re.compile(r"必ず(?!しも)[^。]{0,24}(する|確認|取る|求める|徴求|もらう)"),  # 「必ずしも〜ない」は方針ではない
    re.compile(r"(当社|弊社|うち)の?(方針|ルール|決まり)"),
)


def classify_knowledge_kind(text: str) -> str:
    body = str(text or "")
    return POLICY if any(p.search(body) for p in _POLICY_PATTERNS) else INSIGHT


def knowledge_kind_of(item: dict[str, Any], text: str = "") -> str:
    """保存済みの knowledge_kind を優先し、無ければ本文から分類する。"""
    stored = str(item.get("knowledge_kind") or "").strip()
    if stored in {POLICY, INSIGHT}:
        return stored
    return classify_knowledge_kind(text or item.get("canonical_statement") or item.get("claim") or item.get("content") or "")


POLICY_HEADER = "【社内方針（ユーザーが定めたルール）】"
POLICY_INSTRUCTIONS = (
    "以下はユーザー（審査担当）が定めた社内方針。質問に関係する方針があれば、一般論で弱めたり上書きしたりせず、"
    "回答の冒頭で結論として述べる（例:「社内方針では、〜とは取引しません」）。例外や追加の確認は方針を示した後に"
    "補足として書く。方針と一般的な審査の考え方が食い違う時は方針を優先し、食い違いがあることを一言添える。"
    "方針を使った時は回答の末尾に「出典: 」として下の出典表記をそのまま添える。関係のない方針には触れない。"
)
INSIGHT_CITATION_INSTRUCTION = "判断資産・教わった知識を使った時は、回答の末尾に「出典: 」として下の出典表記を添える。"


def format_policy_block(policies: list[dict[str, str]]) -> str:
    """policies は {"text", "source"} の並び。空なら空文字。"""
    if not policies:
        return ""
    lines = [POLICY_HEADER, POLICY_INSTRUCTIONS]
    for item in policies:
        lines.append(f"- 方針: {item['text']}（出典: {item['source']}）")
    return "\n".join(lines)


def asset_citation(asset_id: str, date: str = "", *, label: str = "判断資産") -> str:
    date_part = f"・{date[:10]} 教示" if date else ""
    return f"{label} {asset_id}{date_part}" if asset_id else (f"教わった知識{date_part}" if date else "教わった知識")
