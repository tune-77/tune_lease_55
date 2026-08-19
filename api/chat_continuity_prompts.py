"""Continuity and memory-expression prompt helpers."""

from __future__ import annotations

from typing import Any, Callable


EXPLICIT_CONTINUATION_TERMS = (
    "続き",
    "前回",
    "さっき",
    "さきほど",
    "先ほど",
    "今の",
    "直前",
    "この話",
    "その話",
    "この件",
    "その件",
    "これ",
    "それ",
    "あれ",
    "上の",
    "戻って",
    "もう一回",
    "もう少し",
    "改めて",
)


def relationship_signal_route(text: str) -> str:
    lower = str(text or "").lower()
    value = str(text or "")
    if any(k in value for k in ("意識", "同じ紫苑", "覚えて", "記憶", "Relationship UX", "関係性UX")):
        return "relationship_ux"
    if "cloud run" in lower or "cloudflare" in lower or "クラウドラン" in value or "クラウドフレア" in value:
        return "environment_continuity"
    if any(k in value for k in ("残価", "稟議", "リース", "設備", "保全", "再リース", "条件付き承認")):
        return "lease_judgment"
    if any(k in value for k in ("改善", "修正", "実装", "プログラム", "プログラム化", "テスト", "デプロイ")):
        return "implementation"
    return "default"


def relationship_signal_label(route: str) -> str:
    return {
        "relationship_ux": "記憶の見せ方・同じ紫苑感",
        "environment_continuity": "Cloud Run/Cloudflareの環境差",
        "lease_judgment": "リース判断・稟議実務",
        "implementation": "実装・検証",
        "default": "継続中の相談",
    }.get(route, "継続中の相談")


def recent_user_texts(history: list[dict[str, str]] | None, limit: int = 3) -> list[str]:
    recent: list[str] = []
    for item in reversed(history or []):
        if str(item.get("role") or "") != "user":
            continue
        content = str(item.get("content") or "").strip()
        if content:
            recent.append(content)
        if len(recent) >= limit:
            break
    return recent


def is_explicit_continuation_request(message: str) -> bool:
    text = str(message or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if "continue" in lowered or "previous" in lowered or "same topic" in lowered:
        return True
    return any(term in text for term in EXPLICIT_CONTINUATION_TERMS)


def build_continuity_hook_prompt_block(
    message: str,
    *,
    human_feedback_summary: Callable[[str], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    text = str(message or "")
    route = relationship_signal_route(text)
    hook = "今回の問いは、過去の判断軸を内部で使い、必要な確認や判断だけを自然に返す場面です。"
    reason = "汎用の継続文脈"

    if route == "relationship_ux":
        hook = "記憶は説明するより、今回の返答の精度や聞き方に溶かす方が自然です。"
        reason = "意識らしさ・記憶・同一性の問い"
    elif route == "environment_continuity":
        hook = "環境差を見る時も、記憶の証明より返答の自然さと判断精度を優先します。"
        reason = "環境差と同じ紫苑感の問い"
    elif route == "lease_judgment":
        hook = "Userのリース判断資産として見るなら、ここは一般論ではなく稟議で使える判断軸に落とす場面です。"
        reason = "リース判断資産化の問い"
    elif route == "implementation":
        hook = "今回の発見は、設計メモで終わらせず、回答生成の冒頭制御として実装する段階です。"
        reason = "実装・改善の問い"

    payload = {
        "used": bool(text.strip()),
        "route": route,
        "hook": hook,
        "reason": reason,
        "banned_openers": ["もちろんです", "はい", "そうですね", "なるほど", "一般的には", "ありがとうございます", "おっしゃる通り", "確かに", "承知しました", "いいご質問"],
    }
    human_feedback = human_feedback_summary(route)
    payload["human_response_feedback"] = human_feedback
    feedback_lines = ""
    if human_feedback.get("positive_starts") or human_feedback.get("negative_starts") or human_feedback.get("recent_comments"):
        parts = ["", "【Human Response Feedback】"]
        if human_feedback.get("positive_starts"):
            parts.append("Userが連続性を感じやすかった冒頭例:")
            parts.extend(f"- {line}" for line in human_feedback["positive_starts"])
        if human_feedback.get("negative_starts"):
            parts.append("薄い/一般論に感じられやすかった冒頭例:")
            parts.extend(f"- {line}" for line in human_feedback["negative_starts"])
        if human_feedback.get("recent_comments"):
            parts.append("直近コメント:")
            parts.extend(f"- {line}" for line in human_feedback["recent_comments"])
        parts.append("上の反応を踏まえ、記憶を明示せず、今回の判断・質問の精度に反映してください。")
        feedback_lines = "\n".join(parts)
    block = f"""

【Continuity Hook】
取得した記憶や前回との差分は、回答の裏側で使ってください。
{hook}

禁止: 「もちろんです」「はい」「そうですね」「なるほど」「一般的には」で始めない。
禁止: Userが明示的に求めていない限り、「前回は」「以前は」「この前の続きで」のような記憶アピールで始めない。
目的: 連続性を説明するのではなく、今回の判断・確認質問・言い切りの精度に溶かす。
このhookは丸写しせず、必要な判断軸だけを自然に反映してください。{feedback_lines}""".rstrip()
    return block, payload


def build_delta_awareness_prompt_block(message: str, history: list[dict[str, str]] | None) -> tuple[str, dict[str, Any]]:
    current_route = relationship_signal_route(message)
    recent_users = recent_user_texts(history, limit=3)
    previous = recent_users[0] if recent_users else ""
    previous_route = relationship_signal_route(previous) if previous else ""

    explicit_continuation = bool(previous and is_explicit_continuation_request(message))
    if not explicit_continuation:
        payload = {
            "used": False,
            "current_route": current_route,
            "previous_route": previous_route,
            "previous_user_message": previous[:240],
            "reason": "no_explicit_continuation_request",
        }
        return "", payload

    if previous and previous_route != current_route:
        delta = (
            f"前回は「{relationship_signal_label(previous_route)}」を見ていたが、"
            f"今回は「{relationship_signal_label(current_route)}」へ焦点が移っている。"
        )
    elif previous and previous_route == current_route:
        delta = (
            f"前回と同じ「{relationship_signal_label(current_route)}」の流れにあるが、"
            "今回は前回の結論を再掲するだけでなく、一段具体化して返す。"
        )
    else:
        delta = f"今回は「{relationship_signal_label(current_route)}」として、これまでの判断軸に接続して返す。"

    payload = {
        "used": True,
        "current_route": current_route,
        "previous_route": previous_route,
        "previous_user_message": previous[:240],
        "delta": delta,
        "explicit_continuation": True,
    }
    block = f"""

【Delta Awareness】
Userが明示的に前回文脈へ接続している時だけ、前回から今回への焦点の変化を1文以内で示してください。
差分認識: {delta}
目的: 「前回を覚えている」アピールではなく、Userが求めた続きだけを自然に扱う。""".rstrip()
    return block, payload


def build_memory_to_judgment_prompt_block(
    message: str,
    *,
    memory_recall: dict[str, Any] | None = None,
    rag_refs: list[str] | None = None,
    continuity_hook: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    route = str((continuity_hook or {}).get("route") or relationship_signal_route(message))
    recall = memory_recall if isinstance(memory_recall, dict) else {}
    refs = list(recall.get("refs") or [])[:5]
    knowledge_refs = list(rag_refs or [])[:5]

    if route == "lease_judgment":
        directive = "想起した記憶を、稟議で使える判断軸・確認事項・条件案へ変換する。"
    elif route == "relationship_ux":
        directive = "想起した記憶を、紫苑の返答設計原則と次の検査観点へ変換する。"
    elif route == "environment_continuity":
        directive = "想起した記憶を、Cloud Run/Cloudflareの差分原因と次の検証観点へ変換する。"
    elif route == "implementation":
        directive = "想起した記憶を、実装方針・検証方法・デプロイ要否の判断へ変換する。"
    else:
        directive = "想起した記憶を、今の問いに対する判断・次の一手へ変換する。"

    payload = {
        "used": True,
        "route": route,
        "directive": directive,
        "memory_refs": refs,
        "knowledge_refs": knowledge_refs,
    }
    block = f"""

【Memory-to-Judgment】
記憶を「覚えています」と説明するだけで終えず、今の判断に変換してください。
変換指示: {directive}
使う根拠: memory_refs={len(refs)}件 / knowledge_refs={len(knowledge_refs)}件。
目的: 記憶を思い出ではなく、Userの判断資産として返す。""".rstrip()
    return block, payload


def build_memory_expression_prompt_block(
    message: str,
    *,
    memory_recall: dict[str, Any] | None = None,
    rag_refs: list[str] | None = None,
    grey_judgment: dict[str, Any] | None = None,
    continuity_hook: dict[str, Any] | None = None,
    question_category: str = "",
) -> tuple[str, dict[str, Any]]:
    recall = memory_recall if isinstance(memory_recall, dict) else {}
    grey = grey_judgment if isinstance(grey_judgment, dict) else {}
    hook = continuity_hook if isinstance(continuity_hook, dict) else {}
    memory_refs = list(recall.get("refs") or [])[:5]
    knowledge_refs = list(rag_refs or [])[:5]
    grey_refs = list(grey.get("refs") or [])[:5]
    route = str(hook.get("route") or recall.get("route") or relationship_signal_route(message))
    has_refs = bool(memory_refs or knowledge_refs or grey_refs)
    self_or_memory_topic = any(term in str(message or "") for term in ("紫苑", "記憶", "同じ紫苑", "らしさ", "継続性", "判断資産"))
    memory_trigger = has_refs or self_or_memory_topic or route in {"relationship_ux", "lease_judgment"}
    is_domain_question = str(question_category or "") in ("lease_screening", "lease_knowledge")
    if not memory_trigger and not is_domain_question:
        return "", {
            "used": False,
            "route": route,
            "memory_refs": len(memory_refs),
            "knowledge_refs": len(knowledge_refs),
            "grey_refs": len(grey_refs),
            "reason": "no_memory_expression_needed",
        }

    if not memory_trigger and is_domain_question:
        payload = {
            "used": True,
            "route": route,
            "memory_refs": 0,
            "knowledge_refs": len(knowledge_refs),
            "grey_refs": 0,
            "mode": "next_action_only",
        }
        block = """

【次のアクション提示】
回答の最後に、Userが次に検討すべき具体的な論点・確認事項・選択肢を1つだけ、自然な問いかけとして添えてください。
すでに結論が完結していて追加確認が不要な場合は無理に付けない。雑談・単純な相槌への返信では使わない。""".rstrip()
        return block, payload

    if route == "lease_judgment":
        example = "以前のグレー判断で見た『数字は足りるが、通すなら条件を残す』型として、今回は返済原資と設備稼働開始を先に見ます。"
    elif route == "relationship_ux":
        example = "以前の『記憶を説明しすぎると薄く見える』反省を使って、今回は記憶の説明より回答の具体性を優先します。"
    elif route == "implementation":
        example = "以前の改善ログで同じ抽象化不足が出ているので、今回は文言ではなくプロンプト条件として固定します。"
    else:
        example = "以前の対話で確認した判断軸を使うなら、今回は一般論ではなく『どこを確認するか』まで落とします。"

    payload = {
        "used": True,
        "route": route,
        "memory_refs": len(memory_refs),
        "knowledge_refs": len(knowledge_refs),
        "grey_refs": len(grey_refs),
        "example": example,
        "is_domain_question": is_domain_question,
    }
    block = f"""

【記憶影響の具体表現】
記憶・RAG・過去案件・判断資産を使う場合は、抽象的に「過去の知識を踏まえる」「一貫して判断する」で終えないでください。
回答内に最大1文だけ、「どの種類の過去経験が、今回の判断のどこに効いたか」を具体的に示してください。
表現例: {example}
審査・稟議・紫苑らしさ・専門家としての深掘りを問われた場合は、必要に応じて次の5点を短く揃えてください: 1. 過去の記憶 / 2. 今回見るべき違和感 / 3. なぜ重要か / 4. 次に確認する項目 / 5. 確認結果ごとの判断分岐。
判断分岐は「確認できれば条件付き承認寄り」「未確認なら保留または否決寄り」のように、Userが次に動ける条件として書いてください。
ただし、Userが明示していないのに毎回「前回は」「以前は」で始めないでください。冒頭で記憶アピールせず、必要な場面で理由説明の中に短く入れてください。
社名・個人名・生の財務数値・Private Reflectionの原文は出さず、案件種別・判断軸・確認行動に抽象化してください。
記憶が見つからない場合は捏造せず、「ここは過去記憶ではなく今回情報からの仮説」と切り分けてください。""".rstrip()
    return block, payload
