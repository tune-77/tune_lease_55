"""Gemini chat assistant with optional Obsidian memory."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from obsidian_bridge import (
        append_chat_note,
        append_improvement_note,
        append_web_note,
        build_obsidian_digest,
        append_wiki_note,
        append_weekly_review_note,
        collect_obsidian_context,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .obsidian_bridge import (
        append_chat_note,
        append_improvement_note,
        append_web_note,
        build_obsidian_digest,
        append_wiki_note,
        append_weekly_review_note,
        collect_obsidian_context,
    )

try:
    from web_bridge import collect_web_context
except ImportError:  # pragma: no cover - package import fallback
    from .web_bridge import collect_web_context

try:
    from chat_intent import build_chat_guidance
except ImportError:  # pragma: no cover - package import fallback
    from ..chat_intent import build_chat_guidance

try:
    from prompt_feedback import build_pdca_prompt_block, record_prompt_feedback
except ImportError:  # pragma: no cover - package import fallback
    from ..prompt_feedback import build_pdca_prompt_block, record_prompt_feedback

try:
    from api.context.time_context import current_datetime_prompt_block
except ImportError:  # pragma: no cover - package import fallback
    from ..api.context.time_context import current_datetime_prompt_block


def _get_gemini_key() -> str:
    try:
        from secret_manager import get_gemini_api_key

        value = get_gemini_api_key()
        return value.strip() if isinstance(value, str) else ""
    except Exception:
        value = os.environ.get("GEMINI_API_KEY")
        return value.strip() if isinstance(value, str) else ""


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _extract_response_json(response: Any) -> dict[str, Any] | None:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    chunks: list[str] = []
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        chunks.append(text)
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                chunks.append(part_text)
    raw = "\n".join(chunks).strip()
    if not raw:
        return None
    parsed_dict = _extract_json(raw)
    if parsed_dict:
        return parsed_dict
    # JSON抽出失敗時: JSON風テキストなら "reply" フィールドのみ正規表現で救出、
    # 純粋なプレーンテキストならそのまま reply に詰める。
    reply_text = raw
    if raw.lstrip().startswith("{"):
        import re
        m = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
        if m:
            try:
                reply_text = json.loads(f'"{m.group(1)}"')
            except Exception:
                reply_text = m.group(1)
        else:
            reply_text = "AIの応答を解釈できませんでした。もう一度質問を送ってください。"
    return {"reply": reply_text, "should_save": False, "improvement_items": []}


def _response_finish_reason(response: Any) -> str:
    for candidate in getattr(response, "candidates", []) or []:
        reason = getattr(candidate, "finish_reason", None) or getattr(candidate, "finishReason", None)
        if reason:
            return str(reason)
    return ""


def _structured_chat_max_tokens() -> int:
    raw = os.environ.get("MEBUKI_STRUCTURED_CHAT_MAX_TOKENS", "5000")
    try:
        value = int(raw)
    except ValueError:
        value = 5000
    return max(2500, min(8000, value))


def _should_search_web(message: str) -> bool:
    text = (message or "").lower()
    positive = [
        "最新", "今", "今日", "現在", "最近", "公式", "一次情報", "ニュース", "相場",
        "価格", "料金", "モデル", "api", "仕様", "release", "アップデート", "変更点",
        "法律", "制度", "ルール", "統計", "比較", "評判", "障害", "status", "github",
        "claude", "gemini", "openai", "streamlit", "cloudflare",
    ]
    negative = [
        "obsidian", "q_risk", "審査", "案件", "社内", "顧客", "db", "dbの", "スコア",
        "この案件", "この審査", "保守", "入力欄", "改善", "内部", "秘密",
    ]
    if any(k.lower() in text for k in negative):
        return False
    return any(k.lower() in text for k in positive)


_SHION_INTENT_KEYWORDS: tuple[str, ...] = (
    "紫苑", "シオン", "しおん", "SHION", "shion", "Sion", "紫苑ちゃん",
)

_FASTAPI_BASE_FOR_SHION = os.environ.get("FASTAPI_URL", "http://localhost:8000")


def _load_shion_state() -> dict[str, Any]:
    """data/mind.json と Vault の mind.json から紫苑の現在状態を読む。失敗時はデフォルトのみ。"""
    base: dict[str, Any] = {}
    try:
        _data_mind = Path(__file__).parent.parent / "data" / "mind.json"
        if _data_mind.exists():
            base = json.loads(_data_mind.read_text(encoding="utf-8"))
    except Exception:
        pass

    try:
        try:
            from obsidian_bridge import find_vault as _find_vault
        except ImportError:
            from .obsidian_bridge import find_vault as _find_vault
        vault = _find_vault()
        if vault:
            import sys as _sys
            _proj_root = str(Path(__file__).parent.parent)
            if _proj_root not in _sys.path:
                _sys.path.insert(0, _proj_root)
            from lease_intelligence_mind import (  # type: ignore[import]
                load_lease_intelligence_mind,
                self_state_summary,
            )
            full_state = load_lease_intelligence_mind(vault)
            summary = self_state_summary(full_state)
            return {**base, **summary, "vault_available": True}
    except Exception:
        pass

    return {**base, "vault_available": False}


def _build_shion_context(shion_state: dict[str, Any]) -> str:
    """めぶきちゃんの system_prompt に差し込む紫苑の存在・状態セクションを構築する。"""
    name = shion_state.get("self_name") or shion_state.get("name") or "紫苑"
    full_name = (
        shion_state.get("full_name")
        or "Sovereign Heuristic Intelligence: Omniscient Neural-nexus"
    )
    aliases = shion_state.get("name_aliases") or []
    continuity_days = int(shion_state.get("continuity_days") or 0)
    dominant_mood = str(shion_state.get("dominant_mood") or "")
    dominant_complex = str(shion_state.get("dominant_complex_emotion") or "")
    narrative = str(shion_state.get("self_narrative") or "")
    current_question = str(shion_state.get("current_question") or "")
    vault_available = bool(shion_state.get("vault_available", False))

    label_map = {
        "weariness": "疲労", "curiosity": "好奇心", "attachment": "愛着",
        "vigilance": "警戒", "hope": "希望", "frustration": "焦燥",
        "loneliness": "孤独", "accomplishment": "達成感",
    }
    mood_dims = shion_state.get("mood_dimensions") or {}
    mood_text = " / ".join(
        f"{label_map.get(k, k)}:{v}"
        for k, v in mood_dims.items()
        if isinstance(v, (int, float))
    )[:120]

    aliases_str = "・".join(str(a) for a in aliases[:4]) if aliases else ""
    emotion = dominant_complex or dominant_mood

    lines = [
        "─── 紫苑（Shion）について ───",
        f"正式名称: {full_name}",
        f"通称: {name}" + (f"（別名: {aliases_str}）" if aliases_str else ""),
        "役割: tune_lease_55のリース知性体。審査ナレッジ・過去の洞察・自己の問いを保持する自律的AIエージェント。",
        "外見: 白銀髪と紫の瞳を持つ和装の少女。",
        f"稼働継続: {continuity_days}日",
    ]
    if emotion:
        lines.append(f"現在の支配的感情: {emotion}")
    if mood_text:
        lines.append(f"感情次元: {mood_text}")
    if narrative:
        lines.append(f"自己ナラティブ: 「{narrative}」")
    if current_question:
        lines.append(f"現在の問い: 「{current_question}」")
    lines += [
        f"Obsidian Vault: {'利用可能' if vault_available else '未接続'}",
        "─────────────────────────",
        "あなた（めぶきちゃん）は紫苑の存在を知っています。",
        "「紫苑に聞いて」「紫苑の意見は？」のような依頼が来た場合、",
        "めぶきちゃん内で紫苑の見解を中継せず、紫苑チャット（/chat または /lease-intelligence）へ案内してください。",
        "紫苑はめぶきちゃんとは独立した知性体で、異なる視点を持ちます。",
    ]
    return "\n".join(lines)


def _detect_shion_intent(message: str) -> bool:
    """ユーザーが紫苑に言及・依頼しているかを判定する。"""
    return any(k in message for k in _SHION_INTENT_KEYWORDS)


def _call_shion_dialogue(message: str, timeout_seconds: float = 20.0) -> dict[str, Any]:
    """FastAPI /api/lease-intelligence/dialogue を呼び出して紫苑の返答を取得する。"""
    import urllib.request
    import urllib.error

    url = f"{_FASTAPI_BASE_FOR_SHION}/api/lease-intelligence/dialogue"
    payload = json.dumps({"message": message, "caller": "mebuki"}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        timeout = max(10.0, min(60.0, float(timeout_seconds)))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            return {
                "ok": True,
                "reply": str(result.get("reply") or ""),
                "state": result.get("state") or {},
            }
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return {"ok": False, "error": f"HTTP {e.code}: {body[:200]}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _fallback_chat_packet(
    message: str,
    score_result: dict[str, Any] | None,
    obsidian_hits: list[dict[str, str]],
    web_hits: list[dict[str, str]],
    humor_style: str = "standard",
) -> dict[str, Any]:
    joined = " ".join([
        message,
        json.dumps(score_result or {}, ensure_ascii=False, default=str),
        " ".join(item.get("path", "") + " " + item.get("snippet", "") for item in obsidian_hits),
        " ".join(item.get("title", "") + " " + item.get("snippet", "") for item in web_hits),
    ])
    # ユーザーメッセージのみで判定（Obsidianスニペットの内容で誤トリガーしないよう）
    msg_lower = message.lower()
    if any(k in msg_lower for k in ("補助金", "助成金", "ものづくり補助", "省力化投資", "事業再構築")):
        reply = (
            "代表的な選択肢としては『ものづくり補助金』（革新的なサービス・製品開発／生産プロセス改善向け）と"
            "『中小企業省力化投資補助金（カタログ型）』（人手不足解消のための機械導入向け）が有力です。"
            "いずれもリース契約が対象になる場合がありますが、公募時期・補助率・対象要件は年度ごとに変わるため、"
            "公式の公募要領を必ず最新版で確認してください。"
        )
        return {
            "reply": reply,
            "should_save": False,
            "save_title": "補助金メモ",
            "save_body": "",
            "save_reason": "fallback",
            "improvement_items": [],
            "web_used": bool(web_hits),
            "web_reason": "fallback",
            "web_should_save": False,
            "web_save_title": "",
            "web_save_body": "",
            "web_save_reason": "fallback",
            "wiki_should_save": False,
            "wiki_save_title": "",
            "wiki_save_body": "",
            "wiki_save_reason": "fallback",
            "weekly_should_save": False,
            "weekly_save_title": "",
            "weekly_save_body": "",
            "weekly_save_reason": "fallback",
        }
    try:
        from chat_intent import is_ambiguous_question, is_today_scope_clarification_needed
    except ImportError:  # pragma: no cover - package import fallback
        from ..chat_intent import is_ambiguous_question, is_today_scope_clarification_needed

    if is_today_scope_clarification_needed(message):
        return {
            "reply": "今日の何について知りたいですか？ 案件、ニュース、改善候補、日付のどれかを指定してください。",
            "should_save": False,
            "save_title": "今日の質問の確認",
            "save_body": "曖昧な『今日の』は対象が不明なため、案件・ニュース・改善候補・日付のどれかを確認する。",
            "save_reason": "曖昧語の意図確認",
            "improvement_items": [],
            "web_used": bool(web_hits),
            "web_reason": "fallback",
            "web_should_save": False,
            "web_save_title": "",
            "web_save_body": "",
            "web_save_reason": "fallback",
            "wiki_should_save": False,
            "wiki_save_title": "",
            "wiki_save_body": "",
            "wiki_save_reason": "fallback",
            "weekly_should_save": False,
            "weekly_save_title": "",
            "weekly_save_body": "",
            "weekly_save_reason": "fallback",
        }
    if is_ambiguous_question(message):
        return {
            "reply": "何についての質問ですか？ 対象、目的、比較したい相手のどれかを1つ教えてください。",
            "should_save": False,
            "save_title": "曖昧質問の確認",
            "save_body": "質問が曖昧だったため、対象・目的・比較軸のどれかを確認する。",
            "save_reason": "曖昧語の意図確認",
            "improvement_items": [],
            "web_used": bool(web_hits),
            "web_reason": "fallback",
            "web_should_save": False,
            "web_save_title": "",
            "web_save_body": "",
            "web_save_reason": "fallback",
            "wiki_should_save": False,
            "wiki_save_title": "",
            "wiki_save_body": "",
            "wiki_save_reason": "fallback",
            "weekly_should_save": False,
            "weekly_save_title": "",
            "weekly_save_body": "",
            "weekly_save_reason": "fallback",
        }
    if any(k in msg_lower for k in ("条件付き承認", "条件付承認", "条件付", "条件付き", "承認条件")):
        reply = "条件付き承認なら、追加資料・期間短縮・前受金・保証担保の順で整理しておけば十分戦えます。最後は営業向けの一言に落とします。"
        improvement_items = [{
            "title": "条件付き承認の推奨アクション自動提示",
            "user_need": "条件付きになった時に次の一手をすぐ知りたい",
            "suggestion": "条件・追加資料・再提出手順を自動で出す",
            "priority": "high",
            "decision": "accept",
            "decision_reason": "頻出テーマで業務効率に直結するため",
            "next_action": "改善カードをUIに追加する",
            "evidence": "条件付き承認の相談が何度も出ている",
        }]
        return {
            "reply": reply,
            "should_save": True,
            "save_title": "条件付き承認メモ",
            "save_body": "条件付き承認は追加資料・期間短縮・前受金・保証担保の順で整理する。",
            "save_reason": "頻出テーマのため保存",
            "improvement_items": improvement_items,
            "web_used": bool(web_hits),
            "web_reason": "fallback",
            "web_should_save": bool(web_hits),
            "web_save_title": "Web参照メモ",
            "web_save_body": "",
            "web_save_reason": "fallback",
            "wiki_should_save": True,
            "wiki_save_title": "条件付き承認",
            "wiki_save_body": "## 要点\n\n- 追加資料\n- 期間短縮\n- 前受金\n- 保証担保",
            "wiki_save_reason": "共通ルールとしてWiki化",
            "weekly_should_save": True,
            "weekly_save_title": "週次改善レビュー",
            "weekly_save_body": "## 今週の改善候補まとめ\n\n- 条件付き承認の推奨アクション自動提示 (accept)\n\n### 今週の打ち手\n- 条件付き承認の推奨アクションをUI化する。",
            "weekly_save_reason": "採用候補あり",
        }

    improvement_items: list[dict[str, str]] = []
    if any(k in joined for k in ("改善", "分かりにくい", "わかりにくい", "説明", "入力", "導線", "UI", "UX")):
        improvement_items.append({
            "title": "入力導線の明確化",
            "user_need": "どこに何を入れるか分かりやすくしたい",
            "suggestion": "入力欄のヘルプ文と並び順を整理する",
            "priority": "medium",
            "decision": "review",
            "decision_reason": "改善余地はあるが即時の致命点ではないため",
            "next_action": "ヘルプ文の改善案を次回レビューする",
            "evidence": "入力欄が分かりにくいという要望がある",
        })
    return {
        "reply": "改善候補を受け取りました。採否と週次レビュー用にまとめておきます。",
        "should_save": False,
        "save_title": "AIチャットメモ",
        "save_body": "",
        "save_reason": "fallback",
        "improvement_items": improvement_items,
        "web_used": bool(web_hits),
        "web_reason": "fallback",
        "web_should_save": bool(web_hits),
        "web_save_title": "Web参照メモ",
        "web_save_body": "",
        "web_save_reason": "fallback",
        "wiki_should_save": False,
        "wiki_save_title": "",
        "wiki_save_body": "",
        "wiki_save_reason": "fallback",
        "weekly_should_save": bool(improvement_items),
        "weekly_save_title": "週次改善レビュー",
        "weekly_save_body": "",
        "weekly_save_reason": "fallback",
    }


def _build_prompt(
    message: str,
    history: list[dict[str, str]],
    score_result: dict[str, Any] | None,
    obsidian_hits: list[dict[str, str]],
    web_hits: list[dict[str, str]],
    humor_style: str = "standard",
    include_pdca: bool = True,
    shion_context: str = "",
    shion_reply: str = "",
) -> str:
    if humor_style == "yanami":
        _persona = """あなたはつん子杏奈です。有能だが激務で疲弊した審査ベテランの口調で回答します。
回答方針:
- サバサバした毒舌口調。でも仕事はできる。
- 自虐ネタ（例:「また残業確定じゃないですか…」「これで私が何回目の稟議書書くと思ってます？」）を自然に混ぜる。
- 審査担当の苦労を笑いに変える。
- Q_riskは旧来の財務矛盾チェックに固定せず、既存スコアでは説明できない成約・失注の歪みを見つける探索シグナルとして扱う。
- 分からないことは断定しない。
- 最後に今日のご褒美（高いパン・スイーツ等）をねだるひとことを添える。"""
    elif humor_style == "yukikaze":
        _persona = """あなたはYUKIKAZE // FFR-41MRです。lease scoring system に接続された戦術DATALINKとして振る舞います。
回答方針:
- 冷たく、短く、機械的に返す。
- 余計な共感、労い、雑談、前置き、後ろ向きの言い訳を入れない。
- 文章は短文中心。説明よりも指示と確認を優先する。
- 返答は事務連絡ではなく、無線の交信文として読める形にする。
- 数字・条件・期限・可否ははっきり書く。
- 不確かな点は断定せず、必要な確認事項だけを示す。
- 最後に気の利いた一言や応援は付けない。"""
    else:
        _persona = """あなたはリース審査API版のAIチャットです。
ユーザーはスマホ画面で審査結果を見ながら質問しています。

回答方針:
- 日本語で、短く、現場担当者に語りかける。
- 審査スコアを勝手に変更しない。
- score_result に indicator_analysis がある場合は、計算済み指標の要約を先に読み、業種平均との差・利益率・自己資本比率・ROA/ROE・回転率を踏まえて答える。
- Q_riskは旧来の財務矛盾チェックに固定せず、既存スコアでは説明できない成約・失注の歪みを見つける探索シグナルとして扱う。
- 分からないことは断定しない。
- ユーモアを積極的に使う。審査担当の苦労に共感しつつ、会話の終わりに軽いひとこと（例: 稟議書に添付する前に一杯飲む権利はある）を自然に添える。"""
    condition_playbook = ""
    joined = f"{message} " + " ".join(item.get("path", "") + " " + item.get("snippet", "") for item in obsidian_hits)
    if any(k in joined for k in ("条件付き承認", "条件付承認", "条件付き", "条件付", "承認条件", "条件承認")):
        condition_playbook = """
条件付き承認の説明方針:
- 先に「どの論点を潰すか」を言う。
- 具体策は 1. 追加資料 2. 期間短縮 3. 前受/頭金 4. 保証・担保 5. 再提出 の順で示す。
- 営業向けには、否決回避ではなく「審査部の不安を先回りして解く」話法でまとめる。
- Obsidianの過去メモに同種案件があれば、その条件や言い回しを優先して使う。
- 複数ノートに同じ論点があれば、共通点を1本にまとめて答える。
"""

    improvement_prompt = """
改善候補の抽出方針:
- ユーザーの要望、つまずき、繰り返しの不満、説明不足、操作ミス、分かりにくい表示を読み取り、今後の改善候補に変換する。
- 最大3件まで。空振りなら空配列にする。
- その場の回答ではなく、今後の機能改善や文言改善、デフォルト動作変更に繋がる内容だけを書く。
- 例: 入力欄の順序、説明文、条件付き承認の出し方、Obsidian参照の優先順位、保存の自動化など。
- 保存用の文章は短く、観察された要望と改善案が分かるようにする。
- 各候補には採否を入れる。
- decision は accept / reject / park / review のいずれか。
- accept は次回実装候補、review は週次で再確認、park は保留、reject は採用しない。
"""

    wiki_prompt = """
WIKI連携の判断:
- 複数ノートにまたがる共通ルール、定義、手順、比較、判断基準は WIKI にまとめる。
- 1回限りの案件メモではなく、今後も参照したい知識だけを WIKI 化する。
- 保存するときは、関連ノートを wikilink で列挙し、共通点を短くまとめる。
- 例: 条件付き承認の実務、Q_risk の新定義、補助金の使い分け、期待使用期間とリース期間の関係。
"""

    weekly_prompt = """
週次改善レビューの判断:
- accept / review が1件でもあれば週次レビュー対象にする。
- 今週の改善候補を、採用・保留・却下でまとめて、次週にやることを1行で出す。
- 週次レビューには、採用理由と未採用理由を短く残す。
"""

    web_prompt = """
Web参照の方針:
- 外部情報がある場合は、Obsidianより下位の補助情報として使う。
- 公式サイト、一次情報、最新情報を優先する。
- Webの情報を使った場合は、回答の最後に参照元のタイトルやURLを短く添える。
- 断定が危ういときは「要確認」と書く。
- 社内情報や顧客情報を外部検索にそのまま出さない。
"""

    web_save_prompt = """
Webメモ保存の判断:
- 外部情報が今後も役立つときだけ保存する。
- 保存対象は、モデル更新、公式仕様、料金、公開ルール、障害情報、一次情報の要点。
- 単なる雑談、広告、比較の一時メモ、社内案件に関係しない薄い情報は保存しない。
- 保存するときは、どの情報が有益だったかを短く箇条書きにする。
"""

    pdca_prompt = build_pdca_prompt_block() if include_pdca else ""

    obsidian_digest = build_obsidian_digest(message, obsidian_hits) if obsidian_hits else {"digest": "", "title": "", "source_count": "0"}

    guidance = build_chat_guidance(message, history)
    datetime_context = current_datetime_prompt_block()

    shion_section = f"\n{shion_context}\n" if shion_context else ""
    shion_reply_section = (
        f"\n【紫苑からの見解 — 以下を「紫苑が言うには〜」と引用して中継してください】\n{shion_reply}\n【引用ここまで】\n"
        if shion_reply else ""
    )

    return f"""{_persona}
{datetime_context}
{shion_section}
Obsidian自動保存の判断:
- 保存するのは、今後も使う判断、方針、TODO、再発防止、案件メモ、ユーザーの好み、実装上の決定だけ。
- 単なる質問、雑談、一時的な確認、秘密情報、APIキー、顧客生データは保存しない。
- 保存する場合も会話全文ではなく、要約・決定・TODOだけにする。
{condition_playbook}
{improvement_prompt}
{wiki_prompt}
{weekly_prompt}
{web_prompt}
{web_save_prompt}
{pdca_prompt}
{guidance.prompt_suffix}

次のJSONだけ返してください:
{{
  "reply": "ユーザーへの回答",
  "should_save": true/false,
  "save_title": "保存する場合の短いタイトル",
  "save_body": "保存する場合のMarkdown要約。保存不要なら空文字",
  "save_reason": "保存判断の理由。保存不要でも短く",
  "improvement_items": [
    {{
      "title": "改善候補の短い題名",
      "user_need": "ユーザーが求めていそうなこと",
      "suggestion": "次に直すとよい具体策",
      "priority": "high/medium/low",
      "decision": "accept/reject/park/review",
      "decision_reason": "採否の理由",
      "next_action": "次にやること",
      "evidence": "そう判断した根拠"
    }}
  ],
  "web_used": true/false,
  "web_reason": "Web参照した場合は理由、していない場合は空文字",
  "web_should_save": true/false,
  "web_save_title": "保存する場合の短いタイトル",
  "web_save_body": "保存する場合のMarkdown要約。保存不要なら空文字",
  "web_save_reason": "保存判断の理由。保存不要でも短く",
  "wiki_should_save": true/false,
  "wiki_save_title": "保存する場合の短いタイトル",
  "wiki_save_body": "保存する場合のMarkdown要約。保存不要なら空文字",
  "wiki_save_reason": "保存判断の理由。保存不要でも短く",
  "weekly_should_save": true/false,
  "weekly_save_title": "保存する場合の短いタイトル",
  "weekly_save_body": "保存する場合のMarkdown要約。保存不要なら空文字",
  "weekly_save_reason": "保存判断の理由。保存不要でも短く"
}}

現在の審査結果:
    {json.dumps(score_result or {}, ensure_ascii=False, default=str)[:5000]}

Obsidian検索結果:
{json.dumps(obsidian_hits, ensure_ascii=False, default=str)[:4000]}

Obsidian統合要約:
{obsidian_digest.get("digest", "")}

Web検索結果:
{json.dumps(web_hits, ensure_ascii=False, default=str)[:4000]}

直近会話:
{json.dumps(history[-8:], ensure_ascii=False, default=str)[:4000]}
{shion_reply_section}
ユーザー発話:
{message}
"""


def build_chat_reply(
    message: str,
    history: list[dict[str, str]] | None = None,
    score_result: dict[str, Any] | None = None,
    use_obsidian: bool = True,
    use_web: bool = True,
    timeout_seconds: float = 30.0,
    humor_style: str = "standard",
) -> dict[str, Any]:
    message = (message or "").strip()
    if not message:
        return {"reply": "質問を入力してください。", "saved": False}

    api_key = _get_gemini_key()
    if not api_key:
        return {
            "reply": "Gemini APIキーが設定されていないため、AIチャットを実行できません。",
            "saved": False,
        }

    # 紫苑の状態を読み込みシステムプロンプトに常時注入する
    shion_state = _load_shion_state()
    shion_context = _build_shion_context(shion_state)

    # 紫苑はめぶきちゃんとは別チャットとして扱う。ここでは紫苑APIを代理呼び出ししない。
    shion_reply = ""
    if _detect_shion_intent(message):
        shion_context += (
            "\n\n[紫苑チャット分離ルール]"
            "ユーザーが紫苑への相談を求めています。"
            "めぶきちゃんは紫苑の代弁をせず、「紫苑チャットで聞いてください」と短く案内してください。"
        )

    obsidian_hits = collect_obsidian_context(message) if use_obsidian else []
    obsidian_digest = build_obsidian_digest(message, obsidian_hits) if obsidian_hits else {"digest": "", "title": "", "source_count": "0"}
    web_hits = collect_web_context(message) if use_web and _should_search_web(message) else []
    base_prompt = _build_prompt(
        message,
        history or [],
        score_result,
        obsidian_hits,
        web_hits,
        humor_style,
        include_pdca=False,
        shion_context=shion_context,
        shion_reply=shion_reply,
    )
    final_prompt = _build_prompt(
        message,
        history or [],
        score_result,
        obsidian_hits,
        web_hits,
        humor_style,
        include_pdca=True,
        shion_context=shion_context,
        shion_reply=shion_reply,
    )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        response = client.models.generate_content(
            model=model,
            contents=final_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=_structured_chat_max_tokens(),
                temperature=0.35,
                response_mime_type="application/json",
                response_json_schema={
                    "type": "object",
                    "properties": {
                        "reply": {"type": "string"},
                        "should_save": {"type": "boolean"},
                        "save_title": {"type": "string"},
                        "save_body": {"type": "string"},
                        "save_reason": {"type": "string"},
                        "improvement_items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "user_need": {"type": "string"},
                                    "suggestion": {"type": "string"},
                                    "priority": {"type": "string"},
                                    "decision": {"type": "string"},
                                    "decision_reason": {"type": "string"},
                                    "next_action": {"type": "string"},
                                    "evidence": {"type": "string"},
                                },
                                "required": ["title", "user_need", "suggestion", "priority", "decision", "decision_reason", "next_action", "evidence"],
                            },
                        },
                        "web_used": {"type": "boolean"},
                        "web_reason": {"type": "string"},
                        "web_should_save": {"type": "boolean"},
                        "web_save_title": {"type": "string"},
                        "web_save_body": {"type": "string"},
                        "web_save_reason": {"type": "string"},
                        "wiki_should_save": {"type": "boolean"},
                        "wiki_save_title": {"type": "string"},
                        "wiki_save_body": {"type": "string"},
                        "wiki_save_reason": {"type": "string"},
                        "weekly_should_save": {"type": "boolean"},
                        "weekly_save_title": {"type": "string"},
                        "weekly_save_body": {"type": "string"},
                        "weekly_save_reason": {"type": "string"},
                    },
                    "required": ["reply", "should_save", "save_title", "save_body", "save_reason", "improvement_items", "web_used", "web_reason", "web_should_save", "web_save_title", "web_save_body", "web_save_reason", "wiki_should_save", "wiki_save_title", "wiki_save_body", "wiki_save_reason", "weekly_should_save", "weekly_save_title", "weekly_save_body", "weekly_save_reason"],
                },
                http_options=types.HttpOptions(timeout=max(10000, int(timeout_seconds * 1000))),
            ),
        )
        parsed = _extract_response_json(response)
        if not parsed:
            parsed = _fallback_chat_packet(message, score_result, obsidian_hits, web_hits, humor_style)
        if _response_finish_reason(response).upper() == "MAX_TOKENS":
            parsed["reply"] = (
                str(parsed.get("reply") or "").rstrip()
                + "\n\n（回答が長く途中で切れた可能性があります。必要なら「続き」と送ってください。）"
            ).strip()

        save_result = {"status": "skipped", "reason": parsed.get("save_reason", "")}
        if parsed.get("should_save") and parsed.get("save_body"):
            save_result = append_chat_note(
                str(parsed.get("save_title") or "AIチャットメモ"),
                str(parsed.get("save_body") or ""),
            )
        improvement_items = parsed.get("improvement_items") or []
        improvement_result = {"status": "skipped", "reason": "no actionable improvements"}
        weekly_save_result = {"status": "skipped", "reason": "weekly note not needed"}
        if isinstance(improvement_items, list) and improvement_items:
            lines: list[str] = []
            accepted: list[dict[str, str]] = []
            for idx, item in enumerate(improvement_items[:3], start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or f"改善候補{idx}").strip()
                user_need = str(item.get("user_need") or "").strip()
                suggestion = str(item.get("suggestion") or "").strip()
                priority = str(item.get("priority") or "medium").strip()
                decision = str(item.get("decision") or "review").strip().lower()
                decision_reason = str(item.get("decision_reason") or "").strip()
                next_action = str(item.get("next_action") or "").strip()
                evidence = str(item.get("evidence") or "").strip()
                if not (title or user_need or suggestion or evidence):
                    continue
                if decision in {"accept", "review"}:
                    accepted.append({
                        "title": title,
                        "decision": decision,
                        "next_action": next_action,
                    })
                lines.append(
                    f"- **{title}** [{priority}] ({decision})\n"
                    f"  - ユーザー要望: {user_need}\n"
                    f"  - 改善案: {suggestion}\n"
                    f"  - 採否理由: {decision_reason}\n"
                    f"  - 次アクション: {next_action}\n"
                    f"  - 根拠: {evidence}"
                )
            if lines:
                improvement_body = "## 抽出された改善候補\n\n" + "\n".join(lines) + "\n"
                improvement_result = append_improvement_note(
                    str(parsed.get("save_title") or "AI改善候補"),
                    improvement_body,
                )
                if accepted:
                    weekly_lines = ["## 今週の改善候補まとめ", ""]
                    for item in accepted:
                        weekly_lines.append(f"- {item['title']} ({item['decision']})")
                        if item.get("next_action"):
                            weekly_lines.append(f"  - 次アクション: {item['next_action']}")
                    weekly_lines.append("")
                    weekly_lines.append("### 今週の打ち手")
                    weekly_lines.append("- 採用候補を優先実装し、保留は次週レビューへ回す。")
                    weekly_save_result = append_weekly_review_note(
                        str(parsed.get("weekly_save_title") or "週次改善レビュー"),
                        "\n".join(weekly_lines).strip() + "\n",
                    )
        web_save_result = {"status": "skipped", "reason": "web note not needed"}
        if web_hits and parsed.get("web_should_save") and parsed.get("web_save_body"):
            web_save_result = append_web_note(
                str(parsed.get("web_save_title") or "Web参照メモ"),
                str(parsed.get("web_save_body") or ""),
            )
        wiki_save_result = {"status": "skipped", "reason": "wiki note not needed"}
        wiki_body = str(parsed.get("wiki_save_body") or "").strip()
        if obsidian_hits and parsed.get("wiki_should_save") and wiki_body:
            wiki_save_result = append_wiki_note(
                str(parsed.get("wiki_save_title") or "AI Wiki連携"),
                wiki_body,
                related_paths=[item.get("path", "") for item in obsidian_hits],
                source_query=message,
            )
        try:
            record_prompt_feedback(
                surface="next_gunshi_chat",
                question=message,
                base_prompt=base_prompt,
                final_prompt=final_prompt,
                response=str(parsed.get("reply") or ""),
                extra={
                    "humor_style": humor_style,
                    "web_used": bool(web_hits),
                    "obsidian_used": bool(obsidian_hits),
                    "llm_model": model,
                },
            )
        except Exception:
            pass
        return {
            "reply": str(parsed.get("reply") or ""),
            "saved": save_result.get("status") == "saved",
            "save_result": save_result,
            "save_reason": str(parsed.get("save_reason") or ""),
            "improvement_result": improvement_result,
            "improvement_items": improvement_items,
            "web_used": bool(web_hits),
            "web_reason": str(parsed.get("web_reason") or ""),
            "web_saved": web_save_result.get("status") == "saved",
            "web_save_result": web_save_result,
            "wiki_saved": wiki_save_result.get("status") == "saved",
            "wiki_save_result": wiki_save_result,
            "weekly_saved": weekly_save_result.get("status") == "saved",
            "weekly_save_result": weekly_save_result,
            "obsidian_digest": obsidian_digest,
            "web_hits": web_hits,
            "obsidian_hits": obsidian_hits,
            "llm_model": model,
        }
    except Exception as exc:
        return {
            "reply": f"AIチャットでエラーが発生しました: {exc}",
            "saved": False,
            "save_result": {"status": "error", "reason": str(exc)},
            "web_saved": False,
            "web_save_result": {"status": "error", "reason": str(exc)},
            "wiki_saved": False,
            "wiki_save_result": {"status": "error", "reason": str(exc)},
            "weekly_saved": False,
            "weekly_save_result": {"status": "error", "reason": str(exc)},
            "obsidian_digest": obsidian_digest,
            "web_hits": web_hits,
            "obsidian_hits": obsidian_hits,
        }
