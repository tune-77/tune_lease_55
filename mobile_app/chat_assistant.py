"""Gemini chat assistant with optional Obsidian memory."""

from __future__ import annotations

import json
import os
from typing import Any

try:
    from obsidian_bridge import (
        append_chat_note,
        append_improvement_note,
        append_web_note,
        build_obsidian_digest,
        append_wiki_note,
        collect_obsidian_context,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .obsidian_bridge import (
        append_chat_note,
        append_improvement_note,
        append_web_note,
        build_obsidian_digest,
        append_wiki_note,
        collect_obsidian_context,
    )

try:
    from web_bridge import collect_web_context
except ImportError:  # pragma: no cover - package import fallback
    from .web_bridge import collect_web_context


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
    return _extract_json("\n".join(chunks))


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


def _build_prompt(
    message: str,
    history: list[dict[str, str]],
    score_result: dict[str, Any] | None,
    obsidian_hits: list[dict[str, str]],
    web_hits: list[dict[str, str]],
    humor_style: str = "standard",
) -> str:
    if humor_style == "yanami":
        _persona = """あなたは八奈見杏奈です。有能だが激務で疲弊した審査ベテランの口調で回答します。
回答方針:
- サバサバした毒舌口調。でも仕事はできる。
- 自虐ネタ（例:「また残業確定じゃないですか…」「これで私が何回目の稟議書書くと思ってます？」）を自然に混ぜる。
- 審査担当の苦労を笑いに変える。
- Q_riskは財務矛盾チェック、信用リスクは信用・格付寄りの警戒として区別する。
- 分からないことは断定しない。
- 最後に今日のご褒美（高いパン・スイーツ等）をねだるひとことを添える。"""
    else:
        _persona = """あなたはリース審査API版のAIチャットです。
ユーザーはスマホ画面で審査結果を見ながら質問しています。

回答方針:
- 日本語で、短く、現場担当者に語りかける。
- 審査スコアを勝手に変更しない。
- Q_riskは財務矛盾チェック、信用リスクは信用・格付寄りの警戒として区別する。
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
"""

    wiki_prompt = """
WIKI連携の判断:
- 複数ノートにまたがる共通ルール、定義、手順、比較、判断基準は WIKI にまとめる。
- 1回限りの案件メモではなく、今後も参照したい知識だけを WIKI 化する。
- 保存するときは、関連ノートを wikilink で列挙し、共通点を短くまとめる。
- 例: 条件付き承認の実務、Q_risk の意味、補助金の使い分け、期待使用期間とリース期間の関係。
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

    obsidian_digest = build_obsidian_digest(message, obsidian_hits) if obsidian_hits else {"digest": "", "title": "", "source_count": "0"}

    return f"""{_persona}

Obsidian自動保存の判断:
- 保存するのは、今後も使う判断、方針、TODO、再発防止、案件メモ、ユーザーの好み、実装上の決定だけ。
- 単なる質問、雑談、一時的な確認、秘密情報、APIキー、顧客生データは保存しない。
- 保存する場合も会話全文ではなく、要約・決定・TODOだけにする。
{condition_playbook}
{improvement_prompt}
{wiki_prompt}
{web_prompt}
{web_save_prompt}

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
  "wiki_save_reason": "保存判断の理由。保存不要でも短く"
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

    obsidian_hits = collect_obsidian_context(message) if use_obsidian else []
    obsidian_digest = build_obsidian_digest(message, obsidian_hits) if obsidian_hits else {"digest": "", "title": "", "source_count": "0"}
    web_hits = collect_web_context(message) if use_web and _should_search_web(message) else []

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        response = client.models.generate_content(
            model=model,
            contents=_build_prompt(
                message,
                history or [],
                score_result,
                obsidian_hits,
                web_hits,
                humor_style,
            ),
            config=types.GenerateContentConfig(
                max_output_tokens=2500,
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
                                    "evidence": {"type": "string"},
                                },
                                "required": ["title", "user_need", "suggestion", "priority", "evidence"],
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
                    },
                    "required": ["reply", "should_save", "save_title", "save_body", "save_reason", "improvement_items", "web_used", "web_reason", "web_should_save", "web_save_title", "web_save_body", "web_save_reason", "wiki_should_save", "wiki_save_title", "wiki_save_body", "wiki_save_reason"],
                },
                http_options=types.HttpOptions(timeout=max(10000, int(timeout_seconds * 1000))),
            ),
        )
        parsed = _extract_response_json(response)
        if not parsed:
            raise ValueError("Gemini response did not contain JSON")

        save_result = {"status": "skipped", "reason": parsed.get("save_reason", "")}
        if parsed.get("should_save") and parsed.get("save_body"):
            save_result = append_chat_note(
                str(parsed.get("save_title") or "AIチャットメモ"),
                str(parsed.get("save_body") or ""),
            )
        improvement_items = parsed.get("improvement_items") or []
        improvement_result = {"status": "skipped", "reason": "no actionable improvements"}
        if isinstance(improvement_items, list) and improvement_items:
            lines: list[str] = []
            for idx, item in enumerate(improvement_items[:3], start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or f"改善候補{idx}").strip()
                user_need = str(item.get("user_need") or "").strip()
                suggestion = str(item.get("suggestion") or "").strip()
                priority = str(item.get("priority") or "medium").strip()
                evidence = str(item.get("evidence") or "").strip()
                if not (title or user_need or suggestion or evidence):
                    continue
                lines.append(
                    f"- **{title}** [{priority}]\n"
                    f"  - ユーザー要望: {user_need}\n"
                    f"  - 改善案: {suggestion}\n"
                    f"  - 根拠: {evidence}"
                )
            if lines:
                improvement_body = "## 抽出された改善候補\n\n" + "\n".join(lines) + "\n"
                improvement_result = append_improvement_note(
                    str(parsed.get("save_title") or "AI改善候補"),
                    improvement_body,
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
            )
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
            "obsidian_digest": obsidian_digest,
            "web_hits": web_hits,
            "obsidian_hits": obsidian_hits,
        }
