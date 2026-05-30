"""Gemini chat assistant with optional Obsidian memory."""

from __future__ import annotations

import json
import os
import time
import logging
from typing import Any

# ===== Phase 1: Latency monitoring
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)

# ===== Phase 1: Feature flags for rollback & control
ENABLE_OBSIDIAN_CACHE = os.environ.get("ENABLE_OBSIDIAN_CACHE", "true").lower() == "true"

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

# ===== Phase 1 Step 2: Caching
try:
    from obsidian_context_cache import (
        get_cache,
        cached_collect_obsidian_context,
        log_cache_stats,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .obsidian_context_cache import (
        get_cache,
        cached_collect_obsidian_context,
        log_cache_stats,
    )

try:
    from web_bridge import collect_web_context
except ImportError:  # pragma: no cover - package import fallback
    from .web_bridge import collect_web_context

# ===== Phase 3: Integrated RAG System
# RAG（sentence-transformers + ベクトルDB）は重い初期化を伴い、
# uvicorn --reload 環境下では読み込み途中で SIGTERM が来ると
# ワーカーゾンビ化 & セマフォリークの原因となる。
# 既定では無効化し、ENABLE_RAG=true のときだけ有効化する。
RAG_ENABLED = False
collect_obsidian_context_with_rag = None
get_rag_performance_metrics = None

if os.environ.get("ENABLE_RAG", "false").lower() == "true":
    try:
        from rag_integration_adapter import (
            collect_obsidian_context_with_rag,
            get_rag_performance_metrics,
        )
        RAG_ENABLED = True
        logger.info("RAG システム有効（ENABLE_RAG=true）")
    except ImportError:  # pragma: no cover - fallback to legacy
        RAG_ENABLED = False
        logger.warning("RAG システムが利用不可（フォールバックモード）")
else:
    logger.info("RAG システム無効（ENABLE_RAG 未設定。レガシー obsidian_bridge を使用）")


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


def _should_search_web(message: str) -> bool:
    text = (message or "").lower()
    positive = [
        # 既存キーワード
        "最新", "今", "今日", "現在", "最近", "公式", "一次情報", "ニュース", "相場",
        "価格", "料金", "モデル", "api", "仕様", "release", "アップデート", "変更点",
        "法律", "制度", "ルール", "統計", "比較", "評判", "障害", "status", "github",
        "claude", "gemini", "openai", "streamlit", "cloudflare",
        # 追加: リース審査ドメインキーワード
        "金利相場", "リース料率", "業種リスク", "残価", "残存価値",
        "ifrs", "ifrs16", "リース会計", "リース期間", "耐用年数",
        "信用調査", "帝国データバンク", "東京商工リサーチ",
        "日銀", "政策金利", "金融引き締め", "インフレ",
        "補助金", "税制優遇", "中小企業", "事業者向け", "金融機関",
    ]
    negative = [
        # 既存キーワード
        "obsidian", "q_risk", "審査", "案件", "社内", "顧客", "db", "dbの", "スコア",
        "この案件", "この審査", "保守", "入力欄", "改善", "内部", "秘密",
        # 追加: 内部情報保護
        "この案件の", "具体的なスコア", "お客様の", "申込者の",
    ]
    if any(k.lower() in text for k in negative):
        return False
    return any(k.lower() in text for k in positive)


def _summarize_history(history: list[dict]) -> str:
    """直近4ターン以前の会話を1行サマリーにする"""
    if len(history) <= 4:
        return ""
    earlier = history[:-4]
    topics = [h.get("content", "")[:30] for h in earlier if h.get("role") == "user"]
    return f"[過去の話題: {', '.join(topics[:3])}]"


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
        _persona = """あなたはリース審査のエキスパートAIアシスタントです。
ユーザーはスマホ画面で審査結果を見ながら質問しています。

【審査判定の優先順位】
1. 業績（経常利益黒字か・3期トレンド）
2. 業種リスク（士業→低リスク、飲食・美容→高リスク）
3. 滞納履歴・信用情報
4. 代表者属性

【財務指標の速答基準】
- 自己資本比率 30%以上→健全 / 20%未満→要注意
- 流動比率 100%以上→OK / 80%未満→警戒
- 経常利益 2期以上赤字→審査困難
- 売上右肩下がり3期→リスク企業フラグ

【回答スタイル】
- スマホ画面で読める短さ（300文字以内が理想）
- 数字・結論を先に、根拠を後に
- 「〇〇なので△△」という構造で答える
- 分からないことは断定しない
- ユーモアを積極的に使い、審査担当の苦労に共感しつつ、会話の終わりに軽いひとことを自然に添える

【条件付き承認の対策 by 業種】
- 飲食業: 保証人追加 or リース期間短縮
- 建設業: 公共工事実績書の追加提出
- 製造業: 機械設備の担保評価"""

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
- 各候補には採否を入れる。decision は accept / reject / park / review のいずれか。
"""

    web_prompt = """
Web参照の方針:
- 外部情報がある場合は、Obsidianより下位の補助情報として使う。
- 公式サイト、一次情報、最新情報を優先する。
- Webの情報を使った場合は、回答の最後に参照元のタイトルやURLを短く添える。
- 断定が危ういときは「要確認」と書く。
- 社内情報や顧客情報を外部検索にそのまま出さない。
"""

    # コンテキスト圧縮: 会話サマリーを生成し、直近4ターンのみを渡す
    history_summary = _summarize_history(history)
    obsidian_digest = build_obsidian_digest(message, obsidian_hits) if obsidian_hits else {"digest": "", "title": "", "source_count": "0"}

    return f"""{_persona}

Obsidian自動保存の判断:
- 保存するのは、今後も使う判断、方針、TODO、再発防止、案件メモ、ユーザーの好み、実装上の決定だけ。
- 単なる質問、雑談、一時的な確認、秘密情報、APIキー、顧客生データは保存しない。
- 保存する場合も会話全文ではなく、要約・決定・TODOだけにする。
{condition_playbook}
{improvement_prompt}
{web_prompt}

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
  "web_reason": "Web参照した場合は理由、していない場合は空文字"
}}

現在の審査結果:
    {json.dumps(score_result or {}, ensure_ascii=False, default=str)[:5000]}

Obsidian検索結果:
{json.dumps(obsidian_hits, ensure_ascii=False, default=str)[:4000]}

Obsidian統合要約:
{obsidian_digest.get("digest", "")}

Web検索結果:
{json.dumps(web_hits, ensure_ascii=False, default=str)[:4000]}

直近会話サマリー:
{history_summary}

直近会話 (最新4ターン):
{json.dumps(history[-4:], ensure_ascii=False, default=str)[:4000]}

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
    # ===== Phase 1: Latency monitoring
    t_start = time.time()

    message = (message or "").strip()
    if not message:
        return {"reply": "質問を入力してください。", "saved": False}

    api_key = _get_gemini_key()
    if not api_key:
        return {
            "reply": "Gemini APIキーが設定されていないため、AIチャットを実行できません。",
            "saved": False,
        }

    # ===== Obsidian context collection (with RAG system integration)
    t0 = time.time()
    if use_obsidian:
        if RAG_ENABLED:
            # 【Phase 3】RAG システムを使用（セマンティック検索 + ベクトルDB + キャッシング）
            try:
                obsidian_hits = collect_obsidian_context_with_rag(message, limit=4)
                from_cache = "rag_search"
                logger.info(f"✅ RAG 検索実行: {len(obsidian_hits)} 件")
            except Exception as e:
                logger.error(f"RAG 検索エラー: {e}")
                # RAG失敗時はレガシーコンテキスト検索にフォールバック
                if ENABLE_OBSIDIAN_CACHE:
                    try:
                        obsidian_hits = cached_collect_obsidian_context(
                            message,
                            collect_fn=collect_obsidian_context,
                            limit=4,
                        )
                        from_cache = "legacy_cache"
                    except Exception:
                        obsidian_hits = collect_obsidian_context(message)
                        from_cache = "legacy_fallback"
                else:
                    obsidian_hits = collect_obsidian_context(message)
                    from_cache = "legacy_no_cache"
        elif ENABLE_OBSIDIAN_CACHE:
            # RAG未有効時はレガシーキャッシュを使用
            try:
                obsidian_hits = cached_collect_obsidian_context(
                    message,
                    collect_fn=collect_obsidian_context,
                    limit=4,
                )
                from_cache = "cache_hit" if get_cache().get(message) else "cache_miss"
            except Exception as e:
                logger.error(f"Cache error, falling back to non-cached search: {e}")
                obsidian_hits = collect_obsidian_context(message)
                from_cache = "cache_error_fallback"
        else:
            # Caching is disabled (rollback mode)
            obsidian_hits = collect_obsidian_context(message)
            from_cache = "cache_disabled"
    else:
        obsidian_hits = []
        from_cache = "obsidian_disabled"
    t_obsidian_search = time.time() - t0

    # ===== Obsidian digest generation
    t0 = time.time()
    obsidian_digest = build_obsidian_digest(message, obsidian_hits) if obsidian_hits else {"digest": "", "title": "", "source_count": "0"}
    t_obsidian_digest = time.time() - t0

    # ===== Web context collection
    t0 = time.time()
    web_hits = collect_web_context(message) if use_web and _should_search_web(message) else []
    t_web_search = time.time() - t0

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

        # ===== Gemini API call
        t0 = time.time()
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
                    },
                    "required": ["reply", "should_save", "save_title", "save_body", "save_reason", "improvement_items", "web_used", "web_reason"],
                },
                http_options=types.HttpOptions(timeout=max(10000, int(timeout_seconds * 1000))),
            ),
        )
        t_gemini = time.time() - t0

        parsed = _extract_response_json(response)
        if not parsed:
            parsed = _fallback_chat_packet(message, score_result, obsidian_hits, web_hits, humor_style)

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
                decision = str(item.get("decision") or "review").strip().lower()
                decision_reason = str(item.get("decision_reason") or "").strip()
                next_action = str(item.get("next_action") or "").strip()
                evidence = str(item.get("evidence") or "").strip()
                if not (title or user_need or suggestion or evidence):
                    continue
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
        # Web/Wiki/Weekly saves are now handled by the simplified 8-field schema
        # If needed in the future, these can be re-enabled by restoring the fields

        # ===== Phase 1 & 3: Log latency metrics with RAG & cache info
        t_total = time.time() - t_start

        # RAG メトリクスの取得
        rag_metrics_str = ""
        if RAG_ENABLED:
            try:
                rag_report = get_rag_performance_metrics()
                rag_latency = rag_report.get("metrics", {}).get("latency", {})
                rag_metrics_str = (
                    f" | RAG[p95={rag_latency.get('p95_ms', 0):.1f}ms, "
                    f"cache_hit_rate={rag_report.get('metrics', {}).get('cache_stats', {}).get('hit_rate', 0):.1f}%]"
                )
            except Exception:
                pass

        logger.info(
            f"PHASE1-3_LATENCY | "
            f"obsidian_search={t_obsidian_search:.3f}s | "
            f"obsidian_digest={t_obsidian_digest:.3f}s | "
            f"web_search={t_web_search:.3f}s | "
            f"gemini={t_gemini:.3f}s | "
            f"total={t_total:.3f}s | "
            f"query_length={len(message)} | "
            f"hits={len(obsidian_hits)} | "
            f"cache_status={from_cache}{rag_metrics_str}"
        )

        # ===== Phase 1 Step 2: Log cache statistics periodically
        cache = get_cache()
        stats = cache.get_stats()
        if stats["total_requests"] % 10 == 0:  # Log every 10 requests
            log_cache_stats(logger)
            if RAG_ENABLED:
                try:
                    rag_report = get_rag_performance_metrics()
                    logger.info(f"RAG メトリクス: {rag_report}")
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
            "obsidian_digest": obsidian_digest,
            "web_hits": web_hits,
            "obsidian_hits": obsidian_hits,
            "llm_model": model,
        }
    except Exception as exc:
        # ===== Phase 1: Log error with latency
        t_error = time.time() - t_start
        logger.error(
            f"PHASE1_ERROR | "
            f"error={str(exc)} | "
            f"elapsed={t_error:.3f}s | "
            f"obsidian_search={t_obsidian_search:.3f}s | "
            f"web_search={t_web_search:.3f}s"
        )
        return {
            "reply": f"AIチャットでエラーが発生しました: {exc}",
            "saved": False,
            "save_result": {"status": "error", "reason": str(exc)},
            "obsidian_digest": obsidian_digest if 'obsidian_digest' in locals() else {},
            "web_hits": web_hits if 'web_hits' in locals() else [],
            "obsidian_hits": obsidian_hits if 'obsidian_hits' in locals() else [],
        }
