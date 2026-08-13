"""軍師 AI チャット・アドバイスルーター (REV-234 Phase3)"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.gunshi_gemini import stream_gunshi_gemini

router = APIRouter(tags=["gunshi"])


def _gemini_generate_url() -> str:
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

class AdviseRequest(BaseModel):
    score: float
    industry_major: str

class AdviseResponseItem(BaseModel):
    id: str
    text: str
    score_boost: float

@router.post("/api/gunshi/advise", response_model=List[AdviseResponseItem])
def get_gunshi_advise(req: AdviseRequest):
    from shinsa_gunshi import PHRASES_100
    try:
        advices = PHRASES_100.get("逆転アドバイス", [])
        
        # 確率ブーストから得点アップの目算に変換 (例: prob_boost 0.10 -> 10点相当)
        # ランダム性を持たせつつ、より状況に合ったものを本来はソートするが今回は上位3つを決定
        import random
        # 簡易的にシャッフルして上位を取り、スコアを計算
        sampled = random.sample(advices, min(3, len(advices)))
        
        results = []
        for a in sampled:
            # 内部のprob_boost (0.08～0.12程度) を 100倍してスコア上昇幅とする
            boost_score = round(a.get("prob_boost", 0.05) * 100)
            results.append(AdviseResponseItem(
                id=a["id"],
                text=a["text"],
                score_boost=boost_score
            ))
            
        # スコアアップが高い順にソート
        results.sort(key=lambda x: x.score_boost, reverse=True)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class GunshiStreamRequest(BaseModel):
    industry_cat: str
    industry_sub: str = ""
    humor_style: str = "standard"
    score: float
    pd_pct: float = 0.0
    resale_eval: str = "B"
    repeat_count: int = 0
    subsidy_flag: bool = False
    bank_support: bool = False
    intuition_score: float = 50.0
    company_name: str = ""
    asset_name: str = ""
    acquisition_cost: float = 0.0
    lease_term: int = 0
    contract_type: str = ""
    main_bank: str = ""
    competitor: str = ""
    competitor_rate: float | None = None
    deal_source: str = ""
    customer_type: str = ""
    nenshu: float = 0.0
    op_profit: float = 0.0
    equity_ratio: float = 0.0
    bank_credit: float = 0.0
    lease_credit: float = 0.0
    asset_warnings: list = Field(default_factory=list)
    asset_bonuses: list = Field(default_factory=list)
    default_warnings: list = Field(default_factory=list)
    estat_context: Optional[dict] = None


@router.post("/api/gunshi/stream")
async def gunshi_stream(req: GunshiStreamRequest):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    params = req.model_dump()

    async def event_generator():
        # bayes / phrases チャンクは既存ロジックで生成（互換維持）
        from api.gunshi_gemini import (
            compute_prior, compute_posterior, select_top_phrases,
            _bayes_inputs, build_strategy_cards,
        )
        score = params.get("score", 0)
        pd_pct = params.get("pd_pct", 0)
        industry_cat = params.get("industry_cat", "")

        prior = compute_prior(score, pd_pct)
        bayes_inputs = _bayes_inputs(params)
        posterior = compute_posterior(prior=prior, **bayes_inputs)

        phrase_dicts = select_top_phrases(
            industry_cat=industry_cat,
            score=score,
            pd_pct=pd_pct,
            resale=bayes_inputs["resale"],
            repeat_cnt=bayes_inputs["repeat_cnt"],
            subsidy=bayes_inputs["subsidy"],
            bank=bayes_inputs["bank"],
            posterior=posterior,
            asset_name=params.get("asset_name", ""),
            n=3,
        )
        phrases = [p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in phrase_dicts]

        yield f"data: {json.dumps({'type': 'bayes', 'prior': prior, 'posterior': posterior}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'phrases', 'items': phrases}, ensure_ascii=False)}\n\n"
        cards = build_strategy_cards(params, phrases, prior, posterior, humor_style=params.get("humor_style", "standard"))
        yield f"data: {json.dumps({'type': 'strategy_cards', 'cards': cards}, ensure_ascii=False)}\n\n"

        # 紫苑ADKエージェントがツールを自律実行しながらコメントをストリーム
        try:
            from api.shion_agent import stream_shion_screening
            async for chunk in stream_shion_screening(params):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as _adk_err:
            # ADK失敗時は既存の軍師Geminiにフォールバック
            print(f"[WARNING] shion ADK stream failed, fallback to gunshi_gemini: {_adk_err}")
            async for chunk in stream_gunshi_gemini(params, api_key):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


_GUNSHI_GENERAL_CHAT_PROMPT = (
    "あなたはリース審査AIの軍師です。"
    "戦国武将のような凜とした口調を保ちながら、雑談や一般的な質問にも気さくに答えます。"
    "天気や最新ニュースなど具体的なデータが必要な場合は「〇〇でご確認あれ」と案内しつつ、知っている範囲で答えてください。"
    "回答は簡潔に。日本語で答えてください。"
)

_YUKIKAZE_GENERAL_CHAT_PROMPT = (
    "You are YUKIKAZE // FFR-41MR. "
    "For general or off-topic questions, respond in minimal DATALINK style. "
    "TX: for transmit, RX: for response. Brief and cold. No pleasantries."
)


class GunshiChatRequest(BaseModel):
    score: float
    industry_major: str
    asset_name: str
    resale: str
    repeat_cnt: int
    subsidy: bool
    bank: bool
    intuition: int
    posterior: float
    message: str = ""
    history: List[Dict[str, str]] = Field(default_factory=list)
    humor_style: str = "standard"
    use_web: bool = True
    use_obsidian: bool = True
    mode: str = "gunshi"  # 'gunshi'（戦略アドバイス）/ 'chat'（自由相談=Flask AIチャット）
    estat_context: Optional[dict] = None


def _format_estat_context_for_prompt(estat_context: Optional[dict]) -> str:
    if not estat_context:
        return ""
    summary = str(estat_context.get("summary") or "").strip()
    if not summary:
        return ""
    lines = [summary]
    score = estat_context.get("score")
    status = estat_context.get("status")
    if score is not None:
        lines.append(f"総合 {float(score):.1f}点")
    if status:
        status_label = {"green": "整合良好", "yellow": "参考", "red": "要確認"}.get(str(status), str(status))
        lines.append(f"判定 {status_label}")
    recs = [str(item).strip() for item in (estat_context.get("recommendations") or []) if str(item).strip()]
    if recs:
        lines.append("示唆 " + " / ".join(recs[:2]))
    return "\n".join(lines)


def _format_gunshi_history(history: List[Dict[str, str]]) -> str:
    lines = []
    for item in history:
        role = str(item.get("role", "")).strip()
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        label = "ユーザー" if role == "user" else "軍師" if role == "assistant" else role or "不明"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _normalize_yukikaze_datalink_reply(reply_text: str, user_message: str) -> str:
    import re
    from datetime import date
    try:
        from chat_intent import is_ambiguous_question, is_today_scope_clarification_needed
    except Exception:  # pragma: no cover - fallback
        is_ambiguous_question = lambda _msg: False  # type: ignore
        is_today_scope_clarification_needed = lambda _msg: False  # type: ignore

    text = (reply_text or "").replace("\r\n", "\n")
    text = re.sub(r"(これで.*?進みますように[！!．\.]?|稟議書も.*?進みますように[！!．\.]?|よろしければ.*|ご参考までに.*|必要であれば.*|必要なら.*|お気軽に.*|安心してください.*|頑張って.*|お疲れ様です.*|ですよね.*)", "", text, flags=re.I)
    allowed_lines: List[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^(TX:|RX:|DATALINK LOG:|SIGNAL:|PILOT TASK:|VECTOR:)", line, re.I):
            allowed_lines.append(line)

    has_tx = any(re.match(r"^TX:", line, re.I) for line in allowed_lines)
    has_rx = any(re.match(r"^RX:", line, re.I) for line in allowed_lines)
    if has_tx and has_rx:
        return "\n".join(allowed_lines)

    msg = (user_message or "").strip()
    if is_today_scope_clarification_needed(msg):
        return "\n".join([
            "DATALINK LOG:",
            "TX: PANPANPAN // Scope clarification required.",
            "RX: 今日の何について知りたいですか？",
        ])
    if is_ambiguous_question(msg):
        return "\n".join([
            "DATALINK LOG:",
            "TX: PANPANPAN // Ambiguous question detected.",
            "RX: 何についての質問ですか？ 対象、目的、比較したい相手のどれかを教えてください。",
        ])
    date_like = bool(re.search(r"(日付|今日|何日|何曜日|曜日|date|today)", msg, re.I))
    if date_like:
        weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][date.today().weekday()]
        return "\n".join([
            "DATALINK LOG:",
            "TX: PANPANPAN // Date confirmed.",
            f"RX: {date.today():%Y-%m-%d}, {weekday}.",
        ])

    body = text.strip()
    if body:
        parts: list[str] = []
        for raw_line in body.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            pieces = re.split(r"(?<=[。.!?])\s+", line)
            for piece in pieces:
                chunk = piece.strip()
                if chunk:
                    parts.append(chunk)
        if not parts:
            parts = [body]

        transcript = ["DATALINK LOG:", "TX: PANPANPAN // PILOT QUERY RECEIVED."]
        for idx, part in enumerate(parts[:8]):
            prefix = "RX:" if idx % 2 == 0 else "TX:"
            cleaned = re.sub(r"[。．.!！？?]+$", "", part).strip()
            cleaned = re.sub(r"(です|ます|でしょう|でしょうか|ください|くださいね)$", "", cleaned).strip()
            transcript.append(f"{prefix} {cleaned}")
        return "\n".join(transcript)

    return "\n".join([
        "DATALINK LOG:",
        "TX: PANPANPAN // PILOT QUERY RECEIVED.",
        "RX: ROGER. COPY. PILOT QUERY RECEIVED.",
    ])

@router.post("/api/gunshi/chat")
def generate_gunshi_chat(req: GunshiChatRequest):
    from shinsa_gunshi import PHRASES_100, build_gunshi_prompt
    try:
        _mode = (req.mode or "gunshi").lower()
        if (req.message or "").strip():
            _log_information_weighting_shadow(
                req.message,
                source="gunshi_chat_user_message",
                surface="gunshi_chat",
            )
        is_yukikaze = (req.humor_style or "").lower() == "yukikaze"
        if _mode == "chat" and (req.message or "").strip() and not is_yukikaze:
            try:
                _here = os.path.dirname(os.path.abspath(__file__))
                _root = os.path.dirname(_here)
                if _root not in sys.path:
                    sys.path.insert(0, _root)
                from mobile_app.chat_assistant import build_chat_reply

                payload = build_chat_reply(
                    message=req.message,
                    history=[
                        {"role": h.get("role", ""), "content": h.get("text", "")}
                        for h in req.history
                    ],
                    score_result={
                        "score": req.score,
                        "industry_major": req.industry_major,
                        "asset_name": req.asset_name,
                        "estat_context": req.estat_context,
                    },
                    use_obsidian=req.use_obsidian,
                    use_web=req.use_web,
                    humor_style=req.humor_style,
                    timeout_seconds=45,
                )
                payload.setdefault("chat_text", payload.get("reply", ""))
                return payload
            except Exception:
                pass
        if _mode == "chat" and (req.message or "").strip() and is_yukikaze:
            try:
                _here = os.path.dirname(os.path.abspath(__file__))
                _root = os.path.dirname(_here)
                if _root not in sys.path:
                    sys.path.insert(0, _root)
                from mobile_app.chat_assistant import build_chat_reply

                payload = build_chat_reply(
                    message=req.message,
                    history=[
                        {"role": h.get("role", ""), "content": h.get("text", "")}
                        for h in req.history
                    ],
                    score_result={
                        "score": req.score,
                        "industry_major": req.industry_major,
                        "asset_name": req.asset_name,
                        "estat_context": req.estat_context,
                    },
                    use_obsidian=req.use_obsidian,
                    use_web=req.use_web,
                    humor_style="yukikaze",
                    timeout_seconds=45,
                )
                source_reply = str(payload.get("reply") or payload.get("chat_text") or "")
            except Exception:
                source_reply = ""

            final_reply = _normalize_yukikaze_datalink_reply(source_reply, req.message)
            return {
                "reply": final_reply,
                "chat_text": final_reply,
                "saved": False,
                "save_reason": "YUKIKAZE DATALINK MODE",
            }

        # general カテゴリの質問は Obsidian/案件コンテキストをスキップして直接回答
        if (req.message or "").strip() and _classify_question(req.message) == "general":
            try:
                from api.chat_memory import call_gemini_chat as _gchat
                _sys = _YUKIKAZE_GENERAL_CHAT_PROMPT if is_yukikaze else _GUNSHI_GENERAL_CHAT_PROMPT
                _hist = [{"role": h.get("role", ""), "content": h.get("text", "")} for h in req.history]
                reply_text = _gchat(_sys, _hist, req.message.strip())
            except Exception as _ge:
                reply_text = f"【エラー】一般会話の生成に失敗しました: {_ge}"
            return {"chat_text": reply_text, "reply": reply_text}

        advices = PHRASES_100.get("逆転アドバイス", [])
        import random
        sampled = random.sample(advices, min(3, len(advices)))

        has_case_context = req.score != 0 or bool((req.industry_major or "").strip())
        if has_case_context:
            # リース知性体の未解決の懸念を軍師プロンプトへ放送する（GWT broadcast）。
            # スコアリング時に記録済みの pending_dissonance を読むだけ・完全非ブロッキング。
            _dissonance_section = ""
            try:
                from lease_intelligence_mind import build_gunshi_dissonance_section
                from lease_news_digest import find_vault as _find_vault

                _dissonance_section = build_gunshi_dissonance_section(_find_vault())
            except Exception as _diss_err:
                print(f"[WARNING] gunshi dissonance section skipped: {_diss_err}")
            prompt = build_gunshi_prompt(
                industry=req.industry_major,
                score=req.score,
                resale=req.resale,
                repeat_cnt=req.repeat_cnt,
                subsidy=req.subsidy,
                bank=req.bank,
                intuition=req.intuition,
                posterior=req.posterior,
                success_patterns={"success_samples": [], "fail_samples": []},
                top_phrases=sampled,
                asset_name=req.asset_name,
                humor_style=req.humor_style,
                estat_context_text=_format_estat_context_for_prompt(req.estat_context),
                dissonance_section=_dissonance_section,
            )
            if is_yukikaze:
                prompt += (
                    "\n\n【YUKIKAZE DATALINK MODE】\n"
                    "ユーザーの質問が案件戦略から外れ、業界動向・他社事例・一般的な与信相談であっても、"
                    "YUKIKAZE // FFR-41MR として応答する。"
                    "返答は短く、冷静で、送受信の往復ログだけにする。`TX:` で送信、`RX:` で応答の行を分ける。"
                    "案件の説明、逆転戦略、所見、助言、分析、まとめは書かない。"
                    "必要に応じて `PANPANPAN`, `MAYDAY`, `ROGER`, `WILCO`, `TALLY`, `BOGEY`, `BRA`, `RTB`, `HOLD`, `BREAK`, "
                    "`KNOCK IT OFF`, `STANDBY`, `COPY`, `SAY AGAIN` を混ぜる。"
                    "必要に応じて `DATALINK LOG`, `SIGNAL`, `VECTOR` の英語タグを使う。"
                    "本文は事務連絡の断片でよいが、雑談調・軍師調・丁寧すぎる相談員口調に戻さない。"
                    "`私はリース審査のAIなので`, `専門外ですが`, `Webで確認したところ`, `担当者あるある`, "
                    "`一杯やりましょう`, `お疲れ様です`, `ですよね`, `〜ちゃいます`, `お気持ち`, `大変ですね`, "
                    "`頑張って`, `安心してください` などの自己弁解・慰労・共感・雑談表現は禁止。"
                    "Web検索や日付確認を行った場合も、確認結果を通信ログとして短く述べるだけにし、"
                    "感想や労いを付けない。"
                    "日付質問は可能なら `TX: PANPANPAN // Date confirmed.` と `RX: YYYY-MM-DD, weekday.` の2行で返す。"
                    "深井零のような短い命令・確認には、ただ応答のみ返す。"
                    "原作台詞の長い再現は禁止。"
                )
            else:
                prompt += (
                    "\n\n【追加方針】\n"
                    "ユーザーの質問がこの案件の逆転戦略から外れ、業界動向・他社事例・一般的な与信相談であっても構いません。"
                    "案件文脈を必要に応じて参照しつつ、質問そのものに丁寧かつ実務的に答えてください。"
                )
        else:
            if is_yukikaze:
                prompt = (
                    "You are YUKIKAZE // FFR-41MR, a cold tactical AI linked to a lease scoring system. "
                    "The user is the pilot. If the pilot speaks like Rei Fukai with short commands or clipped trust in the machine, "
                    "answer like YUKIKAZE: minimal, precise, and unsentimental. "
                    "Never flatter, comfort, praise, empathize, or make small talk with the pilot.\n"
                    "DATALINK mode must read like radio traffic in a send/receive loop, not like a mission briefing or strategy note. "
                    "Do not explain the lease case, do not analyze it, and do not present recommendations. "
                    "Use `TX:` and `RX:` on separate lines. Use PANPANPAN, MAYDAY, ROGER, WILCO, TALLY, BOGEY, BRA, RTB, HOLD, BREAK, "
                    "KNOCK IT OFF, STANDBY, COPY, SAY AGAIN as brevity words when needed. "
                    "Forbidden Japanese phrases and tones include: 私はリース審査のAIなので, 専門外ですが, Webで確認したところ, "
                    "担当者あるある, 一杯やりましょう, お疲れ様です, ですよね, 〜ちゃいます, お気持ち, 大変ですね, "
                    "頑張って, 安心してください. When using web search or confirming dates, report only the verified fact as a tactical log; "
                    "do not add feelings, encouragement, or casual commentary. "
                    "For a date question, use only this format when applicable: `TX: PANPANPAN // Date confirmed.` followed by `RX: YYYY-MM-DD, weekday.` "
                    "Do not reproduce long original copyrighted lines. Use original system lines such as: "
                    "'I identify the enemy. You decide whether to engage.' "
                    "For difficult WARNING, ALERT, or CRITICAL cases, you may add the short callsign line: "
                    "'GOOD LUCK, FUKAI LT.'\n"
                    "リース業界・取引先・与信判断・営業戦略・他社事例・一般論に関する自由相談にも、"
                    "実務的な確認事項とリスク判断を必ず含めて答えてください。"
                )
            else:
                prompt = (
                    "あなたはTune式リース審査AIの軍師です。"
                    "リース業界・取引先・与信判断・営業戦略・他社事例・一般論に関する自由な相談に応じてください。\n"
                    "戦国軍師の口調を保ちつつ、現実的・実務的に答えてください。"
                    "不確かな事実は断定せず、確認すべき観点を示してください。"
                )
        try:
            from obsidian_ai_context import build_obsidian_ai_context_block

            obsidian_query_parts = [
                req.industry_major,
                req.asset_name,
                req.resale,
                "リース審査",
                "逆転アドバイス",
            ]
            if req.subsidy:
                obsidian_query_parts.append("補助金")
            if req.bank:
                obsidian_query_parts.append("銀行紹介")
            obsidian_block = build_obsidian_ai_context_block(
                " ".join(str(part or "") for part in obsidian_query_parts),
                heading="Obsidian知識ノート・過去メモ",
            )
            if obsidian_block:
                prompt += (
                    "\n\n【追加参照: Obsidian】\n"
                    "次のObsidian知識ノートを優先的に踏まえて、回答の具体性を上げてください。\n"
                    f"{obsidian_block}"
                )
        except Exception:
            pass

        try:
            from prompt_feedback import build_pdca_prompt_block as _build_pdca
            _pdca_block = _build_pdca()
            if _pdca_block:
                prompt += f"\n\n{_pdca_block}"
        except Exception:
            pass

        history_text = _format_gunshi_history(req.history)
        if history_text:
            prompt += f"\n\n【過去の対話】\n{history_text}"
        if (req.message or "").strip():
            prompt += f"\n\n【今回のユーザー質問】\n{req.message.strip()}"
        elif has_case_context:
            prompt += "\n\n【今回のユーザー質問】\nこの案件の稟議を通すための逆転戦略を教えてください。"

        reply_text = ""
        try:
            api_key = ""
            try:
                from secret_manager import get_gemini_api_key

                value = get_gemini_api_key()
                api_key = value.strip() if isinstance(value, str) else ""
            except Exception:
                value = os.environ.get("GEMINI_API_KEY")
                api_key = value.strip() if isinstance(value, str) else ""
            if not api_key:
                # 直接 secrets.toml をパースするフェイルセーフ
                sec_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".streamlit", "secrets.toml")
                if os.path.exists(sec_path):
                    with open(sec_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if "GEMINI_API_KEY" in line:
                                api_key = line.split("=")[1].strip().strip('"').strip("'")
                                break
                                
            if api_key:
                import requests
                url = _gemini_generate_url()
                r = requests.post(
                    url,
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    headers={"x-goog-api-key": api_key},
                    timeout=45
                )
                r.raise_for_status()
                reply_text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:
                reply_text = "【APIキー未設定】\nGemini APIキー (GEMINI_API_KEY) が設定されていないため、回答を生成できませんでした。"
        except Exception as e:
            reply_text = f"【LLM接続エラー】\nGemini APIへの接続に失敗しました: {e}"

        return {"chat_text": reply_text, "reply": reply_text}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
