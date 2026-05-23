"""軍師AI — Gemini streamGenerateContent SSE版"""
import json
import httpx
from shinsa_gunshi_logic import (
    compute_prior,
    compute_posterior,
    select_top_phrases,
    PHRASES_100,
)

GEMINI_STREAM_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:streamGenerateContent"
)


def build_system_instruction() -> str:
    # PHRASES_100 は dict[str, list[dict]] なので全カテゴリのtextを列挙
    all_phrases = []
    for phrases in PHRASES_100.values():
        for p in phrases:
            text = p.get("text", "") if isinstance(p, dict) else str(p)
            if text:
                all_phrases.append(text)
    phrases_text = "\n".join(f"- {p}" for p in all_phrases)
    return (
        "あなたは「軍師AI」です。リース審査担当者に対し、"
        "承認奪取のための戦略・フレーズを武将スタイルで提案してください。\n"
        f"利用可能なフレーズ辞書（{len(all_phrases)}件）:\n{phrases_text}\n"
    )


def build_user_prompt(params: dict) -> str:
    score = params.get("score", 0)
    pd_pct = params.get("pd_pct", 0)
    industry_cat = params.get("industry_cat", "")
    prior = compute_prior(score, pd_pct)
    posterior = compute_posterior(
        prior=prior,
        resale=params.get("resale_eval", "B"),
        repeat_cnt=params.get("repeat_count", 0),
        subsidy=params.get("subsidy_flag", False),
        bank=params.get("bank_support", False),
        intuition=int(params.get("intuition_score", 50) // 20),  # 0-100 → 0-5 相当に変換
    )
    phrases = select_top_phrases(
        industry_cat=industry_cat,
        score=score,
        pd_pct=pd_pct,
        resale=params.get("resale_eval", "B"),
        repeat_cnt=params.get("repeat_count", 0),
        subsidy=params.get("subsidy_flag", False),
        bank=params.get("bank_support", False),
        posterior=posterior,
        asset_name=params.get("asset_name", ""),
        n=3,
    )
    return (
        f"案件: 業種={industry_cat} "
        f"スコア={score} "
        f"ベイズ推定: {prior:.1%} → {posterior:.1%}\n"
        f"推奨フレーズ(top3): {phrases}\n"
        "この案件の承認奪取戦略を述べよ。"
    )


def build_fallback_strategy_text(params: dict, phrases: list[str], reason: str = "") -> str:
    """Build a deterministic strategy when Gemini streaming is unavailable."""
    score = float(params.get("score", 0) or 0)
    pd_pct = float(params.get("pd_pct", 0) or 0)
    industry_cat = str(params.get("industry_cat") or "指定なし")
    asset_name = str(params.get("asset_name") or "対象物件")
    company_name = str(params.get("company_name") or "本件")

    prior = compute_prior(score, pd_pct)
    posterior = compute_posterior(
        prior=prior,
        resale=params.get("resale_eval", "B"),
        repeat_cnt=params.get("repeat_count", 0),
        subsidy=params.get("subsidy_flag", False),
        bank=params.get("bank_support", False),
        intuition=int(params.get("intuition_score", 50) // 20),
    )

    if score >= 70:
        stance = "承認方針を前提に、条件を絞って稟議を短く通す局面です。"
        first_move = "過度な追加条件を増やさず、返済原資・物件換価・取引継続性の3点を押さえてください。"
    elif score >= 50:
        stance = "境界案件です。否決懸念を先回りして、条件付き承認へ寄せる局面です。"
        first_move = "保証・頭金・期間短縮・中途解約時の残債管理など、審査部が飲みやすい条件を先に提示してください。"
    else:
        stance = "低スコア案件です。無理押しではなく、損失限定策を明確にして再審議の土台を作る局面です。"
        first_move = "金額圧縮、保証追加、対象物件の換価根拠、既存取引実績の補強をセットで出してください。"

    evidence = []
    if params.get("bank_support"):
        evidence.append("銀行紹介・銀行支援があるため、回収導線とモニタリング体制を稟議の押し材料にできます。")
    if params.get("subsidy_flag"):
        evidence.append("補助金・助成金文脈があるため、投資目的と資金繰り改善効果を明文化してください。")
    if int(params.get("repeat_count", 0) or 0) > 0:
        evidence.append(f"再リース・反復実績が {int(params.get('repeat_count', 0) or 0)} 回あるため、利用継続性を強調できます。")
    if not evidence:
        evidence.append("財務数値だけで押し切らず、物件の必要性・換価性・返済原資の順に補強してください。")

    phrase_lines = [f"- {p}" for p in phrases[:3] if p]
    phrase_block = "\n".join(phrase_lines) if phrase_lines else "- 物件の必要性、換価性、返済原資を一体で説明する。"
    note = f"\n\n※ Gemini接続の代替応答です（{reason}）。" if reason else ""

    return (
        f"【軍師AI 代替戦略】\n"
        f"{company_name}（{industry_cat} / {asset_name}）は、スコア {score:.1f} 点、"
        f"ベイズ推定 {prior:.1%} → {posterior:.1%} です。\n\n"
        f"一、局面判断\n{stance}\n\n"
        f"二、まず取る作戦\n{first_move}\n\n"
        f"三、稟議で押す材料\n"
        + "\n".join(f"- {item}" for item in evidence)
        + "\n\n"
        f"四、使うべき軍師フレーズ\n{phrase_block}"
        f"{note}"
    )


async def stream_gunshi_gemini(params: dict, api_key: str):
    """AsyncGenerator: SSEチャンクを yield する。

    送信順序:
      1. {"type": "bayes", "prior": float, "posterior": float}
      2. {"type": "phrases", "items": list[str]}
      3. {"type": "stream", "delta": str}  (複数回)
      4. {"type": "done"}
    """
    score = params.get("score", 0)
    pd_pct = params.get("pd_pct", 0)
    industry_cat = params.get("industry_cat", "")

    prior = compute_prior(score, pd_pct)
    posterior = compute_posterior(
        prior=prior,
        resale=params.get("resale_eval", "B"),
        repeat_cnt=params.get("repeat_count", 0),
        subsidy=params.get("subsidy_flag", False),
        bank=params.get("bank_support", False),
        intuition=int(params.get("intuition_score", 50) // 20),
    )
    phrase_dicts = select_top_phrases(
        industry_cat=industry_cat,
        score=score,
        pd_pct=pd_pct,
        resale=params.get("resale_eval", "B"),
        repeat_cnt=params.get("repeat_count", 0),
        subsidy=params.get("subsidy_flag", False),
        bank=params.get("bank_support", False),
        posterior=posterior,
        asset_name=params.get("asset_name", ""),
        n=3,
    )
    # phrase_dicts は list[dict] — text フィールドを抽出
    phrases = [p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in phrase_dicts]

    yield {"type": "bayes", "prior": round(prior, 4), "posterior": round(posterior, 4)}
    yield {"type": "phrases", "items": phrases}

    if not api_key:
        yield {
            "type": "stream",
            "delta": build_fallback_strategy_text(params, phrases, "GEMINI_API_KEY未設定"),
        }
        yield {"type": "done"}
        return

    payload = {
        "system_instruction": {"parts": [{"text": build_system_instruction()}]},
        "contents": [
            {"role": "user", "parts": [{"text": build_user_prompt(params)}]}
        ],
        "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.7},
    }
    url = f"{GEMINI_STREAM_URL}?key={api_key}&alt=sse"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    yield {
                        "type": "stream",
                        "delta": f"【Gemini APIエラー (HTTP {resp.status_code})】しばらく待ってから再試行してください。",
                    }
                    yield {"type": "done"}
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                        delta = (
                            chunk["candidates"][0]["content"]["parts"][0].get("text", "")
                        )
                        if delta:
                            yield {"type": "stream", "delta": delta}
                    except Exception:
                        pass
    except Exception as exc:
        yield {
            "type": "stream",
            "delta": build_fallback_strategy_text(params, phrases, type(exc).__name__),
        }

    yield {"type": "done"}
