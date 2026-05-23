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
        f"PD={pd_pct}%\n"
        f"ベイズ推定: {prior:.1%} → {posterior:.1%}\n"
        f"推奨フレーズ(top3): {phrases}\n"
        "この案件の承認奪取戦略を述べよ。"
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
            "delta": f"【Gemini API接続エラー】{type(exc).__name__}: 接続に失敗しました。しばらく待ってから再試行してください。",
        }

    yield {"type": "done"}
