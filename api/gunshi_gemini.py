"""軍師AI — Gemini streamGenerateContent SSE版"""
import json
import httpx
from shinsa_gunshi_logic import (
    EVIDENCE_WEIGHTS,
    compute_prior,
    compute_posterior,
    select_top_phrases,
    PHRASES_100,
)

GEMINI_STREAM_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:streamGenerateContent"
)


def _normalize_resale_eval(value: object) -> str:
    """Normalize frontend asset grades to the labels expected by the Bayes engine."""
    text = str(value or "").strip()
    mapping = {
        "A": "高",
        "B": "中",
        "C": "低",
        "S": "高",
        "高": "高",
        "中": "中",
        "標準": "中",
        "低": "低",
    }
    return mapping.get(text.upper(), mapping.get(text, "中"))


def _bayes_inputs(params: dict) -> dict:
    return {
        "resale": _normalize_resale_eval(params.get("resale_eval", "中")),
        "repeat_cnt": int(params.get("repeat_count", 0) or 0),
        "subsidy": bool(params.get("subsidy_flag", False)),
        "bank": bool(params.get("bank_support", False)),
        "intuition": int(float(params.get("intuition_score", 50) or 50) // 20),
    }


def build_bayes_factors(params: dict, prior: float, posterior: float) -> list[dict]:
    """Return human-readable factors behind the posterior probability."""
    inputs = _bayes_inputs(params)
    score = float(params.get("score", 0) or 0)
    pd_pct = float(params.get("pd_pct", 0) or 0)

    factors = [
        {
            "label": "事前確率",
            "detail": f"スコア {score:.1f} 点 × (1 - PD {pd_pct:.1f}%) を初期値に使用",
            "delta_pct": 0.0,
            "direction": "base",
        }
    ]

    resale = inputs["resale"]
    resale_key = "resale_high" if resale == "高" else "resale_mid" if resale == "中" else "resale_low"
    resale_delta = float(EVIDENCE_WEIGHTS.get(resale_key, 0.0))
    factors.append({
        "label": f"物件リセール: {resale}",
        "detail": "換価性が高いほど回収余力を上げ、低い場合は保全懸念として減点",
        "delta_pct": round(resale_delta * 100, 1),
        "direction": "up" if resale_delta > 0 else "down" if resale_delta < 0 else "flat",
    })

    repeat_delta = min(float(EVIDENCE_WEIGHTS.get("repeat", 0.0)) * inputs["repeat_cnt"], 0.20)
    factors.append({
        "label": "既存・再リース実績",
        "detail": f"{inputs['repeat_cnt']} 件を反映。実績が多いほど支払継続性を評価",
        "delta_pct": round(repeat_delta * 100, 1),
        "direction": "up" if repeat_delta > 0 else "flat",
    })

    subsidy_delta = float(EVIDENCE_WEIGHTS.get("subsidy", 0.0)) if inputs["subsidy"] else 0.0
    factors.append({
        "label": "補助金・助成金文脈",
        "detail": "投資目的や資金繰り補完材料として確認済み" if inputs["subsidy"] else "未確認。採択・対象要件が取れれば押し材料",
        "delta_pct": round(subsidy_delta * 100, 1),
        "direction": "up" if subsidy_delta > 0 else "flat",
    })

    bank_delta = float(EVIDENCE_WEIGHTS.get("bank", 0.0)) if inputs["bank"] else 0.0
    factors.append({
        "label": "銀行支援・銀行紹介",
        "detail": "主取引銀行の導線を回収・モニタリング材料として評価" if inputs["bank"] else "未確認。銀行温度感が取れないため上振れなし",
        "delta_pct": round(bank_delta * 100, 1),
        "direction": "up" if bank_delta > 0 else "flat",
    })

    intuition_delta = float(EVIDENCE_WEIGHTS.get("intuition", 0.0)) * (inputs["intuition"] - 3)
    factors.append({
        "label": "担当者直感",
        "detail": f"入力値を5段階換算して {inputs['intuition']}。基準3との差を反映",
        "delta_pct": round(intuition_delta * 100, 1),
        "direction": "up" if intuition_delta > 0 else "down" if intuition_delta < 0 else "flat",
    })

    raw_delta = posterior - prior
    factors.append({
        "label": "更新後の差分",
        "detail": f"事前 {prior:.1%} から事後 {posterior:.1%} へ更新",
        "delta_pct": round(raw_delta * 100, 1),
        "direction": "up" if raw_delta > 0 else "down" if raw_delta < 0 else "flat",
    })
    return factors


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
    bayes_inputs = _bayes_inputs(params)
    posterior = compute_posterior(
        prior=prior,
        **bayes_inputs,
    )
    phrases = select_top_phrases(
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
    bayes_inputs = _bayes_inputs(params)
    posterior = compute_posterior(
        prior=prior,
        **bayes_inputs,
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


def build_strategy_cards(params: dict, phrases: list[str], prior: float, posterior: float) -> dict:
    """Build deterministic strategy cards consumed by the Next.js Gunshi panel."""
    score = float(params.get("score", 0) or 0)
    industry_cat = str(params.get("industry_cat") or "業種未設定")
    industry_sub = str(params.get("industry_sub") or "")
    asset_name = str(params.get("asset_name") or "対象物件")
    company_name = str(params.get("company_name") or "本件")
    acquisition_cost = float(params.get("acquisition_cost", 0) or 0)
    lease_term = int(float(params.get("lease_term", 0) or 0))
    contract_type = str(params.get("contract_type") or "")
    competitor = str(params.get("competitor") or "")
    competitor_rate = params.get("competitor_rate")
    bank_support = bool(params.get("bank_support"))
    subsidy_flag = bool(params.get("subsidy_flag"))
    bayes_inputs = _bayes_inputs(params)
    repeat_count = bayes_inputs["repeat_cnt"]
    equity_ratio = float(params.get("equity_ratio", 0) or 0)
    op_profit = float(params.get("op_profit", 0) or 0)
    nenshu = float(params.get("nenshu", 0) or 0)
    op_margin = op_profit / nenshu * 100 if nenshu else None

    if score >= 70:
        stance = "承認寄せ"
        headline = "承認前提で、審査部が確認したい穴だけ先に塞ぐ"
        risk_intro = "条件を増やしすぎると商談速度が落ちる"
    elif score >= 50:
        stance = "条件付き承認"
        headline = "否決理由を先回りし、条件付き承認へ着地させる"
        risk_intro = "財務・保全・競合条件の説明不足で止まりやすい"
    else:
        stance = "再審議準備"
        headline = "無理押しせず、損失限定策を作って再審議へ回す"
        risk_intro = "現状のままでは返済原資と保全の説明が弱い"

    facts = [
        f"案件: {company_name}",
        f"業種: {industry_cat}{(' / ' + industry_sub) if industry_sub else ''}",
        f"物件: {asset_name}",
        f"スコア: {score:.1f}点",
        f"ベイズ推定: {prior:.1%} → {posterior:.1%}",
    ]
    if acquisition_cost > 0:
        facts.append(f"取得価額: {acquisition_cost:,.0f}")
    if lease_term > 0:
        facts.append(f"期間: {lease_term}か月")
    if contract_type:
        facts.append(f"契約: {contract_type}")

    risks = [risk_intro]
    if op_margin is not None:
        if op_margin < 0:
            risks.append(f"営業利益率 {op_margin:.1f}% は赤字。返済原資の説明を必ず補強")
        elif op_margin < 2:
            risks.append(f"営業利益率 {op_margin:.1f}% は薄い。利益改善要因を確認")
    if equity_ratio < 0:
        risks.append(f"自己資本比率 {equity_ratio:.1f}% は債務超過。保証・保全・返済計画が必須")
    elif equity_ratio < 10:
        risks.append(f"自己資本比率 {equity_ratio:.1f}% は低位。短期資金繰りを確認")
    if not bank_support:
        risks.append("銀行支援・紹介の裏取りが弱い場合、主取引銀行の温度感を確認")

    today_moves = [
        "返済原資を、直近実績・受注見込み・費用削減効果の順に資料化する",
        "物件の必要性と換価性を、見積・カタログ・中古相場で補強する",
        "審査条件を先に1つ提示し、審査部の懸念を交渉材料へ変える",
    ]
    if score < 50:
        today_moves[0] = "金額圧縮・頭金・保証追加のどれで損失限定できるか先に決める"
    if subsidy_flag:
        today_moves.append("補助金は採択前提にせず、対象要件・期限・資金繰り影響を確認する")

    competitor_moves = []
    if competitor:
        competitor_moves.append(f"競合「{competitor}」の金利だけでなく、満了条件・保守・手続負担を比較")
    else:
        competitor_moves.append("競合見積の有無を確認し、金利以外の比較軸を作る")
    if competitor_rate not in (None, "", 0):
        competitor_moves.append(f"競合金利 {float(competitor_rate):.2f}% に対し、保守・満了・審査速度で差別化")
    competitor_moves.append("主取引銀行が絡む場合は、銀行側の本気度と当社の役割を明確化")

    questions = [
        "今回の設備で売上・粗利・人件費はどれだけ変わるか",
        "既存設備の入替か増設か。旧設備の処分・下取り予定はあるか",
        "主取引銀行は今回投資をどう見ているか",
    ]
    if repeat_count > 0:
        questions.append(f"既存リース・再リース実績 {repeat_count} 件の支払状況に問題はないか")

    phrase_lines = [p for p in phrases[:2] if p]
    customer_lines = [
        "審査で見られるのは、設備そのものより投資後に返済原資がどう増えるかです。",
        "金利だけでなく、導入後の手間と満了時の出口まで含めて比較しましょう。",
    ]
    ringi_lines = [
        f"{asset_name} は事業継続・収益改善に直結する投資であり、条件設定により回収懸念を限定できる。",
        "返済原資、物件保全、取引継続性の3点を確認済みとして稟議化する。",
    ]

    badges = [stance, f"Score {score:.0f}", f"Bayes {posterior:.0%}"]
    if subsidy_flag:
        badges.append("補助金確認")
    if bank_support:
        badges.append("銀行導線あり")

    return {
        "headline": headline,
        "stance": stance,
        "case_facts": facts[:7],
        "risk_cards": risks[:4],
        "today_moves": today_moves[:4],
        "competitor_moves": competitor_moves[:3],
        "questions_to_ask": questions[:4],
        "customer_one_liners": customer_lines,
        "ringi_lines": phrase_lines + ringi_lines,
        "badges": badges,
        "bayes_factors": build_bayes_factors(params, prior, posterior),
        "disclaimer": "軍師AIは判定を上書きしません。最終判断は審査ルール、スコア、担当者確認に従ってください。",
    }


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
    bayes_inputs = _bayes_inputs(params)
    posterior = compute_posterior(
        prior=prior,
        **bayes_inputs,
    )
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
    # phrase_dicts は list[dict] — text フィールドを抽出
    phrases = [p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in phrase_dicts]

    yield {
        "type": "bayes",
        "prior": round(prior, 4),
        "posterior": round(posterior, 4),
        "factors": build_bayes_factors(params, prior, posterior),
    }
    yield {"type": "phrases", "items": phrases}
    yield {
        "type": "strategy_cards",
        "cards": build_strategy_cards(params, phrases, prior, posterior),
    }

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
