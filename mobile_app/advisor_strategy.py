"""Rule-based strategy advisor for lease screening API.

The advisor is intentionally deterministic. It explains the score result and
suggests next actions, but it never overrides the scoring judgment.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _get_gemini_key() -> str:
    try:
        from secret_manager import get_gemini_api_key

        value = get_gemini_api_key()
        return value.strip() if isinstance(value, str) else ""
    except Exception:
        value = os.environ.get("GEMINI_API_KEY")
        return value.strip() if isinstance(value, str) else ""


def _extract_json_object(text: str) -> dict[str, Any] | None:
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


def _extract_gemini_json(response: Any) -> dict[str, Any] | None:
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

    return _extract_json_object("\n".join(chunks))


def _normalize_llm_advice(raw: dict[str, Any], fallback: dict[str, Any], source: str) -> dict[str, Any]:
    out = dict(fallback)
    for key in ("summary", "strategy", "executive_summary", "humor_note", "disclaimer"):
        if isinstance(raw.get(key), str) and raw[key].strip():
            out[key] = raw[key].strip()
    for key in (
        "risk_points",
        "recommended_conditions",
        "sales_talk",
        "evidence",
        "approval_conditions",
        "positive_factors",
        "concerns_and_responses",
        "counter_offer",
    ):
        vals = raw.get(key)
        if isinstance(vals, list):
            cleaned = [str(v).strip() for v in vals if str(v).strip()]
            if cleaned:
                out[key] = cleaned[:6]
    decision = raw.get("decision")
    if isinstance(decision, dict):
        merged = dict(out.get("decision") or {})
        if decision.get("stance"):
            merged["stance"] = str(decision["stance"])
        if decision.get("confidence") is not None:
            merged["confidence"] = max(0.0, min(1.0, _safe_float(decision["confidence"], merged.get("confidence", 0.0))))
        out["decision"] = merged
    out["source"] = source
    return out


def _build_llm_prompt(fallback: dict[str, Any], score_result: dict[str, Any], case: dict[str, Any], humor_style: str = "standard") -> str:
    payload = {
        "case": case,
        "score_result": score_result,
        "rule_based_advice": fallback,
    }
    return f"""あなたは法人リース審査の「審査軍師AI」です。
Streamlit版の軍師モードと同じく、決裁者と営業担当者に語りかける戦略アドバイザーです。
役割は、審査スコアを上書きせず、「どう通すか」「どこを先に潰すか」を短く具体的に示すことです。

制約:
- 判定は変更しない。最終判断は score_result の score / judgment を尊重する。
- 根拠のない断定は禁止。
- Q_risk、信用リスク群、競合圧力、推奨金利、営業利益率を必要に応じて参照する。
- 個人名・実在企業名があっても出力では不用意に広げない。
- 日本語で、少し語りかける文体にする。「この案件は」「まず」「ここは先に押さえましょう」などを使ってよい。
- 軽いユーモアを1〜2箇所だけ入れる。審査の品位は保ち、決裁者が読みやすくなる程度にする。
- Streamlit版と同じ構成で、審議の要旨、承認条件、戦略的ポジティブ要因、懸念点と返しを必ず分ける。
- 懸念点は隠さない。ただし必ず「それでも進める根拠」または「潰し方」とセットで書く。
- 否決を煽るのではなく、条件を整えて承認確度を上げる説明にする。

次の JSON オブジェクトだけを返してください。Markdownや説明文は禁止です。
全体を短くし、各配列の要素は45文字前後にしてください。
キー:
- summary: 語りかける1文
- executive_summary: 審議の要旨。なぜこの進め方かを1〜2文
- strategy: 1文。「まず何をするか」を明確に
- approval_conditions: Streamlit版の「融資・リース承認条件」。3〜5個
- positive_factors: Streamlit版の「戦略的ポジティブ要因」。2〜4個
- concerns_and_responses: 「懸念点 → 返し/潰し方」の形で2〜4個
- counter_offer: 逆転の条件・再提案。2〜4個
- risk_points: 3〜6個の配列
- recommended_conditions: 3〜6個の配列
- sales_talk: 2〜5個の配列。顧客に話す言葉として自然に
- humor_note: 最後に添える短い一言。{"八奈見杏奈モードの場合は毒舌・自虐・ご褒美ねだり口調で（例: また残業確定じゃないですか…ケーキ1ホール要求します）" if humor_style == "yanami" else "軽いユーモアを含める（例: 稟議書に添付する前に一杯飲む権利はある）"}
- evidence: 2〜6個の配列
- decision: {{"stance": "短い方針", "confidence": 0.0〜1.0}}
- disclaimer: 1文

入力:
{json.dumps(payload, ensure_ascii=False, default=str)}
"""


def build_gemini_strategy_advice(
    score_result: dict[str, Any] | None = None,
    case: dict[str, Any] | None = None,
    mode: str = "審査軍師",
    timeout_seconds: float = 20.0,
    humor_style: str = "standard",
) -> dict[str, Any]:
    """Use Gemini to rewrite the deterministic advice; fallback on any failure."""
    score_result = score_result or {}
    case = case or {}
    fallback = build_strategy_advice(score_result=score_result, case=case, mode=mode, humor_style=humor_style)
    fallback["source"] = "rule_based"
    api_key = _get_gemini_key()
    if not api_key:
        fallback["source"] = "rule_fallback"
        fallback["llm_error"] = "GEMINI_API_KEY is not configured"
        return fallback

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        timeout_ms = max(10000, int(timeout_seconds * 1000))
        response = client.models.generate_content(
            model=model,
            contents=_build_llm_prompt(fallback, score_result, case, humor_style),
            config=types.GenerateContentConfig(
                max_output_tokens=4000,
                temperature=0.2,
                response_mime_type="application/json",
                response_json_schema={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "executive_summary": {"type": "string"},
                        "strategy": {"type": "string"},
                        "approval_conditions": {"type": "array", "items": {"type": "string"}},
                        "positive_factors": {"type": "array", "items": {"type": "string"}},
                        "concerns_and_responses": {"type": "array", "items": {"type": "string"}},
                        "counter_offer": {"type": "array", "items": {"type": "string"}},
                        "humor_note": {"type": "string"},
                        "risk_points": {"type": "array", "items": {"type": "string"}},
                        "recommended_conditions": {"type": "array", "items": {"type": "string"}},
                        "sales_talk": {"type": "array", "items": {"type": "string"}},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "decision": {
                            "type": "object",
                            "properties": {
                                "stance": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                        },
                        "disclaimer": {"type": "string"},
                    },
                    "required": [
                        "summary",
                        "executive_summary",
                        "strategy",
                        "approval_conditions",
                        "positive_factors",
                        "concerns_and_responses",
                        "counter_offer",
                        "humor_note",
                        "risk_points",
                        "recommended_conditions",
                        "sales_talk",
                        "evidence",
                        "decision",
                        "disclaimer",
                    ],
                },
                http_options=types.HttpOptions(timeout=timeout_ms),
            ),
        )
        parsed = _extract_gemini_json(response)
        if not parsed:
            raise ValueError("Gemini response did not contain a JSON object")
        out = _normalize_llm_advice(parsed, fallback, "gemini")
        out["mode"] = mode
        out["llm_model"] = model
        return out
    except Exception as exc:
        fallback["source"] = "rule_fallback"
        fallback["llm_error"] = str(exc)
        return fallback


def _stance_from_score(score: float, q_risk: float, credit_risk: float) -> tuple[str, float]:
    if score >= 71 and q_risk < 60 and credit_risk < 70:
        return "前向き承認", 0.82
    if score >= 71:
        return "承認候補・要確認", 0.72
    if score >= 50:
        return "条件付き前向き", 0.64
    if score >= 40:
        return "条件再設計", 0.48
    return "慎重対応", 0.34


def build_strategy_advice(
    score_result: dict[str, Any] | None = None,
    case: dict[str, Any] | None = None,
    mode: str = "審査軍師",
    humor_style: str = "standard",
) -> dict[str, Any]:
    """Build a deterministic strategy recommendation from score outputs."""
    score_result = score_result or {}
    case = case or {}
    streamlit = score_result.get("streamlit") or {}
    aurion = score_result.get("aurion") or {}
    q_risk = _safe_float(
        score_result.get("quantum_risk", streamlit.get("quantum_risk", aurion.get("quantum_risk"))),
        0.0,
    )
    credit_risk = _safe_float(streamlit.get("credit_risk_group_score"), 0.0)
    credit_level = str(streamlit.get("credit_risk_group_level") or "unknown")
    financial_q = aurion.get("q_risk") or {}
    financial_q_score = _safe_float(financial_q.get("score"), 0.0)
    financial_q_level = str(financial_q.get("level") or "ok")
    competitor_pressure = aurion.get("competitor_pressure") or {}
    competitor_pressure_score = _safe_float(competitor_pressure.get("score"), 0.0)
    competitor_pressure_level = str(competitor_pressure.get("level") or "ok")

    score = _safe_float(score_result.get("score", streamlit.get("score")), 0.0)
    rf_score = _safe_float(score_result.get("rf_score"), 0.0)
    judgment = str(score_result.get("judgment") or streamlit.get("hantei") or "未判定")
    recommended_rate = _safe_float(score_result.get("recommended_rate"), 0.0)
    base_rate = _safe_float(score_result.get("base_rate"), 0.0)
    spread_pred = _safe_float(score_result.get("spread_pred"), 0.0)

    customer_type = str(case.get("customer_type") or "不明")
    industry = str(case.get("industry_sub") or case.get("industry") or case.get("industry_major") or "不明")
    main_bank = str(case.get("main_bank") or "不明")
    competitor = str(case.get("competitor") or "競合なし")
    competitor_rate = _safe_float(case.get("competitor_rate"), 0.0)
    op_profit = _safe_float(case.get("op_profit"), 0.0)
    nenshu = _safe_float(case.get("nenshu"), 0.0)
    acquisition_cost = _safe_float(case.get("acquisition_cost"), 0.0)
    op_margin = op_profit / nenshu * 100 if nenshu else 0.0
    acq_to_sales = acquisition_cost / nenshu * 100 if nenshu else 0.0

    risk_points: list[str] = []
    recommended_conditions: list[str] = []
    sales_talk: list[str] = []
    evidence: list[str] = []

    _append_unique(evidence, f"総合スコア {score:.0f}/100、判定 {judgment}")
    if rf_score:
        _append_unique(evidence, f"RF成約確率由来スコア {rf_score:.0f}/100")
    if recommended_rate:
        _append_unique(evidence, f"推奨金利 {recommended_rate:.2f}%（基準 {base_rate:.2f}% + スプレッド {spread_pred:.2f}%）")

    if q_risk >= 60:
        _append_unique(risk_points, f"Q_risk {q_risk:.1f} が高く、財務・設備・残高の整合性確認が必要")
        _append_unique(recommended_conditions, "決算書原本、勘定科目内訳、設備明細で入力値の整合性を確認")
    elif q_risk >= 35:
        _append_unique(risk_points, f"Q_risk {q_risk:.1f} は中位。違和感のある財務項目だけ重点確認")
        _append_unique(recommended_conditions, "営業利益、減価償却、リース残高の説明資料を追加取得")
    else:
        _append_unique(evidence, f"Q_risk {q_risk:.1f} は低位")

    if credit_risk >= 70:
        _append_unique(risk_points, f"信用リスク群スコア {credit_risk:.1f} が高く、除外格付DATA群に近い")
        _append_unique(recommended_conditions, "保証人、担保、期間短縮、頭金投入のいずれかを条件化")
    elif credit_risk >= 45:
        _append_unique(risk_points, f"信用リスク群スコア {credit_risk:.1f} は要監視水準")
        _append_unique(recommended_conditions, "直近試算表と主要借入返済予定表を確認")
    elif credit_level != "unknown":
        _append_unique(evidence, f"信用リスク群は {credit_level}（{credit_risk:.1f}/100）")

    if financial_q_level in {"caution", "high_risk"} or financial_q_score >= 20:
        _append_unique(risk_points, f"財務矛盾検知 {financial_q_score:.0f}/100。入力ミスまたは特殊要因の確認が必要")
        _append_unique(recommended_conditions, "財務矛盾の検知項目を担当者が1件ずつ確認")

    if competitor_pressure_level in {"caution", "high_risk"} or competitor_pressure_score >= 35:
        _append_unique(risk_points, f"競合圧力 {competitor_pressure_score:.0f}/100。金利条件の誇張または競合対抗の可能性")
        _append_unique(recommended_conditions, "競合見積の有無、提示条件、対象物件範囲を確認")
        _append_unique(sales_talk, "金利だけで追わず、回答速度・物件理解・メイン行支援を含めた総合提案で比較させる")

    if competitor == "競合あり":
        if competitor_rate > 0 and recommended_rate > 0:
            gap = recommended_rate - competitor_rate
            if gap > 0.3:
                _append_unique(risk_points, f"推奨金利が競合金利を {gap:.2f}% 上回り、失注リスクがある")
                _append_unique(sales_talk, "競合金利との差は、保守・期間・対象設備・回答条件の違いとして説明する")
            else:
                _append_unique(sales_talk, "競合金利との差は小さいため、稟議スピードと条件明確化で押し切る")
        else:
            _append_unique(recommended_conditions, "競合ありの場合は競合提示金利を確認")

    if op_margin < 0:
        _append_unique(risk_points, f"営業利益率 {op_margin:.1f}% で赤字。返済原資の説明が弱い")
        _append_unique(recommended_conditions, "赤字要因、改善見込み、受注残または固定費削減計画を確認")
    elif op_margin < 2:
        _append_unique(risk_points, f"営業利益率 {op_margin:.1f}% で収益余力が薄い")
        _append_unique(recommended_conditions, "月額リース料を含めた資金繰り表を確認")

    if acq_to_sales >= 30:
        _append_unique(risk_points, f"取得価額が年商比 {acq_to_sales:.1f}% と大きい")
        _append_unique(recommended_conditions, "リース期間短縮、頭金、または対象物件の一部分割を検討")

    if main_bank == "メイン先":
        _append_unique(sales_talk, "メイン先である点を活かし、継続取引と支援姿勢を稟議の補強材料にする")
    elif customer_type == "新規先":
        _append_unique(recommended_conditions, "新規先のため、商流、実質経営者、主要取引先の確認を厚めにする")

    if score >= 71:
        _append_unique(sales_talk, "承認圏内でも、確認条件を先に提示して後工程の差し戻しを防ぐ")
    elif score >= 50:
        _append_unique(sales_talk, "条件付き承認を前提に、顧客へ追加条件の理由を先回りして説明する")
        _append_unique(recommended_conditions, "承認ライン到達に不足している条件を1つに絞って交渉")
    else:
        _append_unique(sales_talk, "そのまま押すより、金額・期間・保証条件を再設計して再審査に回す")
        _append_unique(recommended_conditions, "取得額減額、期間短縮、保証追加の再設計案を作成")

    if not risk_points:
        _append_unique(risk_points, "重大な警戒信号は少ない。条件明確化と実行スピードが主戦場")
    if not recommended_conditions:
        _append_unique(recommended_conditions, "通常確認資料を揃え、金利・期間・対象物件を明確化")
    if not sales_talk:
        _append_unique(sales_talk, "スコア根拠、設備効果、返済原資を短く整理して顧客に伝える")

    stance, confidence = _stance_from_score(score, q_risk, credit_risk)
    if q_risk >= 60 and credit_risk >= 70:
        stance = "強警戒・条件厳格化"
        confidence = min(confidence, 0.52)

    summary = (
        f"この{industry}の{customer_type}案件は、{judgment}水準です。"
        f"まずは「{risk_points[0]}」を先に押さえましょう。"
    )
    executive_summary = (
        f"本案件は総合スコア{score:.0f}点で、真正面から否定する案件ではありません。"
        "ただし懸念を隠して進めるより、条件を先に置いて審査部の不安を潰す進め方が有効です。"
        if score >= 50 else
        f"本案件は総合スコア{score:.0f}点で、現条件のまま押し切るには材料が足りません。"
        "金額・期間・保証条件を組み替え、再提案として持ち込む方が現実的です。"
    )
    strategy = (
        "まず追加資料と承認条件をこちらから提示し、審査部に突かれる前に論点を整理します。"
        if score >= 50
        else "そのまま強行せず、取得額・期間・保証を再設計してから再審査へ回します。"
    )
    positive_factors = evidence[:3] or [f"総合スコア {score:.0f}/100、判定 {judgment}"]
    concerns_and_responses = [
        f"{risk_points[0]}。ここは追加資料で先に説明し、判断材料不足にしないことが重要です。"
    ]
    if len(risk_points) > 1:
        concerns_and_responses.append(
            f"{risk_points[1]}。ただし条件設定とモニタリングを付ければ、審査上の不安は下げられます。"
        )
    counter_offer = recommended_conditions[:3]
    if humor_style == "yanami":
        _yanami_notes_high = [
            "スコア見て少し安心した。でも油断したら稟議書が私の枕元に来るんで、ちゃんと条件つけてください。あとコーヒー飲み物おごってください。",
            "まあ…悪くないです。この案件が通ったら、私は今夜ホテルのビュッフェに行く権利があると思ってます。",
            "なんとか形になってますね。審査部に突かれる前に答案を出しましょう。その後、私はしっかり帰宅します。",
        ]
        _yanami_notes_low = [
            "これは…つらいですね正直。でも正面突破より条件組み替えで行きましょ。承認取れたらケーキ1ホール要求します。",
            "今のままだと審査部に返ってきます、絶対。作戦変えましょ。変えたら私に高いパン奢ってください。",
            "厳しい戦況ですね。でも私はまだ諦めてません…疲れてますが。条件を絞り直せばまだ行けます。",
        ]
        humor_note = random.choice(_yanami_notes_high if score >= 50 else _yanami_notes_low)
    else:
        _standard_notes_high = [
            "この案件は、先に宿題を片づければ戦えます。審査部に突かれる前に、こちらから答案用紙を出しましょう。",
            "スコアは合格点。あとは審査部が「これは通さないと損」と思うように見せるだけです。",
            "財務は悪くない。稟議書に「条件と根拠」を先に書いてしまえば、審査部の突込みを先手で潰せます。",
        ]
        _standard_notes_low = [
            "今は力押しより作戦変更です。無理に正面突破すると、稟議書が先に息切れします。",
            "条件を絞り直して再提案する方が早道です。審査部は「断るより通したい」側です。その背中を押しましょう。",
            "スコアが示す通り要審議ですが、条件次第で逆転できます。ここで引いたら負け確定です。",
        ]
        humor_note = random.choice(_standard_notes_high if score >= 50 else _standard_notes_low)

    return {
        "mode": mode,
        "summary": summary,
        "executive_summary": executive_summary,
        "strategy": strategy,
        "approval_conditions": recommended_conditions[:5],
        "positive_factors": positive_factors[:4],
        "concerns_and_responses": concerns_and_responses[:4],
        "counter_offer": counter_offer[:4],
        "humor_note": humor_note,
        "risk_points": risk_points[:6],
        "recommended_conditions": recommended_conditions[:6],
        "sales_talk": sales_talk[:5],
        "evidence": evidence[:6],
        "decision": {
            "stance": stance,
            "confidence": round(confidence, 2),
            "score": round(score, 1),
            "judgment": judgment,
        },
        "metrics": {
            "q_risk": round(q_risk, 2),
            "credit_risk_group_score": round(credit_risk, 2),
            "financial_q_risk_score": round(financial_q_score, 2),
            "competitor_pressure_score": round(competitor_pressure_score, 2),
            "recommended_rate": round(recommended_rate, 2),
            "competitor_rate": round(competitor_rate, 2),
            "op_margin": round(op_margin, 2),
            "acquisition_to_sales": round(acq_to_sales, 2),
        },
        "disclaimer": "軍師AIは判定を上書きしません。最終判断は審査ルール、スコア、担当者確認に従ってください。",
    }
