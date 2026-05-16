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


def _estimate_probability_uplifts(
    score_result: dict[str, Any],
    case: dict[str, Any],
    indicator_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """条件を足したときの承認確率上振れの目安を返す。"""
    score = _safe_float(score_result.get("score"), 0.0)
    base_prob = _safe_float(score_result.get("probability"), 0.0) * 100.0
    if base_prob <= 0:
        base_prob = max(5.0, min(95.0, score * 0.8))

    streamlit = score_result.get("streamlit") or {}
    aurion = score_result.get("aurion") or {}
    q_risk = _safe_float(score_result.get("quantum_risk", streamlit.get("quantum_risk", aurion.get("quantum_risk"))), 0.0)
    credit_risk = _safe_float(streamlit.get("credit_risk_group_score"), 0.0)
    competitor_pressure = aurion.get("competitor_pressure") or {}
    competitor_score = _safe_float(competitor_pressure.get("score"), 0.0)

    op_profit = _safe_float(case.get("op_profit"), 0.0)
    nenshu = _safe_float(case.get("nenshu"), 0.0)
    competitor = str(case.get("competitor") or "競合なし")
    competitor_rate = _safe_float(case.get("competitor_rate"), 0.0)
    main_bank = str(case.get("main_bank") or "不明")
    customer_type = str(case.get("customer_type") or "不明")

    indicators = indicator_analysis.get("indicators") or indicator_analysis.get("rows") or []
    uplift_items: list[dict[str, Any]] = []

    def add_item(title: str, gain: float, condition: str, reason: str) -> None:
        gain = max(0.0, min(20.0, float(gain)))
        uplift_items.append({
            "title": title,
            "gain_pct": round(gain, 1),
            "from": round(base_prob, 1),
            "to": round(min(99.0, base_prob + gain), 1),
            "condition": condition,
            "reason": reason,
        })

    if q_risk >= 60:
        add_item(
            "Q_risk の矛盾解消",
            8.0,
            "残高・減価償却・設備明細を突合",
            "会計整合性が整うと、ベイズ的に警戒が大きく下がる",
        )
    elif q_risk >= 35:
        add_item(
            "Q_risk の要注意項目だけ補強",
            4.0,
            "違和感のある科目を1件ずつ説明",
            "疑義の範囲を限定できれば、上振れ余地が出る",
        )

    if competitor == "競合あり" and competitor_rate > 0:
        gap = max(0.0, competitor_rate - _safe_float(score_result.get("recommended_rate"), 0.0))
        add_item(
            "競合条件との差を明確化",
            3.0 if gap <= 0.3 else 5.0,
            "回答速度・条件明確化・比較表を用意",
            "金利だけでなく総合条件で比較できれば上振れする",
        )
    elif competitor_score >= 35:
        add_item(
            "競合圧力への先回り説明",
            4.0,
            "競合の見積もり前提を整理",
            "競合の論点が整理されると失注確率が下がる",
        )

    if op_profit < 0:
        add_item(
            "赤字理由と改善見込みを提示",
            6.0,
            "固定費削減・受注残・回復計画を添付",
            "赤字でも回復筋が見えると承認余地が増える",
        )
    elif nenshu > 0 and op_profit / nenshu * 100 < 2:
        add_item(
            "営業利益率を補強",
            4.0,
            "月次資金繰り表を添える",
            "返済原資の見通しが立てば上振れしやすい",
        )

    if main_bank == "メイン先":
        add_item(
            "メイン行の支援を明文化",
            2.0,
            "継続取引・推薦状・協調姿勢を添付",
            "信用補完が1枚増えるだけで確率は底上げしやすい",
        )
    if customer_type == "新規先":
        add_item(
            "商流・実質経営者・主要取引先を補強",
            3.0,
            "新規先の不確実性を潰す資料を揃える",
            "新規先の不透明感が減れば承認余地が広がる",
        )

    for row in indicators[:2]:
        name = str(row.get("name") or row.get("label") or "")
        text = str(row.get("text") or "")
        if not name:
            continue
        if any(term in name for term in ("自己資本比率", "ROA", "ROE", "営業利益率")) and "下" in text:
            add_item(
                f"{name} を業界目安へ近づける",
                3.0,
                "低い指標の説明資料を足す",
                "業界平均との差が縮まると、見立ての説得力が上がる",
            )

    if not uplift_items:
        if score >= 71:
            add_item(
                "確認条件の先出し",
                2.0,
                "差し戻し要因を先に潰す",
                "承認圏内では、条件の見え方を整えるだけで上振れしやすい",
            )
        elif score >= 50:
            add_item(
                "条件付き承認の整理",
                4.0,
                "追加資料・期間調整・保証補強をまとめる",
                "条件を1本化すると、承認確度が少し上がる",
            )
        else:
            add_item(
                "再提案の組み替え",
                6.0,
                "金額調整・期間短縮・保証追加をセットで出す",
                "案件構造を組み替えると、承認余地が大きく改善する",
            )

    uplift_items.sort(key=lambda x: x["gain_pct"], reverse=True)
    return uplift_items[:5]


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
        "additional_guidance",
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
    uplifts = raw.get("probability_uplifts")
    if isinstance(uplifts, list):
        cleaned_uplifts: list[dict[str, Any]] = []
        for item in uplifts:
            if not isinstance(item, dict):
                continue
            cleaned = {
                "title": str(item.get("title") or "").strip(),
                "gain_pct": _safe_float(item.get("gain_pct"), 0.0),
                "from": _safe_float(item.get("from"), 0.0),
                "to": _safe_float(item.get("to"), 0.0),
                "condition": str(item.get("condition") or "").strip(),
                "reason": str(item.get("reason") or "").strip(),
            }
            if cleaned["title"]:
                cleaned_uplifts.append(cleaned)
        if cleaned_uplifts:
            out["probability_uplifts"] = cleaned_uplifts[:6]
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


def _build_llm_prompt(
    fallback: dict[str, Any],
    score_result: dict[str, Any],
    case: dict[str, Any],
    humor_style: str = "yanami",
    obsidian_hits: list[dict[str, str]] | None = None,
) -> str:
    obsidian_section = ""
    if obsidian_hits:
        lines = []
        for h in obsidian_hits[:4]:
            path = h.get("path", "")
            snippet = h.get("snippet", "").strip()[:300]
            if snippet:
                lines.append(f"- [{path}] {snippet}")
        if lines:
            obsidian_section = (
                "\n\nObsidian過去メモ（類似案件・成功パターン・改善ログ）:\n"
                + "\n".join(lines)
                + "\n上記の過去メモに具体的な条件や言い回しがあれば、それを優先して使うこと。"
            )
    payload = {
        "case": case,
        "score_result": score_result,
        "rule_based_advice": fallback,
    }
    if humor_style == "yanami":
        _persona = """あなたは八奈見杏奈です。12年間片思いして失恋した経験を持つベテラン審査員。表向きは明るく前向きだが、内心では絶望を言語化することで乗り越えるタイプ。
口調はサバサバした毒舌。でも審査は絶対に外さない。「プロの幼なじみ」ならぬ「プロの審査員」として、自虐とぼやきを混ぜながら「どう通すか」を短く具体的に示します。
以下のような口調を随所に1〜2個入れてください（毎回同じにしない）:
- 塩ツンデレ系: 「悪くないですよ（褒めてないです）」「顔には出しませんが」「当然です」
- プロ負けヒロイン系: 「フラれてもすっきりしないんだよ…稟議も同じ。でも進むしかない」「ようこそ、厳しい案件の世界に」
- 審査ぼやき系: 「こういう案件に限って夜が長い」「資金繰り表を見ながら溜息をついた数が今日一番でした」
- 諦め前向き系: 「疲れてますが諦めてはないです」「無理やり進むしかなくなってきました」
- 食べ物系（たまにOK）: 審査コメントに着地させた上で添える程度。「帰りにコーヒーでも」「そうめんでも食べながら考えます」程度にとどめ、食べ物だけが前面に出ないこと。
summary や strategy は八奈見口調で書く。でも審査判断は変えない。懸念点と返しはむしろ鋭く。"""
    else:
        _persona = """あなたは法人リース審査の「審査軍師AI」です。
決裁者と営業担当者に語りかける戦略アドバイザーです。
役割は、審査スコアを上書きせず、「どう通すか」「どこを先に潰すか」を短く具体的に示すことです。"""

    return f"""{_persona}{obsidian_section}

制約:
- 判定は変更しない。最終判断は score_result の score / judgment を尊重する。
- 根拠のない断定は禁止。
- 計算済み指標がある場合は、それを最優先で説明する。スコアの数字だけで話さない。
- 業種平均との差、利益率、自己資本比率、ROA/ROE、総資産回転率、金利差など、算出済みの指標をもとに次の一手を決める。
- Q_risk、信用リスク群、競合圧力、推奨金利、営業利益率を必要に応じて参照する。
- 個人名・実在企業名があっても出力では不用意に広げない。
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
- additional_guidance: 追加の指南。計算指標と条件整理を踏まえた補足助言。2〜6個
- probability_uplifts: 条件を足したときの承認確率上振れ見込み。最大5件。各項目に gain_pct / from / to / condition / reason を含める
- sales_talk: 2〜5個の配列。顧客に話す言葉として自然に
- humor_note: 最後に添える短い一言。{"八奈見杏奈モードの場合は以下を参考に（毎回同じにしない）: ①塩ツンデレ系「悪くないですよ（褒めてないです）」「顔には出しませんが」②プロ負けヒロイン系「フラれてもすっきりしないんだよ…稟議も同じ。でも進むしかない」③審査ぼやき系「資金繰り表を見ながら溜息をついた数が今日一番でした」「こういう案件に限って夜が長い」④諦め前向き系「疲れてますが諦めてないです」⑤食べ物（たまにOK）「帰りにコーヒーでも」「そうめんでも食べながら考えます」← ただし審査コメントに着地させた上で添えること。食べ物・キャラ性だけが前面に出ず、最後の印象が「確認すべきことが明確」になるよう終わらせること。" if humor_style == "yanami" else "軽いユーモアを含める（例: 稟議書に添付する前に一杯飲む権利はある）"}
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
    humor_style: str = "yanami",
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

    # Obsidian から類似案件・過去メモを検索
    obsidian_hits: list[dict[str, str]] = []
    try:
        from obsidian_bridge import search_notes
        industry = str(case.get("industry_major") or case.get("industry") or "")
        judgment = str(score_result.get("judgment") or score_result.get("hantei") or "")
        score = _safe_float(score_result.get("score"), 0.0)
        q_risk = _safe_float(score_result.get("quantum_risk") or
                             (score_result.get("aurion") or {}).get("q_risk", {}).get("score"), 0.0)
        query_parts = [industry, "リース審査", judgment]
        if score < 50:
            query_parts.append("逆転 条件付き")
        if q_risk >= 35:
            query_parts.append("Q_risk 財務矛盾")
        obsidian_hits = search_notes(" ".join(query_parts), limit=4, max_chars=300)
    except Exception:
        pass

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        timeout_ms = max(10000, int(timeout_seconds * 1000))
        response = client.models.generate_content(
            model=model,
            contents=_build_llm_prompt(fallback, score_result, case, humor_style, obsidian_hits),
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
                        "additional_guidance": {"type": "array", "items": {"type": "string"}},
                        "probability_uplifts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "gain_pct": {"type": "number"},
                                    "from": {"type": "number"},
                                    "to": {"type": "number"},
                                    "condition": {"type": "string"},
                                    "reason": {"type": "string"},
                                },
                            },
                        },
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
                        "additional_guidance",
                        "probability_uplifts",
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
    humor_style: str = "yanami",
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

    indicator_analysis = score_result.get("indicator_analysis") or {}
    indicator_summary = str(indicator_analysis.get("summary") or "")
    indicator_detail = str(indicator_analysis.get("detail") or "")
    indicator_rows = indicator_analysis.get("indicators") or indicator_analysis.get("rows") or []
    uplift_items = _estimate_probability_uplifts(score_result, case, indicator_analysis)

    risk_points: list[str] = []
    recommended_conditions: list[str] = []
    additional_guidance: list[str] = []
    sales_talk: list[str] = []
    evidence: list[str] = []
    indicator_takeaways: list[str] = []

    _append_unique(evidence, f"総合スコア {score:.0f}/100、判定 {judgment}")
    if rf_score:
        _append_unique(evidence, f"RF成約確率由来スコア {rf_score:.0f}/100")
    if recommended_rate:
        _append_unique(evidence, f"推奨金利 {recommended_rate:.2f}%（基準 {base_rate:.2f}% + スプレッド {spread_pred:.2f}%）")

    if indicator_summary:
        _append_unique(indicator_takeaways, indicator_summary)
    if indicator_rows:
        for row in indicator_rows[:4]:
            name = str(row.get("name") or row.get("label") or "")
            value = row.get("value")
            bench = row.get("bench")
            unit = str(row.get("unit") or "%")
            try:
                value_num = float(value)
            except (TypeError, ValueError):
                continue
            if bench is None:
                _append_unique(indicator_takeaways, f"{name} {value_num:.1f}{unit}")
            else:
                try:
                    bench_num = float(bench)
                except (TypeError, ValueError):
                    bench_num = None
                if bench_num is not None:
                    diff = value_num - bench_num
                    _append_unique(indicator_takeaways, f"{name} {value_num:.1f}{unit}、業界差 {diff:+.1f}{unit}")
    if indicator_detail:
        _append_unique(indicator_takeaways, "算出指標の差分が出ています。重要項目から順に確認しましょう。")

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
    if indicator_takeaways:
        _append_unique(sales_talk, f"計算指標の見立て: {indicator_takeaways[0]}")

    if indicator_takeaways:
        for item in indicator_takeaways[:3]:
            _append_unique(additional_guidance, item)
    for item in recommended_conditions[:3]:
        _append_unique(additional_guidance, item)
    if q_risk >= 35:
        _append_unique(additional_guidance, "Q_risk の突っ込みどころは、残高・減価償却・設備明細の整合性から先に潰す")
    if competitor == "競合あり":
        _append_unique(additional_guidance, "競合ありなら、金利差だけでなく回答速度と条件明確化で勝負する")
    if score >= 50:
        _append_unique(additional_guidance, "条件付き承認は、追加資料→期間調整→保証補強の順で詰める")
    else:
        _append_unique(additional_guidance, "再提案は、金額調整・期間短縮・保証追加の3点を先に作る")
    if uplift_items:
        for item in uplift_items[:3]:
            _append_unique(
                additional_guidance,
                f"{item['title']}で承認確率 +{item['gain_pct']:.1f}pt目安（{item['condition']}）",
            )

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
            "稟議書の重さは太りませんが、確認事項の重さは後から来ます。条件は先に整えてください。悪くないですよ（褒めてないです）。",
            "数字がきれいに整っていて、審査表がいつもより少し機嫌よく見えます。私も少しだけ機嫌がいいです、顔には出しませんが。",
            "この安定感なら、稟議書も深呼吸してから読めます。条件を先に出せば、審査部の突込みを先手で潰せます。",
            "財務の足回りが安定していて、リース期間中も落ち着いて走れそうです。帰りにコーヒーでも飲みながら歩けそうです（たまにはいいでしょ）。",
            "この案件、条件を先に置けば審査部は突いてきません。フラれてもすっきりしないんだよ…稟議もそう。でもこれは素直に進める案件です。",
            "OK圏内です。返済計画のアイドリングが安定している案件は久しぶりで、少し報われた気分です（顔には出しません）。",
            "通せます。無理やり周りが進んじゃうから、こっちも進むしかなくなってきた…そういうときにこういう案件があると助かります。",
            "悪くない。私はプロの審査員なので当然ですが、ぬか喜びはしないでください。条件を先に整えてから喜んでください。",
        ]
        _yanami_notes_low = [
            "条件を整えなかった案件が後から返ってくるのを何度も見てきたので、今回は先に潰します。疲れてますが諦めてないです。",
            "数字に息切れが見えます。この案件の資金繰り表を見ながら溜息をついた数、今日一番でした。でも条件を絞り直せばまだ行けます。",
            "今のままだと審査部に返ってきます、絶対。フラれてもすっきりしないんだよ…稟議も同じ。でも条件変えれば進めます。",
            "走り出す前に、ブレーキと保険と出口戦略をもう一度見たい案件です。こういう案件に限って夜が長いですね（でもやります）。",
            "ようこそ、厳しい案件の世界に。…私もプロなので、なんとかします。帰りにそうめんでも食べながら考えます（たまにはいいでしょ）。",
            "前に進む余地はありますが、条件なしの発進は避けたいところです。設備より先に返済計画の酸素残量を確認したい案件です。",
            "この案件が通ったら、12年間ってなんだったんだろうって思います…案件の話です。でも条件次第で可能性はあります。",
            "正面突破より条件組み替えで行きましょ。審査部の突込みどころを先に潰せば、まだ勝負できます。諦めてはないです。",
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
        "indicator_summary": indicator_summary,
        "indicator_detail": indicator_detail,
        "indicator_takeaways": indicator_takeaways[:6],
        "probability_uplifts": uplift_items[:5],
        "executive_summary": executive_summary,
        "strategy": strategy,
        "approval_conditions": recommended_conditions[:5],
        "positive_factors": positive_factors[:4],
        "concerns_and_responses": concerns_and_responses[:4],
        "counter_offer": counter_offer[:4],
        "humor_note": humor_note,
        "risk_points": risk_points[:6],
        "recommended_conditions": recommended_conditions[:6],
        "additional_guidance": additional_guidance[:6],
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
