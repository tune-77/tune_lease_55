"""
AI要因分析モジュール
定性・定量の案件データをGemini APIに送り、成約/失注パターンを分析させる。
ローカルのLightGBM/LRが重い場合の代替として使用。
"""
from __future__ import annotations
import json
import streamlit as st
from data_cases import load_all_cases
from secret_manager import get_gemini_api_key
from config import GEMINI_MODEL_DEFAULT

_QUAL_FIELDS = [
    ("main_bank",       "取引区分"),
    ("competitor",      "競合状況"),
    ("customer_type",   "顧客区分"),
    ("qual_corr_company_history",    "設立・経営年数"),
    ("qual_corr_customer_stability", "顧客安定性"),
    ("qual_corr_repayment_history",  "返済履歴"),
    ("qual_corr_business_future",    "事業将来性"),
    ("qual_corr_equipment_purpose",  "設備目的"),
    ("qual_corr_main_bank",          "メイン取引銀行"),
    ("deal_occurrence", "発生経緯"),
    ("num_competitors", "競合社数"),
]

_QUANT_FIELDS = [
    ("score",           "審査スコア",        False),
    ("sales",           "売上高(百万円)",     True),
    ("op_profit",       "営業利益(百万円)",   True),
    ("net_income",      "当期純利益(百万円)", True),
    ("total_assets",    "総資産(百万円)",     True),
    ("net_assets",      "純資産(百万円)",     True),
    ("bank_credit",     "銀行与信(百万円)",   True),
    ("lease_credit",    "リース与信(百万円)", True),
    ("acquisition_cost","取得価格(百万円)",   True),
    ("grade",           "格付",              False),
    ("final_rate",      "獲得レート(%)",      False),
    ("industry_major",  "業種",              False),
]


def _collect_cases() -> list[dict]:
    cases = load_all_cases()
    return [c for c in cases if c.get("final_status") in ("成約", "失注")]


def _build_qual_prompt(cases: list[dict]) -> str:
    lines = ["# 定性要因分析データ（成約/失注案件）\n"]
    header = ["結果"] + [label for _, label in _QUAL_FIELDS]
    lines.append(",".join(header))
    for c in cases:
        inputs = c.get("inputs") or {}
        row = [c.get("final_status", "")]
        for key, _ in _QUAL_FIELDS:
            val = c.get(key) or inputs.get(key) or ""
            row.append(str(val).replace(",", "、"))
        lines.append(",".join(row))

    prompt = "\n".join(lines)
    prompt += """

## 依頼
上記は温水式リース審査システムに蓄積された成約・失注案件の定性データです。
以下の観点で日本語で分析してください：

1. **成約率に最も影響している定性項目**（上位3〜5項目）とその理由
2. **失注案件に多く見られるパターン**（定性的特徴）
3. **成約率向上のために営業が注意すべき定性的ポイント**
4. **取引区分（メイン先/非メイン先）の影響度**
5. **競合状況と成約率の関係**

回答は箇条書きと表を混ぜて、実務担当者が理解しやすい形式でお願いします。
"""
    return prompt


def _build_quant_prompt(cases: list[dict]) -> str:
    lines = ["# 定量要因分析データ（成約/失注案件）\n"]
    header = ["結果"] + [label for _, label, _ in _QUANT_FIELDS]
    lines.append(",".join(header))
    for c in cases:
        inputs = c.get("inputs") or {}
        result = c.get("result") or {}
        row = [c.get("final_status", "")]
        for key, _, to_man in _QUANT_FIELDS:
            val = (c.get(key)
                   or inputs.get(key)
                   or result.get(key)
                   or "")
            if to_man and val != "":
                try:
                    val = round(float(val) / 1000, 1)
                except (TypeError, ValueError):
                    pass
            row.append(str(val).replace(",", ""))
        lines.append(",".join(row))

    prompt = "\n".join(lines)
    prompt += """

## 依頼
上記は温水式リース審査システムの成約・失注案件の定量データです。
以下の観点で日本語で分析してください：

1. **成約案件と失注案件の財務指標の比較**（平均値・傾向の違い）
2. **成約率に最も影響している定量項目**（上位3〜5項目）
3. **審査スコアと実際の成約率の関係**（スコア帯別の傾向）
4. **失注案件に多い財務的特徴**（格付・利益率・与信状況など）
5. **金利（獲得レート）と成約率の関係**
6. **業種別の傾向**（成約しやすい/しにくい業種）

回答は実務担当者が活用できる形で、具体的な数値への言及も含めてください。
"""
    return prompt


def _call_gemini(prompt: str, max_tokens: int = 3000) -> str:
    api_key = (
        st.session_state.get("gemini_api_key", "").strip()
        or get_gemini_api_key()
        or ""
    )
    if not api_key:
        return "⚠️ Gemini APIキーが設定されていません。サイドバーの「AIモデル設定」で設定してください。"

    model = st.session_state.get("gemini_model", GEMINI_MODEL_DEFAULT) or "gemini-2.0-flash"
    if "1.5" in model:
        model = "gemini-2.0-flash"

    # 新SDK優先、旧SDKにフォールバック
    try:
        import google.genai as _genai
        from google.genai import types as _genai_types
        client = _genai.Client(api_key=api_key)
        config = _genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens, temperature=0.3,
        )
        resp = client.models.generate_content(model=model, contents=prompt, config=config)
        text = None
        try:
            text = resp.text
        except Exception:
            pass
        if not text and getattr(resp, "candidates", None):
            for cand in resp.candidates:
                if getattr(cand, "content", None):
                    for part in cand.content.parts or []:
                        if getattr(part, "text", None):
                            text = (text or "") + part.text
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    try:
        import google.generativeai as _old_genai
        _old_genai.configure(api_key=api_key)
        m = _old_genai.GenerativeModel(
            model_name=model,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.3},
        )
        resp = m.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        if text.strip():
            return text.strip()
    except Exception as e:
        return f"⚠️ Gemini API エラー: {e}"

    return "⚠️ Gemini APIの呼び出しに失敗しました。"


def render_ai_qual_analysis() -> None:
    cases = _collect_cases()
    n = len(cases)
    if n == 0:
        st.warning("成約・失注の登録案件がありません。")
        return

    st.info(f"対象案件: **{n}件**（成約+失注）")

    if st.button("🚀 Gemini で分析実行", key="ai_qual_run", type="primary"):
        with st.spinner("Gemini に分析を依頼中…"):
            result = _call_gemini(_build_qual_prompt(cases), max_tokens=3500)
        st.session_state["ai_qual_result"] = result
        st.rerun()

    result = st.session_state.get("ai_qual_result")
    if result:
        st.markdown(result)
        if st.button("🗑️ 結果をクリア", key="ai_qual_clear"):
            del st.session_state["ai_qual_result"]
            st.rerun()


def render_ai_quant_analysis() -> None:
    cases = _collect_cases()
    n = len(cases)
    if n == 0:
        st.warning("成約・失注の登録案件がありません。")
        return

    st.info(f"対象案件: **{n}件**（成約+失注）")

    if st.button("🚀 Gemini で分析実行", key="ai_quant_run", type="primary"):
        with st.spinner("Gemini に分析を依頼中…"):
            result = _call_gemini(_build_quant_prompt(cases), max_tokens=4000)
        st.session_state["ai_quant_result"] = result
        st.rerun()

    result = st.session_state.get("ai_quant_result")
    if result:
        st.markdown(result)
        if st.button("🗑️ 結果をクリア", key="ai_quant_clear"):
            del st.session_state["ai_quant_result"]
            st.rerun()
