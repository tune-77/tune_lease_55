import json
import streamlit as st
from components.agent_hub import _ai_call

def evaluate_asset_value(asset_name: str, model_no: str, acquisition_cost: float, term_months: int) -> dict:
    """
    Gemini API を使用して物件の資産価値（残価率・中古相場・流動性）を推定する。
    """
    system_prompt = """
    あなたはリース物件の資産価値・二次流通相場を評価する「プロの担保評価士」です。
    入力された物件について、市場価値を客観的に査定してください。
    """
    
    prompt = f"""
    以下の物件について、リース満了時（{term_months}ヶ月後）の資産性を分析してください。

    【物件情報】
    - 物件名: {asset_name}
    - メーカー・型番: {model_no}
    - 取得価格: {acquisition_cost:,.0f}円
    
    以下のJSON形式のみで出力してください。マークダウンのコードブロック（```json ... ```）を含めず、純粋なJSON文字列としてください。

    {{
        "residual_value_pct": "数値（リース満了時の想定残価率。0〜100の整数。例: 35）",
        "liquidity_rank": "流動性ランク（A/B/C/D）",
        "market_demand": "中古市場での需要の強さについての解説（100文字以内）",
        "risks": "想定される陳腐化や売却時の懸念点（100文字以内）",
        "suggested_depreciation_rate": "実務に即した推奨年間減価率（%表示の数値。例: 15.5）"
    }}
    """
    
    res_text = _ai_call(prompt, system=system_prompt)
    
    # JSONのクリーンアップとパース
    res_text = res_text.replace("```json", "").replace("```", "").strip()
    
    try:
        eval_data = json.loads(res_text)
    except Exception:
        # パース失敗時のフォールバック
        eval_data = {
            "residual_value_pct": 20,
            "liquidity_rank": "B",
            "market_demand": "分析エラー。通常の汎用物件としての需要が見込まれます。",
            "risks": "特記事項なし。",
            "suggested_depreciation_rate": 20.0
        }
        
    return eval_data

def render_asset_valuer():
    """Streamlit UI"""
    st.title("🔍 AI 物件資産価値シミュレーター")
    st.caption("Gemini API を活用し、物件の型番や取得価格から「将来の中古市場価値（残価）」を瞬時に推測します。")

    with st.form("asset_valuer_form"):
        col1, col2 = st.columns(2)
        with col1:
            asset_name = st.text_input("物件名", placeholder="例: 油圧ショベル / トラック")
            model_no = st.text_input("メーカー・型番", placeholder="例: コマツ PC200-10")
        with col2:
            acquisition_cost = st.number_input("取得価格 (円)", min_value=0, value=10_000_000, step=500_000)
            term_months = st.number_input("リース期間 (ヶ月)", min_value=1, value=60, step=12)
            
        submitted = st.form_submit_button("💡 AIによる資産価値評価を実行", type="primary")

    if submitted:
        if not asset_name.strip():
            st.error("物件名を入力してください。")
            return
            
        with st.spinner("Gemini が中古市場・残存価値を調査中..."):
            res = evaluate_asset_value(asset_name, model_no, acquisition_cost, term_months)
            
        st.success("評価が完了しました！")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("満了時 想定残価率", f"{res['residual_value_pct']}%")
        with col_b:
            st.metric("流動性ランク", f"Rank {res['liquidity_rank']}")
        with col_c:
            st.metric("推奨年間減価率", f"{res['suggested_depreciation_rate']}%")
            
        st.markdown("### 📊 中古市場・リスク分析")
        st.info(f"**【市場需要】**  \n{res['market_demand']}")
        st.warning(f"**【リスク要因】**  \n{res['risks']}")
        
        # アセットエンジンとの連動アドバイス
        st.markdown("### 🛠️ 与信ロジックへの反映")
        st.caption("現在の配点ルールに対するアドバイスです。")
        if res['liquidity_rank'] in ['A', 'B']:
            st.success("👍 物件の換価性が高いため、財務審査が厳しい場合でも「条件付き承認」に引き上げる保全力があります。")
        else:
            st.error("⚠️ 物件価値に依存した審査（アセットファイナンス）は危険です。企業の純粋な返済能力（PD）を厳格に評価してください。")
