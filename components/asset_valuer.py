import json
import os
from typing import Optional
import streamlit as st
from components.agent_hub import _ai_call
from expected_usage_period import find_item_by_name
from runtime_paths import REPO_ROOT

# 汎用性が低く、リース満了時の転売・再流通が困難な「特注・造作・専用設備」を
# 検出するためのキーワード。該当する場合、AIが楽観的な流動性係数を返しても
# 上限値でキャップし、残価設定が市場価値に見合わない事態を防ぐ。
# 根拠: SHION-9BADD3（紫苑の自己提案）— セントラルキッチン等の特注厨房設備は
# 汎用性・中古市場での需要が乏しく、残価回収リスクが高い。
LOW_VERSATILITY_KEYWORDS = [
    "セントラルキッチン", "central kitchen",
    "特注", "オーダーメイド", "造作", "専用什器", "専用設備",
    "特別仕様", "オンリーワン", "一品もの", "設計施工一式",
    "内装工事", "什器一式", "店舗設計", "店舗内装", "テナント工事", "造作工事",
]

# 特注・造作設備と判定された場合の流動性補正係数の上限。
# Aランク（0.95）等の楽観値をそのまま採用せず、市場流動性の乏しさを
# 保守的に反映する。
LOW_VERSATILITY_LIQUIDITY_CAP = 0.6


def _detect_low_versatility(asset_name: str, model_no: str) -> Optional[str]:
    """物件名・型番から汎用性の低い特注/造作設備かどうかを判定する。
    該当する場合はマッチしたキーワードを返し、該当しなければ None を返す。
    """
    haystack = f"{asset_name or ''} {model_no or ''}".lower()
    for kw in LOW_VERSATILITY_KEYWORDS:
        if kw.lower() in haystack:
            return kw
    return None


def _load_nta_useful_life() -> dict:
    """static_data/useful_life_equipment.json を読み込む"""
    path = REPO_ROOT / "static_data" / "useful_life_equipment.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def find_useful_life_by_name(asset_name: str) -> tuple[Optional[int], str]:
    """static_data/useful_life_equipment.json から耐用年数を検索"""
    data = _load_nta_useful_life()
    if not data:
        return None, ""
    
    asset_name_lower = asset_name.lower()
    for cat in data.get("categories", []):
        for item in cat.get("items", []):
            item_name = item.get("name", "")
            
            # 完全一致または包含チェック
            if item_name.lower() in asset_name_lower or asset_name_lower in item_name.lower():
                return item.get("years"), item_name
            
            # 「ショベル・油圧ショベル」のようなスラッシュ（・）区切りも考慮
            if "・" in item_name:
                for sub in item_name.split("・"):
                    if sub.strip() and sub.strip().lower() in asset_name_lower:
                        return item.get("years"), item_name
    return None, ""

def evaluate_asset_value(asset_name: str, model_no: str, acquisition_cost: float, term_months: int) -> dict:
    """
    Gemini API を使用して物件の資産価値（残価率・中古相場・流動性）を推定する。
    法定耐用年数は static_data/useful_life_equipment.json 等のマスターデータから参照する。
    """
    # ── 1. 法定耐用年数データのマッチング ──
    useful_life, master_item_name = find_useful_life_by_name(asset_name)
    if not useful_life and model_no:
        useful_life, master_item_name = find_useful_life_by_name(model_no)
        
    # 2. 見つからなければ期待使用期間.jsonから補完
    if not useful_life:
        matched_item = find_item_by_name(asset_name)
        if not matched_item and model_no:
            matched_item = find_item_by_name(model_no)
        if matched_item:
            useful_life = matched_item.get("legal_useful_life", 8)
            master_item_name = matched_item.get("item_name", "")
            
    # 3. 最終フォールバック
    if not useful_life:
        useful_life = 8
        master_item_name = "不明（デフォルト判定）"

    system_prompt = """
    あなたはリース物件の資産価値・二次流通相場を評価する「プロの担保評価士」です。
    入力された物件について、市場価値を客観的に査定してください。
    """
    
    prompt = f"""
    以下の物件について、リース満了時（{term_months}ヶ月後）の資産性を分析してください。
    法定耐用年数は {useful_life} 年として判定してください。

    【物件情報】
    - 物件名: {asset_name}
    - メーカー・型番: {model_no}
    - 取得価格: {acquisition_cost:,.0f}円
    - マスターデータによる耐用年数: {useful_life}年（カテゴリ: {master_item_name}）
    
    以下のJSON形式のみで出力してください。マークダウンのコードブロック（```json ... ```）を含めず、純粋なJSON文字列としてください。

    {{
        "liquidity_rank": "流動性ランク（A/B/C/D）",
        "liquidity_factor": "換価性補正係数（0.5〜1.0の範囲で、中古での売りやすさに応じて設定。Aランクなら0.95、Dランクなら0.5 など。例: 0.85）",
        "market_demand": "中古市場での需要の強さについての解説（100文字以内）",
        "risks": "想定される陳腐化や売却時の懸念点（100文字以内）"
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
            "liquidity_rank": "B",
            "liquidity_factor": 0.8,
            "market_demand": "分析エラー。通常の汎用物件としての需要が見込まれます。",
            "risks": "特記事項なし。"
        }
        
    try:
        liquidity_factor = max(0.1, min(1.0, float(eval_data.get("liquidity_factor", 0.8))))
    except (TypeError, ValueError):
        liquidity_factor = 0.8

    # ── 汎用性チェック（SHION-9BADD3対応） ──
    # AIの流動性係数が、特注・造作設備（例: セントラルキッチン等）の転売困難性を
    # 過小評価している場合に備え、ルールベースで上限をキャップする。
    low_versatility_kw = _detect_low_versatility(asset_name, model_no)
    versatility_capped = False
    if low_versatility_kw and liquidity_factor > LOW_VERSATILITY_LIQUIDITY_CAP:
        versatility_capped = True
        liquidity_factor = LOW_VERSATILITY_LIQUIDITY_CAP

    # 定率法 (200%償却ベース)
    depreciation_rate = 2.0 / useful_life
    years = term_months / 12.0
    
    # 残価率 = (1 - 償却率)^年数 × 流動性補正
    residual_pct = 100.0 * ((1.0 - depreciation_rate) ** years) * liquidity_factor
    # 5%〜95%の範囲に丸める
    residual_pct = max(5.0, min(95.0, round(residual_pct, 1)))
    
    eval_data["residual_value_pct"] = int(residual_pct)
    eval_data["suggested_depreciation_rate"] = round(depreciation_rate * 100.0, 1)
    eval_data["useful_life_years"] = useful_life
    eval_data["master_item_name"] = master_item_name
    eval_data["liquidity_factor"] = round(liquidity_factor, 2)
    eval_data["low_versatility_flag"] = bool(low_versatility_kw)
    eval_data["low_versatility_keyword"] = low_versatility_kw
    eval_data["versatility_capped"] = versatility_capped
    
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
            
        st.success(f"評価が完了しました！ (判定カテゴリ: {res.get('master_item_name', '不明')})")

        if res.get("low_versatility_flag"):
            cap_note = "（AI評価値を保守的に補正済み）" if res.get("versatility_capped") else ""
            st.warning(
                f"⚠️ 汎用性チェック: 「{res.get('low_versatility_keyword')}」に該当する特注・造作設備の可能性があります。"
                f"転用・転売が困難で残価回収リスクが高いため、残価設定は保守的に検討してください{cap_note}。"
            )
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("満了時 想定残価率", f"{res['residual_value_pct']}%")
        with col_b:
            st.metric("法定耐用年数 (DATA)", f"{res.get('useful_life_years', '—')}年")
        with col_c:
            st.metric("流動性ランク", f"Rank {res['liquidity_rank']}")
        with col_d:
            st.metric("推奨年間減価率 (定率)", f"{res['suggested_depreciation_rate']}%")
            
        st.markdown("### 📊 中古市場・リスク分析")
        st.info(f"**【市場需要】**  \n{res['market_demand']}")
        st.warning(f"**【リスク要因】**  \n{res['risks']}")
        
        # アセットエンジンとの連動アドバイス
        st.markdown("### 🛠️ 与信ロジックへの反映")
        st.caption("現在の配点ルールに対するアドバイスです。")
        if res.get("low_versatility_flag"):
            st.error("⚠️ 特注・造作設備のため汎用性が低く、リース満了時の転売性が乏しい可能性があります。物件担保力を過大評価せず、企業の返済能力（PD）を主軸に審査してください。")
        elif res['liquidity_rank'] in ['A', 'B']:
            st.success("👍 物件の換価性が高いため、財務審査が厳しい場合でも「条件付き承認」に引き上げる保全力があります。")
        else:
            st.error("⚠️ 物件価値に依存した審査（アセットファイナンス）は危険です。企業の純粋な返済能力（PD）を厳格に評価してください。")
