"""
components/usage_period_widget.py

期待使用期間マスタを活用した物件リース時間最適性評価ウィジェット。

機能:
- 物件カテゴリ別の機種リストを表示（期待使用期間.json から取得）
- リース期間 vs 期待使用期間の適合度を評価
- グラフィカルに「最適/やや長め/リスク高」の区分を表示
- スコアリングに適合度情報を含める

セッションステートキー:
  upw_category    : str  - 物件カテゴリ
  upw_item_name   : str  - 選択機種名
  upw_lease_months: int  - リース期間（月）
  upw_fit_result  : dict - 適合度評価結果
"""

import streamlit as st
import plotly.graph_objects as go
from expected_usage_period import (
    get_categories_by_group,
    calc_lease_period_fit_score,
)


def render_usage_period_widget(key_prefix: str = "upw"):
    """
    期待使用期間ウィジェットをレンダリング。
    
    Parameters
    ----------
    key_prefix : str
        セッション状態キーのプレフィックス
    
    Returns
    -------
    dict
        {
            'category': str,           # 選択カテゴリ
            'item_name': str,          # 選択機種名
            'lease_months': int,       # リース期間（月）
            'fit_result': dict or None, # 適合度評価結果
        }
    """
    st.markdown("### 📋 期待使用期間による最適性評価")
    st.caption("リース期間が機種の期待使用期間に合致しているか評価します。")
    
    # ────────────────────────────────────────────────────────────────────────────
    # カテゴリ選択
    # ────────────────────────────────────────────────────────────────────────────
    categories_by_group = get_categories_by_group()
    if not categories_by_group:
        st.warning("⚠️ 期待使用期間マスタが見つかりません。")
        return {
            "category": None,
            "item_name": None,
            "lease_months": 0,
            "fit_result": None,
        }
    
    category_names = sorted(categories_by_group.keys())
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col1:
        selected_category = st.selectbox(
            "📦 物件カテゴリ",
            options=category_names,
            key=f"{key_prefix}_category",
            help="リース物件のカテゴリを選択してください。",
        )
    
    # ────────────────────────────────────────────────────────────────────────────
    # 機種選択
    # ────────────────────────────────────────────────────────────────────────────
    if selected_category:
        items_in_category = categories_by_group[selected_category]
        item_names = [item["item_name"] for item in items_in_category]
        
        with col2:
            selected_item = st.selectbox(
                "🔧 機種名",
                options=item_names,
                key=f"{key_prefix}_item_name",
                help="期待使用期間マスタから機種を選択してください。",
            )
    else:
        selected_item = None
    
    # ────────────────────────────────────────────────────────────────────────────
    # リース期間入力
    # ────────────────────────────────────────────────────────────────────────────
    with col3:
        lease_months = st.number_input(
            "⏱️ リース期間（月）",
            min_value=0,
            max_value=120,
            value=st.session_state.get(f"{key_prefix}_lease_months", 60),
            step=1,
            key=f"{key_prefix}_lease_months_input",
            help="予定しているリース期間を月数で入力してください。",
        )
        # セッション状態に保存
        st.session_state[f"{key_prefix}_lease_months"] = lease_months
    
    # ────────────────────────────────────────────────────────────────────────────
    # 適合度評価
    # ────────────────────────────────────────────────────────────────────────────
    fit_result = None
    if selected_item and lease_months > 0:
        fit_result = calc_lease_period_fit_score(selected_item, lease_months)
        
        st.divider()
        
        # ── スコア表示 ────────────────────────────────────────────────────────
        remanufacture_score = fit_result.get("remanufacture_score", 50)
        assessment_label = fit_result.get("assessment_label", "未判定")
        
        # カラー設定
        if remanufacture_score >= 85:
            score_color = "🟢"
            bg_color = "#d4edda"  # 薄い緑
            text_color = "#155724"  # 濃い緑
        elif remanufacture_score >= 70:
            score_color = "🟡"
            bg_color = "#fff3cd"  # 薄い黄
            text_color = "#856404"  # 濃い黄
        else:
            score_color = "🔴"
            bg_color = "#f8d7da"  # 薄い赤
            text_color = "#721c24"  # 濃い赤
        
        # スコア表示
        st.markdown(
            f"""
            <div style="background-color: {bg_color}; padding: 15px; border-radius: 8px; border-left: 4px solid {text_color};">
                <h4 style="margin: 0; color: {text_color};">{score_color} 再リース機会スコア</h4>
                <h2 style="margin: 5px 0; color: {text_color};">{remanufacture_score:.0f}/100</h2>
                <p style="margin: 5px 0; color: {text_color};"><strong>{assessment_label}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # ── 法定耐用年数とリース期間の比較 ────────────────────────────────
        st.markdown("#### 📊 法定耐用年数との比較")
        
        expected_years = fit_result.get("expected_years", {})
        lease_years = fit_result.get("lease_years", 0)
        legal_useful_life = expected_years.get("legal_useful_life", 0)
        remaining_years_avg = fit_result.get("remaining_years_avg", 0)
        
        col_periods = st.columns([1, 1, 1, 1])
        with col_periods[0]:
            st.metric("リース期間", f"{lease_years:.1f}年", f"({lease_months}ヶ月)")
        
        with col_periods[1]:
            st.metric("法定耐用年数", f"{legal_useful_life:.0f}年")
        
        with col_periods[2]:
            st.metric("残り期間", f"{remaining_years_avg:.1f}年")
        
        with col_periods[3]:
            st.metric("再リーススコア", f"{remanufacture_score:.0f}/100")
        
        # ── リース期間妥当性チェック ────────────────────────────────────────
        lease_check = fit_result.get("lease_period_check", {})
        if lease_check:
            status = lease_check.get("status", "unknown")
            message = lease_check.get("message", "")
            recommended_max = lease_check.get("recommended_max_years", 0)
            
            if status == "over_limit":
                st.error(f"⚠️ リース期間超過: {message}")
            elif status == "near_limit":
                st.warning(f"⚠️ リース期間注意: {message}")
            else:
                st.success(f"✅ リース期間妥当: {message}")
            
            st.caption(f"推奨最大リース期間: {recommended_max:.1f}年")
        
        # ── 期間別スコア表示（Plotly ゲージ） ────────────────────────────────
        fig_gauge = go.Figure(
            data=[
                go.Indicator(
                    mode="gauge+number+delta",
                    value=remanufacture_score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "再リース機会"},
                    delta={"reference": 80, "increasing": {"color": "green"}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#1f77b4"},
                        "steps": [
                            {"range": [0, 50], "color": "#ffcccc"},
                            {"range": [50, 80], "color": "#fff9e6"},
                            {"range": [80, 100], "color": "#ccffcc"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 70,
                        },
                    },
                )
            ]
        )
        fig_gauge.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # ── 推奨メッセージ ────────────────────────────────────────────────────
        recommendation = fit_result.get("recommendation", "")
        if recommendation:
            st.info(f"💡 **推奨**: {recommendation}")
        
        # ── リスク警告 ────────────────────────────────────────────────────────
        if remanufacture_score < 60:
            st.warning(
                "⚠️ **再リース機会が限定的です。**\n\n"
                "以下の点を確認してください：\n"
                "- 法定耐用年数に近づくため残価リスク増大\n"
                "- 再リース時の市場流動性が低下\n"
                "- メンテナンスコストの上昇リスク\n"
                "- 税務上の耐用年数超過リスク"
            )
    elif selected_item:
        st.info("💡 リース期間を入力すると、適合度が表示されます。")
    
    # 結果を返す
    return {
        "category": selected_category,
        "item_name": selected_item,
        "lease_months": lease_months,
        "fit_result": fit_result,
    }


def integrate_usage_period_into_contract(contract_dict: dict, upw_result: dict) -> dict:
    """
    期待使用期間ウィジェットの結果を contract_dict に統合。
    
    Parameters
    ----------
    contract_dict : dict
        既存の契約条件辞書
    upw_result : dict
        期待使用期間ウィジェット結果（render_usage_period_widget の戻り値）
    
    Returns
    -------
    dict
        統合後の contract_dict
    """
    integrated = dict(contract_dict)
    
    # 機種名・カテゴリを追加
    if upw_result.get("item_name"):
        integrated["item_name"] = upw_result["item_name"]
        integrated["asset_name"] = upw_result["item_name"]
    
    # リース期間を上書き（upw_result のものを優先）
    if upw_result.get("lease_months", 0) > 0:
        integrated["lease_months"] = upw_result["lease_months"]
    
    # 適合度情報を追加
    if upw_result.get("fit_result"):
        integrated["usage_period_fit"] = upw_result["fit_result"]
    
    return integrated


if __name__ == "__main__":
    # テストレンダリング
    st.set_page_config(page_title="期待使用期間ウィジェット テスト", layout="wide")
    
    st.title("期待使用期間ウィジェット テスト")
    
    result = render_usage_period_widget(key_prefix="test_upw")
    
    st.divider()
    st.markdown("### 結果")
    st.json(result)
