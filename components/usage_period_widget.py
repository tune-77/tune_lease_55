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
        
        # ── 基本情報の取得 ────────────────────────────────────────────────────
        remanufacture_score = fit_result.get("remanufacture_score", 50)
        assessment_label = fit_result.get("assessment_label", "未判定")
        expected_years = fit_result.get("expected_years", {})
        lease_years = fit_result.get("lease_years", 0)
        legal_useful_life = expected_years.get("legal_useful_life", 0)
        remaining_years_avg = fit_result.get("remaining_years_avg", 0)
        lease_check = fit_result.get("lease_period_check", {})
        
        # ── スコア表示（大きく目立つように） ──────────────────────────────────
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
        
        st.markdown(
            f"""
            <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {text_color};">
                <h3 style="margin: 0; color: {text_color};">{score_color} 再リース機会スコア</h3>
                <h1 style="margin: 10px 0; color: {text_color}; font-size: 48px;">{remanufacture_score:.0f}</h1>
                <p style="margin: 5px 0; color: {text_color}; font-size: 18px;"><strong>{assessment_label}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # ── 法定耐用年数とリース期間の関係（税務ルール） ─────────────────────
        st.markdown("#### 📋 法定耐用年数とリース期間の税務ルール")
        
        # 法定下限の計算式を表示（これ以上ならOK）
        if legal_useful_life < 10:
            rule = "70%"
            min_limit = legal_useful_life * 0.7
            formula = f"法定耐用年数 {legal_useful_life} 年 × {rule} = {min_limit:.1f}年"
        else:
            rule = "60%"
            min_limit = legal_useful_life * 0.6
            formula = f"法定耐用年数 {legal_useful_life} 年 × {rule} = {min_limit:.1f}年"
        
        st.markdown(f"**法定下限の計算**: {formula}")
        st.markdown("💡 **この数字以上のリース期間を設定すれば、課税上の問題がありません**")
        
        # ── 期間比較テーブル ──────────────────────────────────────────────────
        st.markdown("#### ⏱️ 期間比較")
        
        legal_min = lease_check.get('legal_min_years', 0)
        comparison_data = {
            "項目": [
                "法定耐用年数",
                f"法定下限（×{rule}）",
                "現在のリース期間",
                "残り期間"
            ],
            "年数": [
                f"{legal_useful_life:.1f}年",
                f"{legal_min:.1f}年",
                f"{lease_years:.1f}年",
                f"{remaining_years_avg:.1f}年"
            ],
            "評価": [
                "—",
                "これ以上ならOK",
                "✅" if lease_years >= legal_min else "⚠️",
                "再リース可能期間"
            ]
        }
        
        col_table = st.columns([2, 1.5, 1])
        with col_table[0]:
            st.write("**項目**")
            for item in comparison_data["項目"]:
                st.write(item)
        
        with col_table[1]:
            st.write("**年数**")
            for val in comparison_data["年数"]:
                st.write(val)
        
        with col_table[2]:
            st.write("**評価**")
            for val in comparison_data["評価"]:
                st.write(val)
        
        # ── 視覚的タイムライン ────────────────────────────────────────────────
        st.markdown("#### 📊 タイムライン（年数）")
        
        # プログレスバー表示（法定下限から法定耐用年数までの進捗）
        legal_min = lease_check.get('legal_min_years', legal_useful_life * 0.7 if legal_useful_life < 10 else legal_useful_life * 0.6)
        # リース期間が法定下限からどれくらい離れているかを表示
        progress_range = legal_useful_life - legal_min
        progress_filled = lease_years - legal_min
        progress_ratio = min(max(progress_filled / progress_range, 0), 1.0) if progress_range > 0 else 0
        
        col_timeline = st.columns([10])
        with col_timeline[0]:
            # HTMLでプログレスバーを作成
            st.markdown(
                f"""
                <div style="width: 100%; background-color: #e9ecef; border-radius: 25px; height: 30px; overflow: hidden; margin: 20px 0;">
                    <div style="width: {progress_ratio * 100:.1f}%; background: linear-gradient(90deg, #28a745 0%, #ffc107 50%, #dc3545 100%); height: 100%; display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; color: white; font-weight: bold;">
                        {progress_ratio * 100:.0f}%
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 5px;">
                    <span style="font-weight: bold;">{legal_min:.1f}年（法定下限）</span>
                    <span style="color: blue; font-weight: bold;">{lease_years:.1f}年（現在のリース期間）</span>
                    <span style="font-weight: bold;">{legal_useful_life:.1f}年（法定耐用年数）</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # ── リース期間妥当性チェック（強調版） ────────────────────────────────
        st.markdown("#### ✅ リース期間のチェック結果")
        
        status = lease_check.get("status", "unknown")
        message = lease_check.get("message", "")
        
        if status == "under_limit":
            st.error(
                f"🚨 **リース期間が法定下限未満です（NG）**\n\n"
                f"{message}\n\n"
                f"**対応**: 法定下限({legal_min:.1f}年)以上にリース期間を延長してください。"
            )
        elif status == "near_limit":
            st.warning(
                f"⚠️ **リース期間が法定下限に接近しています**\n\n"
                f"{message}\n\n"
                f"**注意**: 安全圏を確保するため、より長いリース期間を検討してください。"
            )
        else:
            st.success(f"✅ **リース期間は法定下限以上で、課税上安全です**\n\n{message}")
        
        # ── 期間別スコア表示（Plotly ゲージ） ────────────────────────────────
        st.markdown("#### 📈 再リース機会度の詳細評価")
        
        fig_gauge = go.Figure(
            data=[
                go.Indicator(
                    mode="gauge+number",
                    value=remanufacture_score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "スコア"},
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
        fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # ── スコア判定基準 ────────────────────────────────────────────────────
        st.markdown("##### スコア判定基準")
        
        score_criteria = {
            "🟢 優秀（85-100）": "残り期間4年以上。充分な再リース収益機会がある。",
            "🟡 良好（70-84）": "残り期間2-4年。中程度の再リース機会がある。",
            "🟠 標準（50-69）": "残り期間1-2年。再リース機会は限定的。",
            "🔴 要注意（30-49）": "残り期間0-1年。再リース困難。残価回収リスク増。",
            "🔴 リスク高（0-29）": "残り期間が負。法定耐用年数超過。再リース不可。",
        }
        
        for label, description in score_criteria.items():
            st.caption(f"**{label}**: {description}")
        
        # ── 推奨メッセージ ────────────────────────────────────────────────────
        st.markdown("#### 💡 リース会社向け推奨事項")
        
        recommendation = fit_result.get("recommendation", "")
        if recommendation:
            st.info(f"{recommendation}")
        
        # ── 重大リスク警告 ────────────────────────────────────────────────────
        if remanufacture_score < 60:
            st.error(
                "### ❌ 高リスク案件です\n\n"
                "**以下の点を慎重に検討してください:**\n"
                "- 🔴 **残価リスク増大**: 法定耐用年数に接近するため、残価評価が大きく下がる\n"
                "- 🔴 **再リース困難**: 市場流動性が低下し、再リース時の利益が限定的\n"
                "- 🔴 **メンテコスト上昇**: 設備寿命が残り少なく、故障リスク増加\n"
                "- 🔴 **税務リスク**: 法定耐用年数超過の可能性。税務当局の審査対象\n\n"
                "**対応案:**\n"
                "- リース期間を短縮することを検討\n"
                "- 買取オプションを組み込むことを検討\n"
                "- 残価保証オプションの導入\n"
                "- 金利・リース料金での上乗せ検討"
            )
        elif remanufacture_score < 70:
            st.warning(
                "### ⚠️ 注意が必要な案件です\n\n"
                "**確認項目:**\n"
                "- 残り期間が限定的なため、残価設定に注意\n"
                "- 再リース需要の事前調査が重要\n"
                "- メンテナンス計画の確認が必要"
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
