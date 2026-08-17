import numpy as np
import plotly.graph_objects as go
import streamlit as st

# 計算部は UI 非依存の future_simulation_core.py に置き、FastAPI からも同じ関数を使う。
# ここでは再エクスポートするだけで、Streamlit 側の呼び出し方は変えない。
from future_simulation_core import run_business_simulation  # noqa: F401

def plot_future_simulation_plotly(sim_data: dict, var_type: str = "sales"):
    """
    シミュレーション結果(売上または営業利益)のファンチャートをPlotlyで描画する
    """
    if not sim_data:
        return None
        
    years = sim_data["years"]
    
    if var_type == "sales":
        percentiles = sim_data["sales_percentiles"]
        title_text = "売上高 シミュレーション"
        y_axis_title = "売上高 (千円)"
        color_base = "rgba(44, 160, 44,"  # 緑系
        median_color = "rgb(44, 160, 44)"
    elif var_type == "op_profit":
        percentiles = sim_data["op_percentiles"]
        title_text = "営業利益 シミュレーション"
        y_axis_title = "営業利益 (千円)"
        color_base = "rgba(255, 127, 14," # オレンジ系
        median_color = "rgb(255, 127, 14)"
    else:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([years, years[::-1]]),
        y=np.concatenate([percentiles[90], percentiles[10][::-1]]),
        fill='toself',
        fillcolor=f"{color_base} 0.2)",
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='80%信頼区間 (10-90%)'
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([years, years[::-1]]),
        y=np.concatenate([percentiles[75], percentiles[25][::-1]]),
        fill='toself',
        fillcolor=f"{color_base} 0.4)",
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='50%信頼区間 (25-75%)'
    ))

    fig.add_trace(go.Scatter(
        x=years,
        y=percentiles[50],
        line=dict(color=median_color, width=3),
        mode='lines+markers',
        name='中央値 (50%)',
        hovertemplate='経過年数: %{x}年<br>予測値: %{y:,.0f}千円<extra></extra>'
    ))
    
    if var_type == "op_profit":
        fig.add_hline(y=0, line_dash="dash", line_color="red", name="赤字ライン (¥0)")

    fig.update_layout(
        title=title_text,
        xaxis_title="経過年数 (年)",
        yaxis_title=y_axis_title,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig

def render_future_simulation_ui(res: dict):
    """
    ダッシュボードに組み込む将来シミュレーションのUIコンポーネント
    """
    fin = res.get("financials", {})
    current_sales = fin.get("nenshu", 0)
    current_op = fin.get("rieki", 0)
    
    st.markdown("現在の財務データを起点とし、将来5年間の「売上高」と「営業利益」の推移をモンテカルロ法（幾何ブラウン運動）を用いて確率的にシミュレーションします。")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        drift_pct = st.number_input(
            "期待売上成長率 (年率 %)", 
            min_value=-20.0, max_value=30.0, value=1.0, step=0.5,
            help="毎年見込まれる平均的な売上の成長率です。",
            key="sim_drift_pct"
        )
    with col2:
        vol_pct = st.number_input(
            "業績のボラティリティ (年率 %)", 
            min_value=1.0, max_value=50.0, value=15.0, step=1.0,
            help="業績のブレ幅を表します。建設業などは比較的高め(15〜20%)、インフラ系は低め(5〜10%)に設定します。",
            key="sim_vol_pct"
        )
    with col3:
        sim_years = st.slider(
            "シミュレーション期間 (年)", 
            min_value=3, max_value=10, value=5, step=1,
            help="将来何年間を予測するかを設定します。",
            key="sim_years"
        )
        
    if st.button("▶ シミュレーション実行", key="btn_run_future_sim", type="primary"):
        with st.spinner("モンテカルロ・シミュレーションを実行中 (10,000回)..."):
            drift = drift_pct / 100.0
            volatility = vol_pct / 100.0
            
            sim_data = run_business_simulation(
                current_sales=current_sales,
                current_op_profit=current_op,
                drift=drift,
                volatility=volatility,
                years=sim_years,
                n_simulations=10000
            )
            
            if sim_data:
                st.divider()
                st.subheader(f"📊 シミュレーション結果 ({sim_years}年後)")
                
                deficit_prob = sim_data["deficit_prob"]
                final_median = sim_data["final_op_median"]
                final_worst10 = sim_data["final_op_worst10"]
                
                m1, m2, m3 = st.columns(3)
                m1.metric(f"{sim_years}年後の 営業赤字確率", f"{deficit_prob:.1%}", 
                          delta="高リスク" if deficit_prob > 0.3 else "低リスク", delta_color="inverse")
                m2.metric(f"{sim_years}年後の 営業利益 (中央値)", f"{final_median/1000:,.1f} 百万円")
                m3.metric(f"{sim_years}年後の 営業利益 (ワースト10%)", f"{final_worst10/1000:,.1f} 百万円",
                          delta="赤字転落" if final_worst10 < 0 else "黒字維持", delta_color="inverse" if final_worst10 < 0 else "normal")
                
                fig_sales = plot_future_simulation_plotly(sim_data, var_type="sales")
                fig_op = plot_future_simulation_plotly(sim_data, var_type="op_profit")
                
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    if fig_sales:
                        st.plotly_chart(fig_sales, width='stretch', key="fig_future_sales")
                with chart_col2:
                    if fig_op:
                        st.plotly_chart(fig_op, width='stretch', key="fig_future_op")
                        
                st.info("💡 **見方**: 帯の幅が広いほど将来の予測が不確実であることを示します。特に「営業利益」において、赤い破線（0円）より下に帯が多くかかっている場合、将来的に返済原資が不足するリスク（デフォルトリスク）が高まることを意味します。限界利益率や固定費などは現在の業績をもとに簡易推定しています。")
            else:
                st.error("現在の売上高が0以下のため、シミュレーションを実行できません。")
