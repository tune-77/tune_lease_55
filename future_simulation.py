import numpy as np
import plotly.graph_objects as go
import streamlit as st

# TimesFM（オプション）
try:
    from timesfm_engine import forecast_financial_paths as _tfm_paths, TIMESFM_AVAILABLE as _TIMESFM_AVAILABLE
except ImportError:
    _tfm_paths = None
    _TIMESFM_AVAILABLE = False

def run_business_simulation(
    current_sales: float,
    current_op_profit: float,
    drift: float,
    volatility: float,
    years: int = 5,
    n_simulations: int = 10000,
    use_timesfm: bool = False,
    sales_history: list | None = None,
):
    """
    GBM（幾何ブラウン運動）または TimesFM を用いて将来の売上高と営業利益をシミュレーションする関数

    Args:
        current_sales (float): 現在の売上高
        current_op_profit (float): 現在の営業利益
        drift (float): 年間期待成長率 (例: 0.02)
        volatility (float): 年間ボラティリティ (例: 0.10)
        years (int): シミュレーション期間(年)
        n_simulations (int): シミュレーション回数
        use_timesfm (bool): True のとき TimesFM で売上パスを生成（未インストール時は GBM へフォールバック）
        sales_history (list | None): TimesFM 用の過去売上時系列（月次または年次）

    Returns:
        dict: 売上高・営業利益のパーセンタイルデータや赤字確率などを含むメタデータ
    """

    # 簡易的な固定費・変動費の算出 (限界利益率と固定費を推計)
    if current_sales <= 0:
        return None

    # 仮置き: 限界利益率を 30% とみなす (業界平均から設定する等の拡張も可能)
    current_op_margin = current_op_profit / current_sales
    contribution_margin_ratio = max(0.30, current_op_margin + 0.10)

    # 固定費 = 売上高 × 限界利益率 - 営業利益
    fixed_costs = current_sales * contribution_margin_ratio - current_op_profit

    n_periods = years  # 年単位

    # ── 売上パス生成 ─────────────────────────────────────────────────────────────
    if use_timesfm and _TIMESFM_AVAILABLE and _tfm_paths is not None:
        historical = sales_history if sales_history else [current_sales]
        sales_paths = _tfm_paths(
            historical_values=historical,
            n_periods=n_periods,
            n_paths=n_simulations,
            fallback_mu=drift,
            fallback_sigma=volatility,
            dt=1.0,
        )
        method = "timesfm"
    else:
        dt = 1.0  # 年単位
        np.random.seed(42)
        Z = np.random.normal(0, 1, size=(n_simulations, years))
        sales_paths = np.zeros((n_simulations, years + 1))
        sales_paths[:, 0] = current_sales
        for t in range(1, years + 1):
            sales_paths[:, t] = sales_paths[:, t - 1] * np.exp(
                (drift - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * Z[:, t - 1]
            )
        method = "gbm"

    op_profit_paths = sales_paths * contribution_margin_ratio - fixed_costs

    percentiles = [10, 25, 50, 75, 90]

    sales_percentiles = {p: np.percentile(sales_paths, p, axis=0) for p in percentiles}
    op_percentiles = {p: np.percentile(op_profit_paths, p, axis=0) for p in percentiles}

    final_op_profits = op_profit_paths[:, -1]
    deficit_prob = float(np.mean(final_op_profits < 0))

    return {
        "years": np.arange(0, years + 1),
        "sales_percentiles": sales_percentiles,
        "op_percentiles": op_percentiles,
        "deficit_prob": deficit_prob,
        "final_op_median": op_percentiles[50][-1],
        "final_op_worst10": op_percentiles[10][-1],
        "method": method,
    }

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
