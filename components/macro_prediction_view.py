import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import os
from runtime_paths import REPO_ROOT

def render_macro_prediction_view():
    st.title("📈 マクロ経済・S&P500予測 (β)")
    st.caption("グローバル金融指標（S&P500, VIX, 為替）の推移とリスクシミュレーション")

    tab_linear, tab_lstm = st.tabs([
        "💵 円建てS&P500 & 為替シミュレーション",
        "🔮 LSTM ディープラーニング予測 (VIX連動)"
    ])

    with tab_linear:
        st.subheader("円建て為替連動分析")
        st.caption("S&P 500 にドル円為替の変動インパクトを反映します")

        import yfinance as yf
        col1, col2 = st.columns(2)
        with col1:
            sim_rate = st.slider("円高シミュレーション (1ドル = X円)", 100.0, 160.0, 130.0, 0.5)
        with col2:
            period = st.selectbox("データ期間", ["1y", "2y", "5y"], index=1)

        if st.button("為替連動シミュレーションを実行", key="run_linear_sim"):
            with st.spinner("データを取得中..."):
                try:
                    sp500_df = yf.download("^GSPC", period=period, interval="1d", progress=False)
                    usdjpy_df = yf.download("JPY=X", period=period, interval="1d", progress=False)

                    if isinstance(sp500_df.columns, pd.MultiIndex):
                        sp500 = sp500_df['Close'].iloc[:, 0]
                    else:
                        sp500 = sp500_df['Close']
                        
                    if isinstance(usdjpy_df.columns, pd.MultiIndex):
                        usdjpy = usdjpy_df['Close'].iloc[:, 0]
                    else:
                        usdjpy = usdjpy_df['Close']

                    data = pd.DataFrame({'SP500_USD': sp500, 'USDJPY': usdjpy}).dropna()
                    data['SP500_JPY'] = data['SP500_USD'] * data['USDJPY']

                    current_sp500 = data['SP500_USD'].iloc[-1]
                    current_jpy_value = data['SP500_JPY'].iloc[-1]
                    simulated_value = current_sp500 * sim_rate
                    drop_rate = (simulated_value - current_jpy_value) / current_jpy_value * 100

                    # KPI
                    c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
                    c_kpi1.metric("現在の円建て価格", f"{int(current_jpy_value):,} 円")
                    c_kpi2.metric(f"為替が {sim_rate}円 になった場合", f"{int(simulated_value):,} 円")
                    c_kpi3.metric("影響度（下落率）", f"{drop_rate:.2f}%", delta=f"{drop_rate:.1f}%", delta_color="inverse")

                    # Plotly グラフ
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['SP500_JPY'], mode='lines', name='実績値 (円建て)', line=dict(color='#22c55e')))
                    fig.update_layout(
                        title="S&P 500 円建て価格推移",
                        xaxis_title="日付",
                        yaxis_title="価格 (円)",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"データの取得に失敗しました: {e}")

    with tab_lstm:
        st.subheader("LSTM 予測 × VIX 指数")
        st.caption("ディープラーニング（LSTM）によるS&P 500 と市場恐怖指数（VIX）のトレンド先読み")

        vix_model_path = REPO_ROOT.parent / "sp500-vix-predictor" / "predict.py"
        if not os.path.exists(vix_model_path):
            st.error(f"LSTM予測モジュールが見つかりません: {vix_model_path}")
        else:
            import yfinance as yf
            if st.button("VIX 恐怖指数の現在値を取得", key="run_vix_fetch"):
                with st.spinner("VIX データを取得中..."):
                    try:
                        vix_data = yf.download("^VIX", period="6mo", interval="1d", progress=False)
                        if isinstance(vix_data.columns, pd.MultiIndex):
                            vix_close = vix_data['Close'].iloc[:, 0]
                        else:
                            vix_close = vix_data['Close']
                        
                        current_vix = vix_close.iloc[-1]
                        
                        vix_level = "🟢 正常" if current_vix < 20 else "🟡 警戒" if current_vix < 30 else "🔴 パニック（高リスク）"
                        st.metric("現在の VIX 指数", f"{current_vix:.2f}", help="20以下=安定, 30以上=市場パニック")
                        st.markdown(f"**判定結果**: {vix_level}")

                        fig_vix = go.Figure()
                        fig_vix.add_trace(go.Scatter(x=vix_close.index, y=vix_close.values, mode='lines', name='VIX指数', line=dict(color='#ef4444')))
                        fig_vix.update_layout(
                            title="VIX 指数 過去6ヶ月の推移",
                            xaxis_title="日付",
                            yaxis_title="VIX",
                            template="plotly_white",
                            height=400
                        )
                        st.plotly_chart(fig_vix, use_container_width=True)
                    except Exception as e:
                        st.error(f"VIX取得失敗: {e}")
