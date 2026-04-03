# -*- coding: utf-8 -*-
"""
components/timeseries_forecast_view.py
=======================================
TimesFM 時系列予測専用 Streamlit UI（4タブ）

タブ1: 📊 個社スコア予測
タブ2: 🏭 業種トレンド
タブ3: 💴 金利予測
タブ4: ⚙️ GBM vs TimesFM 比較
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from timesfm_engine import (
    TIMESFM_AVAILABLE,
    forecast_company_score,
    forecast_industry_trend,
    forecast_final_rate,
    forecast_financial_paths,
)
from data_cases import load_all_cases


# ── ユーティリティ ─────────────────────────────────────────────────────────────

def _availability_badge() -> None:
    if TIMESFM_AVAILABLE:
        st.success("TimesFM: 利用可能", icon="✅")
    else:
        st.warning("TimesFM が未インストールです — GBM / 線形外挿でフォールバック中", icon="⚠️")


def _fan_chart(
    x_labels: list,
    forecast: list[float],
    band_low: list[float],
    band_high: list[float],
    history: list[float] | None = None,
    history_labels: list | None = None,
    title: str = "",
    y_label: str = "",
) -> go.Figure:
    """汎用ファンチャート（Plotly）"""
    fig = go.Figure()

    # 過去履歴
    if history and history_labels:
        fig.add_trace(go.Scatter(
            x=history_labels,
            y=history,
            mode="lines+markers",
            name="実績",
            line=dict(color="#4C9BE8", width=2),
        ))

    # 不確実性帯
    x_all = x_labels + x_labels[::-1]
    y_all = band_high + band_low[::-1]
    fig.add_trace(go.Scatter(
        x=x_all,
        y=y_all,
        fill="toself",
        fillcolor="rgba(255, 180, 50, 0.25)",
        line=dict(color="rgba(255,255,255,0)"),
        name="予測帯 (±1σ)",
        showlegend=True,
    ))

    # 中央予測
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=forecast,
        mode="lines+markers",
        name="予測 (中央値)",
        line=dict(color="#FFB432", width=2, dash="dash"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="期間",
        yaxis_title=y_label,
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.2),
        height=380,
    )
    return fig


# ── タブ1: 個社スコア予測 ──────────────────────────────────────────────────────

def _tab_company_score(all_cases: list[dict]) -> None:
    st.subheader("📊 個社スコア予測")
    st.caption("過去案件の審査スコア時系列から将来スコアを予測します。")

    # 企業名リスト
    names = sorted({c.get("company_name") or c.get("inputs", {}).get("company_name", "") for c in all_cases} - {""})
    if not names:
        st.info("DBに企業名データがありません。")
        return

    selected = st.selectbox("企業を選択", names)
    horizon = st.slider("予測月数", 3, 24, 12, key="cmp_horizon")

    company_cases = sorted(
        [c for c in all_cases if (c.get("company_name") or c.get("inputs", {}).get("company_name", "")) == selected],
        key=lambda x: x.get("timestamp", ""),
    )
    if not company_cases:
        st.warning("選択した企業の案件が見つかりません。")
        return

    st.caption(f"過去案件数: **{len(company_cases)}** 件")

    result = forecast_company_score(company_cases, horizon_months=horizon)

    if "error" in result:
        st.error(result["error"])
        return

    method_label = "TimesFM" if result["method"] == "timesfm" else "線形外挿（フォールバック）"
    st.caption(f"予測手法: **{method_label}**")

    # トレンド表示
    trend = result["trend"]
    trend_icon = {"up": "📈 上昇", "down": "📉 下降", "stable": "➡️ 横ばい"}.get(trend, trend)
    st.metric("スコアトレンド", trend_icon)

    # グラフ
    hist = result["score_history"]
    fore = result["score_forecast"]
    hist_labels = [f"T-{len(hist)-i}" for i in range(len(hist))]
    fore_labels = [f"M+{i+1}" for i in range(len(fore))]

    fig = _fan_chart(
        x_labels=fore_labels,
        forecast=fore,
        band_low=result["band_low"],
        band_high=result["band_high"],
        history=hist,
        history_labels=hist_labels,
        title=f"{selected} — スコア予測 ({horizon}ヶ月)",
        y_label="審査スコア",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("数値データ"):
        import pandas as pd
        df = pd.DataFrame({
            "月": fore_labels,
            "予測スコア": [round(v, 1) for v in fore],
            "下限 (-1σ)": [round(v, 1) for v in result["band_low"]],
            "上限 (+1σ)": [round(v, 1) for v in result["band_high"]],
        })
        st.dataframe(df, use_container_width=True)


# ── タブ2: 業種トレンド ────────────────────────────────────────────────────────

def _tab_industry_trend(all_cases: list[dict]) -> None:
    st.subheader("🏭 業種トレンド")
    st.caption("業種内の全案件スコアを月次集計し、業種全体のトレンドを予測します。")

    industries = sorted({(c.get("industry_sub") or "")[:6] for c in all_cases} - {""})
    if not industries:
        st.info("業種データがありません。")
        return

    selected_ind = st.selectbox("業種を選択", industries)
    horizon = st.slider("予測月数", 6, 36, 24, key="ind_horizon")

    result = forecast_industry_trend(selected_ind, all_cases, horizon_months=horizon)

    if "error" in result:
        st.error(result["error"])
        return

    method_label = "TimesFM" if result["method"] == "timesfm" else "線形外挿（フォールバック）"
    st.caption(f"予測手法: **{method_label}**")

    signal = result["risk_signal"]
    signal_map = {"positive": "🟢 改善傾向", "neutral": "🟡 横ばい", "negative": "🔴 悪化傾向"}
    st.metric("業種リスクシグナル", signal_map.get(signal, signal))

    fig = _fan_chart(
        x_labels=result["months_forecast"],
        forecast=result["avg_score_fore"],
        band_low=[max(0.0, v - 3) for v in result["avg_score_fore"]],
        band_high=[min(100.0, v + 3) for v in result["avg_score_fore"]],
        history=result["avg_score_hist"],
        history_labels=result["months_history"],
        title=f"業種「{selected_ind}」スコアトレンド予測",
        y_label="月次平均スコア",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("過去データ"):
        import pandas as pd
        df_hist = pd.DataFrame({
            "月": result["months_history"],
            "平均スコア": [round(v, 1) for v in result["avg_score_hist"]],
        })
        st.dataframe(df_hist, use_container_width=True)


# ── タブ3: 金利予測 ────────────────────────────────────────────────────────────

def _tab_rate_forecast(all_cases: list[dict]) -> None:
    st.subheader("💴 成約金利予測")
    st.caption("成約案件の金利履歴から将来の適正金利帯を予測します。")

    industries = ["（全業種）"] + sorted({(c.get("industry_sub") or "")[:6] for c in all_cases} - {""})
    selected_ind = st.selectbox("業種フィルタ", industries, key="rate_ind")
    horizon = st.slider("予測月数", 3, 12, 6, key="rate_horizon")

    ind_filter = "" if selected_ind == "（全業種）" else selected_ind
    result = forecast_final_rate(all_cases, industry=ind_filter, horizon_months=horizon)

    if "error" in result:
        st.error(result["error"])
        return

    method_label = "TimesFM" if result["method"] == "timesfm" else "直近平均（フォールバック）"
    st.caption(f"予測手法: **{method_label}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("将来中央金利", f"{result['rate_forecast']:.2f}%")
    col2.metric("予測下限", f"{result['band_low']:.2f}%")
    col3.metric("予測上限", f"{result['band_high']:.2f}%")

    # 履歴 + 予測ライン
    hist = result["rate_history"]
    fore = result.get("horizon_forecast", [result["rate_forecast"]] * horizon)
    hist_labels = [f"T-{len(hist)-i}" for i in range(len(hist))]
    fore_labels = [f"M+{i+1}" for i in range(len(fore))]

    fig = _fan_chart(
        x_labels=fore_labels,
        forecast=fore,
        band_low=[result["band_low"]] * len(fore),
        band_high=[result["band_high"]] * len(fore),
        history=hist,
        history_labels=hist_labels,
        title=f"成約金利予測（{selected_ind}）",
        y_label="金利 (%)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── タブ4: GBM vs TimesFM 比較 ────────────────────────────────────────────────

def _tab_comparison(all_cases: list[dict]) -> None:
    st.subheader("⚙️ GBM vs TimesFM 比較")
    st.caption("同一の売上履歴データに対して、GBM と TimesFM の予測を並べて比較します。")

    names = sorted({c.get("company_name") or c.get("inputs", {}).get("company_name", "") for c in all_cases} - {""})
    if not names:
        st.info("DBに企業名データがありません。")
        return

    selected = st.selectbox("企業を選択", names, key="cmp2_name")
    n_periods = st.slider("予測期間（月）", 6, 36, 12, key="cmp2_horizon")

    company_cases = sorted(
        [c for c in all_cases if (c.get("company_name") or c.get("inputs", {}).get("company_name", "")) == selected],
        key=lambda x: x.get("timestamp", ""),
    )

    # 売上履歴を抽出
    revenues: list[float] = []
    for c in company_cases:
        inp = c.get("inputs") or {}
        if isinstance(inp, str):
            import json
            try:
                inp = json.loads(inp)
            except Exception:
                inp = {}
        v = inp.get("nenshu") or inp.get("revenue")
        if v:
            try:
                revenues.append(float(v))
            except Exception:
                pass

    if not revenues:
        st.info("売上履歴データが取得できませんでした。現在売上を 1,000 万円として比較します。")
        revenues = [10_000_000.0]

    S0 = revenues[-1]
    st.caption(f"直近売上高: **{S0:,.0f} 円** / 履歴 {len(revenues)} 件")

    col_l, col_r = st.columns(2)

    # GBM パス（常に実行）
    gbm_paths = forecast_financial_paths(
        historical_values=revenues,
        n_periods=n_periods,
        n_paths=200,
        fallback_mu=0.02,
        fallback_sigma=0.15,
    )
    x_labels = [f"M+{i}" for i in range(n_periods + 1)]

    with col_l:
        st.markdown("**GBM（幾何ブラウン運動）**")
        fig_gbm = go.Figure()
        for path in gbm_paths[:50]:
            fig_gbm.add_trace(go.Scatter(
                x=x_labels, y=path,
                mode="lines",
                line=dict(color="rgba(76,155,232,0.15)", width=1),
                showlegend=False,
            ))
        fig_gbm.add_trace(go.Scatter(
            x=x_labels,
            y=np.median(gbm_paths, axis=0),
            mode="lines",
            line=dict(color="#4C9BE8", width=2),
            name="中央値",
        ))
        fig_gbm.update_layout(
            template="plotly_dark", height=300, showlegend=False,
            yaxis_title="売上高 (円)", xaxis_title="期間",
        )
        st.plotly_chart(fig_gbm, use_container_width=True)
        gbm_med = float(np.median(gbm_paths[:, -1]))
        st.metric("最終月 中央値", f"{gbm_med:,.0f} 円")

    with col_r:
        if TIMESFM_AVAILABLE:
            st.markdown("**TimesFM（基盤モデル予測）**")
            tfm_paths = forecast_financial_paths(
                historical_values=revenues,
                n_periods=n_periods,
                n_paths=200,
                fallback_mu=0.02,
                fallback_sigma=0.15,
            )
            fig_tfm = go.Figure()
            for path in tfm_paths[:50]:
                fig_tfm.add_trace(go.Scatter(
                    x=x_labels, y=path,
                    mode="lines",
                    line=dict(color="rgba(255,180,50,0.15)", width=1),
                    showlegend=False,
                ))
            fig_tfm.add_trace(go.Scatter(
                x=x_labels,
                y=np.median(tfm_paths, axis=0),
                mode="lines",
                line=dict(color="#FFB432", width=2),
                name="中央値",
            ))
            fig_tfm.update_layout(
                template="plotly_dark", height=300, showlegend=False,
                yaxis_title="売上高 (円)", xaxis_title="期間",
            )
            st.plotly_chart(fig_tfm, use_container_width=True)
            tfm_med = float(np.median(tfm_paths[:, -1]))
            st.metric("最終月 中央値", f"{tfm_med:,.0f} 円")
            diff_pct = (tfm_med - gbm_med) / max(abs(gbm_med), 1) * 100
            st.caption(f"差分: {diff_pct:+.1f}%（TimesFM vs GBM）")
        else:
            st.info("TimesFM が未インストールのため比較できません。\n\n`pip install timesfm` を実行後に再起動してください。")


# ── メインエントリポイント ─────────────────────────────────────────────────────

def render_timeseries_forecast() -> None:
    st.title("📈 TimesFM 時系列予測")
    _availability_badge()
    st.divider()

    all_cases = load_all_cases()
    if not all_cases:
        st.warning("審査データが見つかりません。先に案件を登録してください。")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 個社スコア予測",
        "🏭 業種トレンド",
        "💴 金利予測",
        "⚙️ GBM vs TimesFM 比較",
    ])

    with tab1:
        _tab_company_score(all_cases)
    with tab2:
        _tab_industry_trend(all_cases)
    with tab3:
        _tab_rate_forecast(all_cases)
    with tab4:
        _tab_comparison(all_cases)
