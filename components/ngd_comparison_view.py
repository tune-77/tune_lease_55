"""
Phase 2: 自然勾配最適化 A/B比較 UI
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


def render_ngd_comparison() -> None:
    st.title("📐 自然勾配最適化 A/B 比較 — Phase 2")
    st.caption(
        "Fisher情報行列を用いた自然勾配降下法(NGD)と通常の勾配降下法(GD)を同条件で比較する。"
        "NGDはパラメータ空間の曲率を補正するため、希少業種（データが少ない業種）で有利になりやすい。"
    )

    with st.expander("⚙️ パラメータ設定", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            coeff_key = st.selectbox("係数セット", ["全体_既存先", "全体_新規先"], key="ngd_coeff")
            n_steps = st.slider("ステップ数", 50, 500, 200, 50, key="ngd_steps")
        with col2:
            gd_lr = st.number_input("GD 学習率", 0.001, 1.0, 0.05, 0.005, format="%.3f", key="ngd_gd_lr")
            ngd_lr = st.number_input("NGD 学習率", 0.1, 5.0, 0.5, 0.1, format="%.1f", key="ngd_ngd_lr")
        with col3:
            l2 = st.number_input("L2正則化", 1e-5, 0.1, 1e-3, 1e-4, format="%.5f", key="ngd_l2")
            by_industry = st.checkbox("業種別分析", value=True, key="ngd_by_ind")

    if st.button("▶ 比較実行", type="primary", key="ngd_run"):
        with st.spinner("GD / NGD を実行中…"):
            from natural_gradient_optimizer import run_ab_comparison
            result = run_ab_comparison(
                coeff_key=coeff_key,
                n_steps=n_steps,
                gd_lr=gd_lr,
                ngd_lr=ngd_lr,
                l2=l2,
                by_industry=by_industry,
            )

        if "error" in result:
            st.error(result["error"])
            return

        st.session_state["ngd_result"] = result

    result = st.session_state.get("ngd_result")
    if not result:
        st.info("「▶ 比較実行」ボタンを押して比較を開始してください。")
        return

    # ── サマリー ────────────────────────────────────────────────────────────
    gd = result["gd"]
    ngd = result["ngd"]
    delta = result["auc_delta"]
    winner = result["winner"]

    winner_label = {"ngd": "🏆 NGD 優勢", "gd": "🏆 GD 優勢", "tie": "🤝 同等"}[winner]
    winner_color = {"ngd": "green", "gd": "blue", "tie": "gray"}[winner]

    st.markdown(f"### {winner_label}  `ΔAUC = {delta:+.4f}`")

    col1, col2, col3 = st.columns(3)
    col1.metric("GD 最終AUC", f"{gd['final_auc']:.4f}")
    col2.metric("NGD 最終AUC", f"{ngd['final_auc']:.4f}", f"{delta:+.4f}")
    col3.metric("データ件数", f"{result['n_cases']} 件 / {result['n_features']} 係数")

    # ── 収束曲線 ────────────────────────────────────────────────────────────
    st.subheader("収束曲線")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["AUC", "損失 (BCE)"])
    steps = list(range(1, len(gd["auc_hist"]) + 1))

    fig.add_scatter(x=steps, y=gd["auc_hist"], name="GD (AUC)", line_color="royalblue", row=1, col=1)
    fig.add_scatter(x=steps, y=ngd["auc_hist"], name="NGD (AUC)", line_color="crimson", row=1, col=1)
    fig.add_scatter(x=steps, y=gd["loss_hist"], name="GD (損失)", line_color="royalblue",
                    line_dash="dot", showlegend=False, row=1, col=2)
    fig.add_scatter(x=steps, y=ngd["loss_hist"], name="NGD (損失)", line_color="crimson",
                    line_dash="dot", showlegend=False, row=1, col=2)

    fig.update_layout(height=380, margin=dict(t=40, b=20))
    fig.update_xaxes(title_text="ステップ")
    fig.update_yaxes(title_text="AUC", row=1, col=1)
    fig.update_yaxes(title_text="損失", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)

    # ── 係数変化量（GD vs NGD の差） ─────────────────────────────────────────
    st.subheader("係数の変化量（最終値 − 初期値）")
    from data_cases import get_effective_coeffs
    coeffs_init = get_effective_coeffs(result.get("coeff_keys", []) and coeff_key)
    keys = result["coeff_keys"]
    w_init_arr = np.array([coeffs_init.get(k, 0.0) for k in keys])
    gd_delta = gd["w"] - w_init_arr
    ngd_delta = ngd["w"] - w_init_arr

    top_n = 15
    idx = np.argsort(np.abs(gd_delta) + np.abs(ngd_delta))[::-1][:top_n]
    fig2 = go.Figure()
    fig2.add_bar(x=[keys[i] for i in idx], y=[gd_delta[i] for i in idx], name="GD Δ", marker_color="royalblue")
    fig2.add_bar(x=[keys[i] for i in idx], y=[ngd_delta[i] for i in idx], name="NGD Δ", marker_color="crimson")
    fig2.update_layout(
        barmode="group", height=360, margin=dict(t=20, b=80),
        xaxis_tickangle=-40, yaxis_title="Δ係数",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── 業種別 ──────────────────────────────────────────────────────────────
    if result.get("by_industry"):
        st.subheader("業種別 AUC 比較")
        ind_data = result["by_industry"]
        rows = [(ind, v) for ind, v in ind_data.items() if not v.get("skip")]
        if rows:
            inds = [r[0] for r in rows]
            gd_aucs = [r[1]["gd_auc"] for r in rows]
            ngd_aucs = [r[1]["ngd_auc"] for r in rows]
            ns = [r[1]["n"] for r in rows]
            deltas_ind = [r[1]["delta"] for r in rows]

            fig3 = go.Figure()
            fig3.add_bar(x=inds, y=gd_aucs, name="GD", marker_color="royalblue")
            fig3.add_bar(x=inds, y=ngd_aucs, name="NGD", marker_color="crimson")
            fig3.update_layout(
                barmode="group", height=380, margin=dict(t=20, b=80),
                xaxis_tickangle=-30, yaxis_title="AUC", yaxis_range=[0.4, 1.0],
            )
            for i, (ind, n, d) in enumerate(zip(inds, ns, deltas_ind)):
                fig3.add_annotation(
                    x=ind, y=max(gd_aucs[i], ngd_aucs[i]) + 0.02,
                    text=f"n={n}<br>Δ{d:+.3f}",
                    showarrow=False, font_size=9,
                )
            st.plotly_chart(fig3, use_container_width=True)

            skipped = [ind for ind, v in ind_data.items() if v.get("skip")]
            if skipped:
                st.caption(f"データ不足でスキップ: {', '.join(skipped)}")
        st.caption("n=件数（参考値）。n<4の業種はスキップ。")

    st.info(
        "**解釈**: NGDはパラメータ空間の曲率（Fisher情報行列）を使って更新方向を補正するため、"
        "同じステップ数でより効率的に収束する。希少業種でΔAUCが正なら、そのセグメントへの"
        "重点的なデータ収集が投資対効果を高める。"
    )
