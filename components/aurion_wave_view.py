"""
AURION CORE — Streamlit 埋め込み波形ビュー (AV.7)
Plotly で静的波形スナップショット + 干渉スコアを表示する。
"""
from __future__ import annotations

import math
from typing import Any

import streamlit as st


def render_aurion_wave_view(
    quantum_inputs: dict[str, Any] | None = None,
    quantum_risk: float | None = None,
    pair_anomalies: dict[str, float] | None = None,
    industry_code: str = "",
) -> None:
    """
    現在案件の量子波形を描画する。

    Parameters
    ----------
    quantum_inputs :
        score_calculation.py が session_state に保存した `quantum_inputs` dict。
        None の場合は session_state["last_result"] から自動取得。
    quantum_risk :
        量子リスクスコア。None の場合は session_state から自動取得。
    pair_anomalies :
        ペア別異常度 dict。
    industry_code :
        業種大分類コード（空の場合は quantum_inputs から推定）。
    """
    # ── 依存 import ────────────────────────────────────────────────────────
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.warning("plotly が未インストールです。`pip install plotly` を実行してください。")
        return

    try:
        from aurion_wave_engine import compute_wave
        from aurion_phase_mapper import PhaseMapper
    except ImportError as e:
        st.error(f"AURION モジュール読み込みエラー: {e}")
        return

    # ── データ取得 ─────────────────────────────────────────────────────────
    if quantum_inputs is None:
        res = st.session_state.get("last_result", {})
        quantum_inputs = res.get("quantum_inputs") or {}
        quantum_risk = res.get("quantum_risk") if quantum_risk is None else quantum_risk
        pair_anomalies = pair_anomalies or res.get("quantum_anomalies") or {}

    if not quantum_inputs:
        st.info("審査を実行すると量子波形が表示されます。")
        return

    # ── 業種コード確定 ─────────────────────────────────────────────────────
    if not industry_code:
        raw = str(quantum_inputs.get("industry_major") or "")
        industry_code = raw.split(" ")[0].strip()
        if not industry_code:
            try:
                from quantum_analysis_module import _infer_major_code
                industry_code = _infer_major_code(
                    str(quantum_inputs.get("industry_sub") or "")
                )
            except Exception:
                pass

    var_a, var_b = PhaseMapper.get_pair(industry_code)
    val_a = float(quantum_inputs.get(var_a, 0.0) or 0.0)
    val_b = float(quantum_inputs.get(var_b, 0.0) or 0.0)
    v1, v2 = PhaseMapper.normalize_pair(var_a, var_b, val_a, val_b)
    result = compute_wave(v1, v2, n_points=256)

    decoherent = result.phase_diff > math.pi * 0.7
    score_color = "#dc2626" if decoherent else "#16a34a"

    # ── ヘッダー ────────────────────────────────────────────────────────────
    st.markdown("#### ⚛️ 量子波形干渉ビジュアル")
    col_s, col_p, col_w = st.columns([1, 1, 2])
    with col_s:
        st.metric("振幅スコア", f"{result.amplitude_score:.3f}")
    with col_p:
        st.metric("位相差", f"{result.phase_diff / math.pi:.2f} π")
    with col_w:
        if decoherent:
            st.error("⚠ データの歪みを検出")
        elif quantum_risk is not None and quantum_risk >= 20:
            st.warning(f"Q_risk: {quantum_risk:.1f}")
        else:
            st.success("共鳴 — 整合的な財務構造")

    # ── Plotly 波形グラフ ─────────────────────────────────────────────────
    import numpy as np
    t = np.linspace(0, 2, 256)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        subplot_titles=["波形干渉", "合成波"],
        vertical_spacing=0.12,
    )

    # 波 A
    fig.add_trace(go.Scatter(
        x=t, y=result.wave_a,
        mode="lines",
        name=f"波A ({var_a})",
        line=dict(color="#06b6d4", width=2),
    ), row=1, col=1)

    # 波 B
    fig.add_trace(go.Scatter(
        x=t, y=result.wave_b,
        mode="lines",
        name=f"波B ({var_b})",
        line=dict(color="#f59e0b", width=2),
    ), row=1, col=1)

    # 合成波
    composite_color = "#dc2626" if decoherent else "#22c55e"
    fig.add_trace(go.Scatter(
        x=t, y=result.composite,
        mode="lines",
        name="合成波",
        line=dict(color=composite_color, width=3),
        fill="tozeroy",
        fillcolor=f"rgba({'220,38,38' if decoherent else '34,197,94'},0.08)",
    ), row=2, col=1)

    # 中心線
    for row in (1, 2):
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dash"), row=row, col=1)

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=40, b=20),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        showlegend=True,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#64748b")
    fig.update_yaxes(showgrid=False, zeroline=False, range=[-1.2, 1.2], color="#64748b")

    st.plotly_chart(fig, use_container_width=True)

    # ── ペア別異常度 ────────────────────────────────────────────────────────
    if pair_anomalies:
        active = {k: v for k, v in pair_anomalies.items() if v > 0.02}
        if active:
            st.markdown("**ペア別干渉度（乖離度）**")
            sorted_pairs = sorted(active.items(), key=lambda x: x[1], reverse=True)
            labels = [k.replace("_x_", " × ") for k, _ in sorted_pairs]
            values = [v for _, v in sorted_pairs]
            colors = [
                "#dc2626" if v >= 0.5 else "#f97316" if v >= 0.25 else "#3b82f6"
                for v in values
            ]
            fig2 = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker_color=colors,
                text=[f"{v:.2f}" for v in values],
                textposition="outside",
            ))
            fig2.update_layout(
                height=max(120, 35 + 32 * len(sorted_pairs)),
                margin=dict(l=10, r=50, t=10, b=10),
                plot_bgcolor="#0f172a",
                paper_bgcolor="#1e293b",
                font=dict(color="#e2e8f0", size=10),
                xaxis=dict(range=[0, 1.1], showgrid=False),
                yaxis=dict(autorange="reversed", showgrid=False),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── 変数値サマリ ────────────────────────────────────────────────────────
    with st.expander("🔍 使用変数の詳細", expanded=False):
        st.caption(
            f"**{var_a}** = {val_a/1000:,.1f} 百万円 → v1 = {v1:.3f}  |  "
            f"**{var_b}** = {val_b/1000:,.1f} 百万円 → v2 = {v2:.3f}"
        )
        st.caption(
            f"業種コード: **{industry_code or '不明'}**  |  "
            f"デコヒーレンス閾値: π × 0.7 = {math.pi * 0.7:.3f} rad"
        )
