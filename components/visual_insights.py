"""
ビジュアルインサイト - 3つの分析チャート

① バブルチャート: スコア × スプレッド × 取得価格 × 成約/失注
② ヒートマップ:  業種 × スコア帯 → 成約率
③ サンキー図:   業種 → スコア帯 → 結果 のフロー
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_cases import load_past_cases


def _build_dataframe() -> pd.DataFrame:
    cases = load_past_cases()
    rows = []
    for c in cases:
        status = c.get("final_status", "未登録")
        if status not in ("成約", "失注"):
            continue
        score = float(c.get("score") or (c.get("result") or {}).get("score") or 0)
        if score <= 0:
            continue
        final_rate = float(c.get("final_rate") or 0)
        base_rate  = float(c.get("base_rate_at_time") or 0)
        spread = (final_rate - base_rate) if base_rate > 0 else float(c.get("winning_spread") or 0)
        inputs = c.get("inputs") or {}
        acq = float(inputs.get("acquisition_cost") or 0)
        rows.append({
            "score": score,
            "spread": spread,
            "acquisition_cost": acq,
            "status": status,
            "industry_major": c.get("industry_major") or "不明",
            "industry_sub": c.get("industry_sub") or "不明",
            "competitor": c.get("competitor_name") or "",
            "timestamp": (c.get("timestamp") or "")[:10],
        })
    return pd.DataFrame(rows)


# ── ① バブルチャート ─────────────────────────────────────────────────────────
def render_bubble_chart(current_case: dict | None = None):
    st.subheader("① バブルチャート — 案件マップ")
    st.caption("X=スコア　Y=スプレッド　バブルサイズ=取得価格　色=成約(青)/失注(赤)　★=現在の案件")

    df = _build_dataframe()
    if df.empty:
        st.info("成約・失注の登録済み案件がありません。")
        return

    fig = go.Figure()

    for status, color, symbol in [("成約", "#3b82f6", "circle"), ("失注", "#ef4444", "x")]:
        sub = df[df["status"] == status]
        if sub.empty:
            continue
        size = np.sqrt(sub["acquisition_cost"].clip(lower=100) / 100).clip(8, 40).values
        fig.add_trace(go.Scatter(
            x=sub["score"],
            y=sub["spread"],
            mode="markers",
            name=status,
            marker=dict(
                color=color,
                size=size,
                symbol=symbol,
                opacity=0.75,
                line=dict(width=1, color="#1e293b"),
            ),
            text=[
                f"{r['industry_sub']}<br>スコア:{r['score']:.0f}<br>"
                f"スプレッド:{r['spread']:.2f}%<br>取得価格:{r['acquisition_cost']:,.0f}千円"
                for _, r in sub.iterrows()
            ],
            hovertemplate="%{text}<extra></extra>",
        ))

    # 現在の案件
    if current_case:
        cur_score = float(current_case.get("score") or (current_case.get("result") or {}).get("score") or 0)
        if cur_score > 0:
            from base_rate_master import get_current_base_rate
            cur_base = get_current_base_rate(fallback=2.1)
            cur_comp = float(current_case.get("competitor_rate") or 0)
            cur_spread = (cur_comp - cur_base) if cur_comp > 0 else df[df["status"] == "成約"]["spread"].median() if not df.empty else 0.3
            fig.add_trace(go.Scatter(
                x=[cur_score], y=[cur_spread],
                mode="markers",
                name="現在の案件",
                marker=dict(color="#f59e0b", size=22, symbol="star", line=dict(width=2, color="#fff")),
                hovertemplate=f"現在の案件<br>スコア:{cur_score:.0f}<extra></extra>",
            ))

    # 承認ライン
    fig.add_vline(x=71, line_dash="dash", line_color="#64748b", line_width=1,
                  annotation_text="承認ライン 71", annotation_position="top right",
                  annotation_font_color="#94a3b8")

    fig.update_layout(
        xaxis=dict(title="審査スコア", range=[0, 105], gridcolor="#1e293b"),
        yaxis=dict(title="スプレッド (%)", gridcolor="#1e293b"),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
        margin=dict(l=50, r=30, t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, key="bubble_chart")


# ── ② ヒートマップ ────────────────────────────────────────────────────────────
def render_heatmap():
    st.subheader("② ヒートマップ — 業種 × スコア帯 成約率")
    st.caption("緑が濃いほど成約率が高い。赤は要注意。")

    df = _build_dataframe()
    if df.empty:
        st.info("成約・失注の登録済み案件がありません。")
        return

    bands = [(0, 50, "～50"), (50, 65, "50〜65"), (65, 80, "65〜80"), (80, 101, "80〜")]
    df["band"] = pd.cut(
        df["score"],
        bins=[b[0] for b in bands] + [101],
        labels=[b[2] for b in bands],
        right=False,
    )

    industries = df["industry_major"].value_counts().index.tolist()
    band_labels = [b[2] for b in bands]

    z, text, count_text = [], [], []
    for ind in industries:
        row_z, row_t, row_c = [], [], []
        for bl in band_labels:
            sub = df[(df["industry_major"] == ind) & (df["band"] == bl)]
            total = len(sub)
            won = (sub["status"] == "成約").sum()
            if total == 0:
                row_z.append(None)
                row_t.append("—")
                row_c.append("")
            else:
                rate = won / total
                row_z.append(rate)
                row_t.append(f"{rate*100:.0f}%")
                row_c.append(f"{won}/{total}件")
        z.append(row_z)
        row_hover = [f"{ind}<br>{bl}<br>成約率: {t}<br>{c}" for bl, t, c in zip(band_labels, row_t, row_c)]
        text.append(row_hover)
        count_text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=band_labels,
        y=industries,
        text=count_text,
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        colorscale=[
            [0.0, "#ef4444"],
            [0.5, "#f59e0b"],
            [1.0, "#22c55e"],
        ],
        zmin=0, zmax=1,
        hovertext=text,
        hovertemplate="%{hovertext}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title="成約率",
            tickformat=".0%",
            tickfont=dict(color="#e2e8f0"),
            titlefont=dict(color="#e2e8f0"),
        ),
    ))

    fig.update_layout(
        xaxis=dict(title="スコア帯", side="top", tickfont=dict(color="#e2e8f0")),
        yaxis=dict(tickfont=dict(color="#e2e8f0")),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        height=max(280, len(industries) * 52 + 80),
        margin=dict(l=120, r=60, t=60, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, key="heatmap_chart")


# ── ③ サンキー図 ──────────────────────────────────────────────────────────────
def render_sankey():
    st.subheader("③ サンキー図 — 案件フロー")
    st.caption("業種 → スコア帯 → 結果 の流れ。幅が太いほど件数が多い。")

    df = _build_dataframe()
    if len(df) < 3:
        st.info("3件以上の登録済み案件が必要です。")
        return

    bands = [(0, 50, "～50"), (50, 65, "50〜65"), (65, 80, "65〜80"), (80, 101, "80〜")]
    df["band"] = pd.cut(
        df["score"],
        bins=[b[0] for b in bands] + [101],
        labels=[b[2] for b in bands],
        right=False,
    )

    industries = df["industry_major"].unique().tolist()
    band_labels = [b[2] for b in bands]
    results = ["成約", "失注"]

    # ノードインデックス
    node_labels = industries + band_labels + results
    ind_idx  = {v: i for i, v in enumerate(industries)}
    band_idx = {v: len(industries) + i for i, v in enumerate(band_labels)}
    res_idx  = {v: len(industries) + len(band_labels) + i for i, v in enumerate(results)}

    node_colors = (
        ["#3b82f6"] * len(industries) +
        ["#8b5cf6"] * len(band_labels) +
        ["#22c55e", "#ef4444"]
    )

    sources, targets, values, link_colors = [], [], [], []

    # 業種 → スコア帯
    for ind in industries:
        for bl in band_labels:
            cnt = len(df[(df["industry_major"] == ind) & (df["band"] == bl)])
            if cnt > 0:
                sources.append(ind_idx[ind])
                targets.append(band_idx[bl])
                values.append(cnt)
                link_colors.append("rgba(139,92,246,0.3)")

    # スコア帯 → 結果
    for bl in band_labels:
        for r in results:
            cnt = len(df[(df["band"] == bl) & (df["status"] == r)])
            if cnt > 0:
                sources.append(band_idx[bl])
                targets.append(res_idx[r])
                values.append(cnt)
                link_colors.append("rgba(34,197,94,0.35)" if r == "成約" else "rgba(239,68,68,0.35)")

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="#1e293b", width=0.5),
            label=node_labels,
            color=node_colors,
            hovertemplate="%{label}<br>件数: %{value}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    ))

    fig.update_layout(
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0", size=12),
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key="sankey_chart")


# ── メインレンダラ ─────────────────────────────────────────────────────────────
def render_visual_insights():
    st.title("📊 ビジュアルインサイト")
    st.caption("過去の審査案件データを多角的に可視化します。")

    # 現在の案件をセッションから取得
    current_case = None
    last_res = st.session_state.get("last_result")
    last_inputs = st.session_state.get("last_submitted_inputs")
    if last_res and last_inputs:
        current_case = {**last_res, **last_inputs}

    tab1, tab2, tab3 = st.tabs(["🫧 バブルチャート", "🌡️ ヒートマップ", "🌊 サンキー図"])

    with tab1:
        render_bubble_chart(current_case)

    with tab2:
        render_heatmap()

    with tab3:
        render_sankey()
