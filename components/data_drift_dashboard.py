"""
データドリフト監視ダッシュボード

スコア分布・業種比率・成約率の時系列推移を監視し、
モデルのコンセプトドリフトや入力分布の変化を検出する。
"""

from __future__ import annotations

import os
import sys
import json
import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_cases import load_all_cases


# ── 定数 ──────────────────────────────────────────────────────────────────────

_WIN_STATUSES = {"成約", "検収", "検収完了"}
_DRIFT_THRESHOLD = 3.0   # スコア平均差がこの点数以上でドリフト警告
_RECENT_DAYS_DEFAULT = 30
_MIN_CASES_FOR_ANALYSIS = 5


# ── データ整形 ─────────────────────────────────────────────────────────────────

def _build_df(cases: list[dict]) -> pd.DataFrame:
    """案件リストを分析用 DataFrame に変換する。"""
    rows = []
    for c in cases:
        ts_raw = (c.get("timestamp") or "").strip()
        result = c.get("result") or {}
        score = result.get("score")
        quantum_risk = result.get("quantum_risk")
        final_status = (c.get("final_status") or "").strip()
        industry_major = (c.get("industry_major") or "").strip()
        industry_sub = (c.get("industry_sub") or "").strip()

        try:
            score_f = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_f = None

        try:
            qr_f = float(quantum_risk) if quantum_risk is not None else None
        except (TypeError, ValueError):
            qr_f = None

        try:
            ts = pd.to_datetime(ts_raw, errors="coerce")
        except Exception:
            ts = pd.NaT

        rows.append(
            {
                "timestamp": ts,
                "score": score_f,
                "quantum_risk": qr_f,
                "final_status": final_status,
                "is_win": final_status in _WIN_STATUSES,
                "industry_major": industry_major,
                "industry_sub": industry_sub,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["week"] = df["timestamp"].dt.to_period("W").astype(str)
    return df


def _split_recent(df: pd.DataFrame, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """直近 N 日と過去全体に分割する。"""
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    recent = df[df["timestamp"] >= cutoff]
    past = df[df["timestamp"] < cutoff]
    return recent, past


# ── ドリフト指標計算 ───────────────────────────────────────────────────────────

def _compute_drift_metrics(df: pd.DataFrame, days: int) -> dict[str, Any]:
    recent, past = _split_recent(df, days)

    result: dict[str, Any] = {
        "n_total": len(df),
        "n_recent": len(recent),
        "n_past": len(past),
        "days": days,
    }

    scored_recent = recent.dropna(subset=["score"])
    scored_past = past.dropna(subset=["score"])

    if not scored_recent.empty and not scored_past.empty:
        result["score_recent_mean"] = scored_recent["score"].mean()
        result["score_past_mean"] = scored_past["score"].mean()
        result["score_drift"] = result["score_recent_mean"] - result["score_past_mean"]
        result["score_recent_std"] = scored_recent["score"].std()
        result["score_past_std"] = scored_past["score"].std()
        result["score_drift_detected"] = abs(result["score_drift"]) >= _DRIFT_THRESHOLD
    else:
        result["score_drift_detected"] = False
        result["score_drift"] = None

    # 成約率ドリフト
    if len(recent) >= _MIN_CASES_FOR_ANALYSIS and len(past) >= _MIN_CASES_FOR_ANALYSIS:
        result["winrate_recent"] = recent["is_win"].mean() * 100
        result["winrate_past"] = past["is_win"].mean() * 100
        result["winrate_drift"] = result["winrate_recent"] - result["winrate_past"]
    else:
        result["winrate_recent"] = None
        result["winrate_past"] = None
        result["winrate_drift"] = None

    # 業種分布ドリフト（KL近似）
    if len(recent) >= _MIN_CASES_FOR_ANALYSIS and len(past) >= _MIN_CASES_FOR_ANALYSIS:
        cats = sorted(set(df["industry_major"].unique()) - {""})
        if cats:
            eps = 1e-9
            past_dist = past["industry_major"].value_counts(normalize=True).reindex(cats, fill_value=0) + eps
            recent_dist = recent["industry_major"].value_counts(normalize=True).reindex(cats, fill_value=0) + eps
            kl = float(np.sum(recent_dist * np.log(recent_dist / past_dist)))
            result["industry_kl"] = round(kl, 4)
            result["industry_drift_detected"] = kl > 0.1
        else:
            result["industry_kl"] = None
            result["industry_drift_detected"] = False
    else:
        result["industry_kl"] = None
        result["industry_drift_detected"] = False

    return result


# ── チャート描画 ───────────────────────────────────────────────────────────────

def _chart_score_trend(df: pd.DataFrame) -> go.Figure:
    monthly = (
        df.dropna(subset=["score"])
        .groupby("month")["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    if monthly.empty:
        fig = go.Figure()
        fig.add_annotation(text="データ不足", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    fig = go.Figure()
    # 信頼帯
    std_upper = monthly["mean"] + monthly["std"].fillna(0)
    std_lower = monthly["mean"] - monthly["std"].fillna(0)
    fig.add_trace(
        go.Scatter(
            x=list(monthly["month"]) + list(monthly["month"][::-1]),
            y=list(std_upper) + list(std_lower[::-1]),
            fill="toself",
            fillcolor="rgba(37,99,235,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±1σ",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    # 平均スコア線
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["mean"],
            mode="lines+markers",
            name="月次平均スコア",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=6),
            customdata=monthly[["count", "std"]].values,
            hovertemplate="%{x}<br>平均: %{y:.1f}点<br>件数: %{customdata[0]}<extra></extra>",
        )
    )
    # 全期間平均ライン
    overall_mean = df["score"].dropna().mean()
    fig.add_hline(
        y=overall_mean,
        line_dash="dash",
        line_color="#9ca3af",
        annotation_text=f"全期間平均 {overall_mean:.1f}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="スコア月次推移",
        xaxis_title="月",
        yaxis_title="審査スコア",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _chart_winrate_trend(df: pd.DataFrame) -> go.Figure:
    scored = df[df["final_status"].isin(_WIN_STATUSES | {"失注"})]
    if scored.empty:
        fig = go.Figure()
        fig.add_annotation(text="成約/失注データなし", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    monthly = (
        scored.groupby("month")
        .apply(lambda g: pd.Series({"win_rate": g["is_win"].mean() * 100, "count": len(g)}))
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly["month"],
            y=monthly["count"],
            name="件数",
            marker_color="#93c5fd",
            yaxis="y2",
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["win_rate"],
            mode="lines+markers",
            name="成約率(%)",
            line=dict(color="#16a34a", width=2),
            marker=dict(size=6),
            customdata=monthly[["count"]].values,
            hovertemplate="%{x}<br>成約率: %{y:.1f}%<br>件数: %{customdata[0]}<extra></extra>",
        )
    )
    overall_wr = scored["is_win"].mean() * 100
    fig.add_hline(
        y=overall_wr,
        line_dash="dash",
        line_color="#9ca3af",
        annotation_text=f"全期間成約率 {overall_wr:.1f}%",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="成約率月次推移",
        xaxis_title="月",
        yaxis_title="成約率 (%)",
        yaxis2=dict(title="件数", overlaying="y", side="right", showgrid=False),
        height=320,
        margin=dict(l=40, r=60, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _chart_industry_dist(recent: pd.DataFrame, past: pd.DataFrame) -> go.Figure:
    cats = sorted(set(past["industry_major"].unique()) | set(recent["industry_major"].unique()) - {""})
    if not cats:
        fig = go.Figure()
        fig.add_annotation(text="業種データなし", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    past_dist = past["industry_major"].value_counts(normalize=True).reindex(cats, fill_value=0) * 100
    recent_dist = recent["industry_major"].value_counts(normalize=True).reindex(cats, fill_value=0) * 100

    fig = go.Figure(
        data=[
            go.Bar(name="過去全体", x=cats, y=past_dist.values, marker_color="#93c5fd"),
            go.Bar(name=f"直近{len(recent)}件", x=cats, y=recent_dist.values, marker_color="#2563eb"),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="業種構成比の変化 (%)",
        xaxis_title="業種",
        yaxis_title="構成比 (%)",
        height=320,
        margin=dict(l=40, r=20, t=40, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickangle=-30),
    )
    return fig


def _chart_score_hist(recent: pd.DataFrame, past: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not past.dropna(subset=["score"]).empty:
        fig.add_trace(
            go.Histogram(
                x=past.dropna(subset=["score"])["score"],
                name="過去",
                opacity=0.6,
                nbinsx=20,
                marker_color="#93c5fd",
                histnorm="probability density",
            )
        )
    if not recent.dropna(subset=["score"]).empty:
        fig.add_trace(
            go.Histogram(
                x=recent.dropna(subset=["score"])["score"],
                name="直近",
                opacity=0.75,
                nbinsx=20,
                marker_color="#2563eb",
                histnorm="probability density",
            )
        )
    fig.update_layout(
        barmode="overlay",
        title="スコア分布の比較",
        xaxis_title="審査スコア",
        yaxis_title="密度",
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _chart_quantum_risk_trend(df: pd.DataFrame) -> go.Figure:
    monthly = (
        df.dropna(subset=["quantum_risk"])
        .groupby("month")["quantum_risk"]
        .mean()
        .reset_index()
    )
    if monthly.empty:
        fig = go.Figure()
        fig.add_annotation(text="Q_riskデータなし", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    fig = go.Figure(
        go.Scatter(
            x=monthly["month"],
            y=monthly["quantum_risk"],
            mode="lines+markers",
            name="月次平均 Q_risk",
            line=dict(color="#dc2626", width=2),
            marker=dict(size=6),
            hovertemplate="%{x}<br>Q_risk平均: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_hline(y=35, line_dash="dash", line_color="#fbbf24", annotation_text="要注意閾値 35", annotation_position="top right")
    fig.update_layout(
        title="量子リスク(Q_risk)月次推移",
        xaxis_title="月",
        yaxis_title="Q_risk",
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ── メイン描画関数 ─────────────────────────────────────────────────────────────

def render_data_drift_dashboard() -> None:
    st.header("📡 データドリフト監視ダッシュボード")
    st.caption("審査スコア・成約率・業種分布の時系列変化を監視し、モデルのコンセプトドリフトや入力分布の変化を検出します。")

    # ── データ読み込み ─────────────────────────────────────────────────────────
    with st.spinner("案件データを読み込み中…"):
        cases = load_all_cases()

    if not cases:
        st.info("過去案件データがありません。審査を実施してから再度ご確認ください。")
        return

    df = _build_df(cases)

    if df.empty or len(df) < _MIN_CASES_FOR_ANALYSIS:
        st.info(f"分析に必要なデータ({_MIN_CASES_FOR_ANALYSIS}件以上)が不足しています。現在 {len(df)} 件。")
        return

    # ── コントロール ───────────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        recent_days = st.slider(
            "直近期間（日数）",
            min_value=7,
            max_value=180,
            value=_RECENT_DAYS_DEFAULT,
            step=7,
            help="「直近」と「過去全体」を比較する際の境界日数",
        )
    with col_ctrl2:
        st.metric("総案件数", f"{len(df)} 件", help="DBに登録されている全案件数")

    recent_df, past_df = _split_recent(df, recent_days)
    metrics = _compute_drift_metrics(df, recent_days)

    # ── アラートバナー ─────────────────────────────────────────────────────────
    alerts = []
    if metrics.get("score_drift_detected") and metrics.get("score_drift") is not None:
        direction = "低下" if metrics["score_drift"] < 0 else "上昇"
        alerts.append(
            f"**スコアドリフト検出**: 直近{recent_days}日の平均スコアが過去比 "
            f"{metrics['score_drift']:+.1f}点（{direction}）しています。"
        )
    if metrics.get("industry_drift_detected") and metrics.get("industry_kl") is not None:
        alerts.append(
            f"**業種分布ドリフト検出**: KLダイバージェンス = {metrics['industry_kl']:.3f}（閾値: 0.1）。"
            "直近の業種構成が過去と大きく異なります。"
        )

    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.success("ドリフト異常は検出されていません。")

    # ── KPI行 ──────────────────────────────────────────────────────────────────
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        if metrics.get("score_recent_mean") is not None:
            delta = f"{metrics['score_drift']:+.1f}点" if metrics.get("score_drift") is not None else None
            st.metric(
                f"直近{recent_days}日 平均スコア",
                f"{metrics['score_recent_mean']:.1f}",
                delta=delta,
                delta_color="inverse" if (metrics.get("score_drift") or 0) < 0 else "normal",
                help="直近期間の平均審査スコア。カッコ内は過去比の差分。",
            )
        else:
            st.metric(f"直近{recent_days}日 平均スコア", "N/A")

    with kpi_cols[1]:
        if metrics.get("winrate_recent") is not None:
            wr_delta = f"{metrics['winrate_drift']:+.1f}%" if metrics.get("winrate_drift") is not None else None
            st.metric(
                f"直近{recent_days}日 成約率",
                f"{metrics['winrate_recent']:.1f}%",
                delta=wr_delta,
                help="成約・検収を勝ちとして集計した成約率。",
            )
        else:
            st.metric(f"直近{recent_days}日 成約率", "N/A")

    with kpi_cols[2]:
        if metrics.get("industry_kl") is not None:
            kl_status = "⚠️ 高" if metrics["industry_drift_detected"] else "✅ 正常"
            st.metric(
                "業種KLダイバージェンス",
                f"{metrics['industry_kl']:.3f}",
                delta=kl_status,
                delta_color="off",
                help="業種分布のずれを表す指標。0.1以上でドリフト警告。",
            )
        else:
            st.metric("業種KLダイバージェンス", "N/A")

    with kpi_cols[3]:
        st.metric(
            f"直近{recent_days}日の案件数",
            f"{metrics['n_recent']} 件",
            help=f"全{metrics['n_total']}件中の直近{recent_days}日分",
        )

    st.divider()

    # ── タブ ──────────────────────────────────────────────────────────────────
    tab_score, tab_win, tab_industry, tab_qrisk, tab_raw = st.tabs(
        ["📈 スコア推移", "🏆 成約率推移", "🏭 業種分布", "⚛️ Q_risk推移", "🗂️ 生データ確認"]
    )

    with tab_score:
        st.plotly_chart(_chart_score_trend(df), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**直近{recent_days}日** (n={len(recent_df.dropna(subset=['score']))})")
            st.plotly_chart(_chart_score_hist(recent_df, past_df), use_container_width=True)
        with col_b:
            scored_all = df.dropna(subset=["score"])
            if not scored_all.empty:
                col_stats1, col_stats2 = st.columns(2)
                with col_stats1:
                    st.metric("全期間 平均", f"{scored_all['score'].mean():.1f}")
                    st.metric("全期間 中央値", f"{scored_all['score'].median():.1f}")
                with col_stats2:
                    st.metric("全期間 最高", f"{scored_all['score'].max():.1f}")
                    st.metric("全期間 最低", f"{scored_all['score'].min():.1f}")

    with tab_win:
        st.plotly_chart(_chart_winrate_trend(df), use_container_width=True)
        if metrics.get("winrate_past") is not None and metrics.get("winrate_recent") is not None:
            col_wr1, col_wr2 = st.columns(2)
            with col_wr1:
                st.metric("過去全体 成約率", f"{metrics['winrate_past']:.1f}%")
            with col_wr2:
                st.metric(f"直近{recent_days}日 成約率", f"{metrics['winrate_recent']:.1f}%")

    with tab_industry:
        if len(recent_df) >= _MIN_CASES_FOR_ANALYSIS:
            st.plotly_chart(_chart_industry_dist(recent_df, past_df), use_container_width=True)
            if metrics.get("industry_kl") is not None:
                if metrics["industry_drift_detected"]:
                    st.warning(
                        f"業種分布ドリフト: KL = {metrics['industry_kl']:.3f}。"
                        "直近と過去の業種構成が大きく異なります。係数の再学習を検討してください。"
                    )
                else:
                    st.success(f"業種分布は安定しています (KL = {metrics['industry_kl']:.3f})。")
        else:
            st.info(f"直近{recent_days}日のデータが{_MIN_CASES_FOR_ANALYSIS}件未満のため業種比較ができません。")

    with tab_qrisk:
        st.plotly_chart(_chart_quantum_risk_trend(df), use_container_width=True)
        qr_scored = df.dropna(subset=["quantum_risk"])
        if not qr_scored.empty:
            high_risk = qr_scored[qr_scored["quantum_risk"] >= 35]
            st.metric(
                "Q_risk≥35 の割合",
                f"{len(high_risk) / len(qr_scored) * 100:.1f}%",
                help="全審査案件のうち量子リスク要注意案件の割合",
            )

    with tab_raw:
        st.caption("直近期間の生データ（スコア降順）")
        view_df = recent_df[["timestamp", "industry_major", "industry_sub", "score", "quantum_risk", "final_status"]].copy()
        view_df["timestamp"] = view_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        view_df = view_df.sort_values("score", ascending=False, na_position="last")
        st.dataframe(view_df, use_container_width=True, hide_index=True)

    # ── マクロドリフト監視との連携 ────────────────────────────────────────────
    st.divider()
    st.subheader("🌐 コンセプトドリフト診断")
    st.caption("macro_drift_monitor.py による詳細診断結果")
    try:
        from macro_drift_monitor import check_concept_drift

        drift_result = check_concept_drift(
            recent_days=recent_days,
            threshold_points=_DRIFT_THRESHOLD,
            min_recent_cases=_MIN_CASES_FOR_ANALYSIS,
        )
        if drift_result.get("is_drift"):
            st.error(f"⚠️ {drift_result.get('message', '')}")
        else:
            st.info(drift_result.get("message", "診断結果なし"))
    except Exception as e:
        st.caption(f"コンセプトドリフト診断スキップ: {e}")
