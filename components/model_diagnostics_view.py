"""
Phase 0: モデル診断 — 棚卸し・可視化

- データ分布タブ: 業種別件数・成約率・主要変数分布
- 予測誤差タブ: 業種/規模/月別の誤差偏り
- 損失地形タブ: 2係数を変動させたAUCコンター
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_REPO_ROOT, "data", "lease_data.db")
_WIN_STATUSES = {"成約", "検収完了", "検収"}
_VALID_STATUSES = _WIN_STATUSES | {"失注"}

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ── データ読み込み ─────────────────────────────────────────────────────────────

def _acost_tier(v: float) -> str:
    if v < 1_000:
        return "~100万"
    if v < 5_000:
        return "100~500万"
    if v < 10_000:
        return "500~1000万"
    if v < 30_000:
        return "1000~3000万"
    return "3000万~"


def _load_df() -> pd.DataFrame:
    if not os.path.exists(_DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(_DB_PATH)
    rows = conn.execute(
        "SELECT id, timestamp, final_status, score, data FROM past_cases"
    ).fetchall()
    conn.close()

    records = []
    for case_id, ts, status, score, data_json in rows:
        d = json.loads(data_json or "{}")
        inp = d.get("inputs", {})
        res = d.get("result", {})
        try:
            month = int(str(ts or "")[:7].split("-")[1])
        except (IndexError, ValueError):
            month = 0
        acost = float(inp.get("acquisition_cost") or 0)
        records.append({
            "id": case_id,
            "timestamp": ts,
            "month": month,
            "final_status": status,
            "score": float(score or 0),
            "industry": d.get("industry_major", "不明"),
            "nenshu": float(inp.get("nenshu") or 0),
            "acquisition_cost": acost,
            "acost_tier": _acost_tier(acost),
            "op_profit": float(inp.get("op_profit") or 0),
            "bank_credit": float(inp.get("bank_credit") or 0),
            "lease_credit": float(inp.get("lease_credit") or 0),
            "score_borrower": float(res.get("score_borrower") or 0),
            "asset_score": float(res.get("asset_score") or 50),
            "label": 1 if status in _WIN_STATUSES else 0,
        })
    df = pd.DataFrame(records)
    df["error"] = df["score"] - df["label"] * 100
    return df


# ── Tab 1: データ分布 ─────────────────────────────────────────────────────────

def _render_distribution(df: pd.DataFrame) -> None:
    st.subheader("業種別 件数・成約率")
    grp = (
        df.groupby("industry")
        .agg(件数=("id", "count"), 成約数=("label", "sum"))
        .reset_index()
    )
    grp["成約率(%)"] = (grp["成約数"] / grp["件数"] * 100).round(1)
    grp = grp.sort_values("件数", ascending=False)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=grp["industry"], y=grp["件数"], name="件数", secondary_y=False)
    fig.add_scatter(
        x=grp["industry"], y=grp["成約率(%)"],
        mode="lines+markers", name="成約率(%)", secondary_y=True,
        marker_color="crimson",
    )
    fig.update_layout(height=380, margin=dict(t=20, b=80), xaxis_tickangle=-30)
    fig.update_yaxes(title_text="件数", secondary_y=False)
    fig.update_yaxes(title_text="成約率(%)", secondary_y=True, range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"n = {len(df)} 件（参考値）")

    st.subheader("主要変数の業種別分布")
    feat = st.selectbox(
        "変数", ["score", "nenshu", "op_profit", "bank_credit", "lease_credit", "asset_score"],
        key="diag_feat",
    )
    fig2 = px.box(
        df, x="industry", y=feat, color="final_status",
        points="all", height=400,
        category_orders={"final_status": sorted(df["final_status"].unique())},
    )
    fig2.update_layout(margin=dict(t=20, b=80), xaxis_tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)


# ── Tab 2: 予測誤差の偏り ─────────────────────────────────────────────────────

def _render_bias(df: pd.DataFrame) -> None:
    valid = df[df["final_status"].isin(_VALID_STATUSES)].copy()
    if valid.empty:
        st.warning("成約/失注データがありません。")
        return

    st.subheader("業種別スコア誤差（予測 − 実績×100）")
    fig = px.box(
        valid, x="industry", y="error", color="final_status",
        points="all", height=380,
        labels={"error": "誤差"},
        category_orders={"final_status": sorted(valid["final_status"].unique())},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(margin=dict(t=20, b=80), xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("誤差 > 0: 過大評価（失注リスク）、誤差 < 0: 過小評価（成約見逃し）")

    st.subheader("取得価格規模別 誤差分布")
    tier_order = ["~100万", "100~500万", "500~1000万", "1000~3000万", "3000万~"]
    fig2 = px.violin(
        valid, x="acost_tier", y="error", color="final_status",
        box=True, points="all", height=360,
        category_orders={"acost_tier": tier_order},
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("月別 平均誤差（季節偏り）")
    monthly = valid.groupby("month")["error"].agg(["mean", "std", "count"]).reset_index()
    monthly.columns = ["month", "mean_err", "std_err", "n"]
    monthly = monthly[monthly["n"] >= 1]
    if len(monthly) >= 2:
        fig3 = go.Figure()
        fig3.add_scatter(
            x=monthly["month"], y=monthly["mean_err"],
            mode="lines+markers", name="平均誤差",
            error_y=dict(type="data", array=monthly["std_err"].tolist(), visible=True),
        )
        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
        fig3.update_layout(
            height=320, xaxis_title="月", yaxis_title="平均誤差",
            xaxis=dict(tickmode="linear", dtick=1), margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(f"各月のエラーバー = 1σ (n={len(valid)})")
    else:
        st.info("月別に分けるにはデータが不足しています。")


# ── Tab 3: 損失地形 ──────────────────────────────────────────────────────────

_CANDIDATE_COEFFS = [
    "intercept", "sales_log", "bank_credit_log", "lease_credit_log",
    "op_profit", "ord_profit", "net_income", "machines", "depreciation",
    "contracts", "ind_medical", "ind_transport", "ind_manufacturing",
    "ind_construction", "grade_4_6", "grade_watch",
]


@st.cache_data(show_spinner=False, ttl=300)
def _compute_landscape(coeff_key: str, feat_x: str, feat_y: str, n: int = 22):
    from data_cases import load_all_cases, get_effective_coeffs, get_score_weights
    from scoring_core import _calculate_z, _safe_sigmoid
    from sklearn.metrics import roc_auc_score

    coeffs = get_effective_coeffs(coeff_key)
    w_b, w_a, _, _ = get_score_weights()
    cx = coeffs.get(feat_x, 0.0)
    cy = coeffs.get(feat_y, 0.0)

    cases = [c for c in load_all_cases() if c.get("final_status") in _VALID_STATUSES]
    if len(cases) < 5:
        return None, cx, cy

    y_true = np.array([1 if c["final_status"] in _WIN_STATUSES else 0 for c in cases])
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None, cx, cy

    def _build_data(c):
        inp = c.get("inputs", {})
        return {
            "nenshu": float(inp.get("nenshu") or 0),
            "bank_credit": float(inp.get("bank_credit") or 0),
            "lease_credit": float(inp.get("lease_credit") or 0),
            "op_profit": float(inp.get("op_profit") or 0) / 1000,
            "ord_profit": float(inp.get("ord_profit") or 0) / 1000,
            "net_income": float(inp.get("net_income") or 0) / 1000,
            "machines": float(inp.get("machines") or 0) / 1000,
            "other_assets": float(inp.get("other_assets") or 0) / 1000,
            "rent": float(inp.get("rent") or 0) / 1000,
            "gross_profit": float(inp.get("gross_profit") or 0) / 1000,
            "depreciation": float(inp.get("depreciation") or 0) / 1000,
            "dep_expense": float(inp.get("dep_expense") or 0) / 1000,
            "rent_expense": float(inp.get("rent_expense") or 0) / 1000,
            "contracts": int(inp.get("contracts") or 0),
            "grade": inp.get("grade") or "1-3",
            "industry_major": c.get("industry_major", ""),
            "sales_dept": c.get("sales_dept", "未設定"),
        }

    data_list = [_build_data(c) for c in cases]
    asset_scores = np.array([float(c.get("result", {}).get("asset_score") or 50) for c in cases])

    span = max(abs(cx) * 0.8, 0.5)
    xs = np.linspace(cx - span, cx + span, n)
    span_y = max(abs(cy) * 0.8, 0.5)
    ys = np.linspace(cy - span_y, cy + span_y, n)

    Z = np.zeros((n, n))
    for i, vy in enumerate(ys):
        for j, vx in enumerate(xs):
            tmp = dict(coeffs)
            tmp[feat_x] = vx
            tmp[feat_y] = vy
            scores = np.array([
                w_b * _safe_sigmoid(_calculate_z(d, tmp)) * 100 + w_a * a
                for d, a in zip(data_list, asset_scores)
            ])
            try:
                Z[i, j] = roc_auc_score(y_true, scores)
            except Exception:
                Z[i, j] = 0.5

    return Z, xs, ys, cx, cy


def _render_landscape() -> None:
    from data_cases import get_effective_coeffs

    coeff_key = st.selectbox("係数セット", ["全体_既存先", "全体_新規先"], key="diag_coeff_key")
    coeffs = get_effective_coeffs(coeff_key)
    available = [k for k in _CANDIDATE_COEFFS if k in coeffs]

    col1, col2 = st.columns(2)
    with col1:
        feat_x = st.selectbox("X軸係数", available, index=available.index("sales_log") if "sales_log" in available else 0, key="diag_fx")
    with col2:
        feat_y = st.selectbox("Y軸係数", available, index=available.index("bank_credit_log") if "bank_credit_log" in available else 1, key="diag_fy")

    if feat_x == feat_y:
        st.warning("X軸とY軸に異なる係数を選択してください。")
        return

    with st.spinner("損失地形を計算中…"):
        result = _compute_landscape(coeff_key, feat_x, feat_y)

    if result[0] is None:
        st.warning("有効データが不足しています（成約/失注が5件以上必要）。")
        return

    Z, xs, ys, cx, cy = result

    fig = go.Figure()
    fig.add_contour(
        x=xs, y=ys, z=Z,
        colorscale="Viridis",
        contours=dict(showlabels=True, labelfont=dict(size=10)),
        colorbar=dict(title="AUC"),
    )
    fig.add_scatter(
        x=[cx], y=[cy],
        mode="markers",
        marker=dict(symbol="star", size=14, color="red"),
        name="現在の係数",
    )
    fig.update_layout(
        height=450,
        xaxis_title=feat_x,
        yaxis_title=feat_y,
        title=f"AUC 損失地形 — {feat_x} × {feat_y}（赤★ = 現在値）",
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"現在値: {feat_x}={cx:.4f}, {feat_y}={cy:.4f}  |  "
        f"グリッド上最大AUC={Z.max():.3f}（現在={Z[len(ys)//2, len(xs)//2]:.3f}）  |  "
        f"n={sum(1 for _ in range(1))}件（参考値）"
    )
    st.info(
        "等高線が密な方向 = 係数変化に対してAUCが急峻に変化する方向。"
        "★が等高線の中心（山頂）付近にあれば最適化済み。外れている場合は改善余地あり。"
    )


# ── メインエントリ ────────────────────────────────────────────────────────────

def render_model_diagnostics() -> None:
    st.title("🔭 モデル診断 — Phase 0 棚卸し・可視化")
    st.caption("学習データの分布・予測誤差の偏り・損失地形を俯瞰し、改善優先箇所を特定する。")

    df = _load_df()
    if df.empty:
        st.error("DBが見つかりません。")
        return

    tab1, tab2, tab3 = st.tabs(["📊 データ分布", "⚖️ 予測誤差の偏り", "🗺️ 損失地形"])

    with tab1:
        _render_distribution(df)
    with tab2:
        _render_bias(df)
    with tab3:
        _render_landscape()
