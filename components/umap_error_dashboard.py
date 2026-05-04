from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_cases import load_all_cases_cached

try:
    import umap
except Exception:
    umap = None


@st.cache_data(show_spinner=False)
def _load_cases_df() -> pd.DataFrame:
    records = load_all_cases_cached() or []
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records, sep=".")


def _extract_model_prob(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "result.predict_proba",
        "result.proba",
        "predict_proba",
        "prediction_probability",
        "probability",
    ]
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


@st.cache_data(show_spinner=True)
def compute_umap_projection(df: pd.DataFrame, features: tuple[str, ...], fill_strategy: str = "median") -> pd.DataFrame:
    if umap is None:
        raise RuntimeError("umap-learn が未インストールです。requirements.txt に追加後、再起動してください。")
    X = df.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    if fill_strategy == "minus_one":
        X = X.fillna(-1)
    else:
        X = X.fillna(X.median(numeric_only=True))
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(X)
    return pd.DataFrame({"umap_x": emb[:, 0], "umap_y": emb[:, 1]}, index=df.index)


@st.fragment
def render_analysis_dashboard(df: pd.DataFrame) -> None:
    st.subheader("🧭 UMAP 誤判定解析（Source B 別動隊抽出）")
    if df.empty:
        st.info("案件データがありません。")
        return

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 3:
        st.warning("UMAP に必要な数値列が不足しています。")
        return

    candidate_source_cols = ["source", "data_source", "result.source", "origin"]
    source_col = next((c for c in candidate_source_cols if c in df.columns), None)
    if source_col is None:
        df["source"] = "Unknown"
        source_col = "source"

    pred_col = next((c for c in ["result.pred_status", "pred_status", "predicted_status"] if c in df.columns), None)
    actual_col = next((c for c in ["final_status", "actual_status"] if c in df.columns), None)

    model_prob = _extract_model_prob(df)
    correctness = pd.Series("Unknown", index=df.index)
    if pred_col and actual_col:
        correctness = np.where(df[pred_col].astype(str) == df[actual_col].astype(str), "Correct", "Incorrect")
    df_plot = df.copy()
    df_plot["model_prob"] = model_prob
    df_plot["prediction_result"] = correctness

    st.caption("40次元までの数値特徴を利用（不足時は利用可能列のみ）。")
    max_dims = min(40, len(numeric_cols))
    dim_count = st.slider("UMAP入力次元数", 5, max_dims, min(40, max_dims), 1)
    fill_strategy = st.radio("欠損値処理", ["median", "minus_one"], format_func=lambda x: "中央値補完" if x == "median" else "-1埋め（欠損フラグ）", horizontal=True)
    run_toggle = st.toggle("UMAPを実行する", value=False)
    color_mode = st.radio("色分け", ["prediction_result", source_col], format_func=lambda x: "予測正誤" if x == "prediction_result" else "ソース", horizontal=True)

    proba_low, proba_high = st.slider("境界線データ (predict_proba)", 0.0, 1.0, (0.4, 0.6), 0.01)

    if not run_toggle:
        st.info("トグルをONにするとUMAPを計算・描画します。")
        return

    use_features = tuple(numeric_cols[:dim_count])
    embedding = compute_umap_projection(df_plot, use_features, fill_strategy=fill_strategy)
    vis = pd.concat([df_plot, embedding], axis=1)

    error_mask = vis["prediction_result"].eq("Incorrect")
    boundary_mask = vis["model_prob"].between(proba_low, proba_high, inclusive="both")
    squad_mask = error_mask | boundary_mask
    squad_df = vis.loc[squad_mask].copy()
    squad_df["special_label"] = "Source_B_Special"

    st.session_state["source_b_special_df"] = squad_df

    fig = px.scatter(
        vis,
        x="umap_x",
        y="umap_y",
        color=color_mode,
        hover_data=[c for c in ["id", "borrower_name", "industry_sub", "model_prob", "prediction_result"] if c in vis.columns],
        opacity=0.85,
        title="UMAP 2D Projection",
    )
    st.plotly_chart(fig, width='stretch')

    c1, c2, c3 = st.columns(3)
    c1.metric("総件数", f"{len(vis):,}")
    c2.metric("誤判定", f"{int(error_mask.sum()):,}")
    c3.metric("別動隊抽出", f"{len(squad_df):,}")

    with st.expander("別動隊データ（先頭50件）", expanded=False):
        cols = [c for c in ["id", "borrower_name", "industry_sub", source_col, "prediction_result", "model_prob", "special_label"] if c in squad_df.columns]
        st.dataframe(squad_df[cols].head(50), width='stretch', hide_index=True)


def render_umap_error_view() -> None:
    st.title("🗺️ UMAP誤判定マップ")
    df = _load_cases_df()
    render_analysis_dashboard(df)
