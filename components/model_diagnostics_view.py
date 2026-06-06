"""
Phase 0: モデル診断 — 棚卸し・可視化

- データ分布タブ: 業種別件数・成約率・主要変数分布
- 予測誤差タブ: 業種/規模/月別の誤差偏り
- 損失地形タブ: 2係数を変動させたAUCコンター
- UMAP誤判定マップ: 誤判定案件を2次元配置して偏りを確認
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

import joblib

from ai_chat import _gemini_chat
from config import GEMINI_MODEL_DEFAULT
from secret_manager import get_gemini_api_key
from runtime_paths import get_data_path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = get_data_path("lease_data.db")
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


def _review_date_text(case_json: dict, timestamp: str | None) -> str:
    for key in ("審査日", "shinsa_date", "review_date", "registration_date", "final_result_date"):
        raw = case_json.get(key)
        if raw:
            text = str(raw).strip()
            if text:
                return text[:10]
    raw_ts = str(timestamp or "").strip()
    return raw_ts[:10]


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
        review_date = _review_date_text(d, ts)
        display_status = "成約" if status in _WIN_STATUSES else status
        try:
            month = int(review_date[:7].split("-")[1])
        except (IndexError, ValueError):
            month = 0
        acost = float(inp.get("acquisition_cost") or 0)
        records.append({
            "id": case_id,
            "timestamp": ts,
            "review_date": review_date,
            "month": month,
            "final_status": display_status,
            "score": float(score or 0),
            "industry": d.get("industry_major", "不明"),
            "sales_dept": d.get("sales_dept", "未設定"),
            "nenshu": float(inp.get("nenshu") or 0),
            "acquisition_cost": acost,
            "acost_tier": _acost_tier(acost),
            "gross_profit": float(inp.get("gross_profit") or 0),
            "op_profit": float(inp.get("op_profit") or 0),
            "ord_profit": float(inp.get("ord_profit") or 0),
            "bank_credit": float(inp.get("bank_credit") or 0),
            "lease_credit": float(inp.get("lease_credit") or 0),
            "final_rate": float(d.get("final_rate") or 0),
            "score_borrower": float(res.get("score_borrower") or 0),
            "asset_score": float(res.get("asset_score") or 50),
            "label": 1 if status in _WIN_STATUSES else 0,
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df["month_key"] = pd.to_datetime(df["review_date"], errors="coerce").dt.strftime("%Y-%m")
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


def _get_misjudge_features(work: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = ["score", "nenshu", "acquisition_cost", "op_profit", "bank_credit", "lease_credit", "asset_score"]
    feat_df = work[feature_cols].copy().fillna(0.0)
    for col in ["nenshu", "acquisition_cost", "op_profit", "bank_credit", "lease_credit"]:
        feat_df[col] = np.log1p(np.clip(feat_df[col].astype(float), a_min=0, a_max=None))
    return feat_df, feature_cols


def _make_quantile_band(series: pd.Series, prefix: str, q: int = 4) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(["未読取"] * len(series), index=series.index)
    try:
        band = pd.qcut(values, q=min(q, int(values.nunique())), duplicates="drop")
        return band.astype(str).fillna("未読取")
    except Exception:
        return pd.Series(["未読取"] * len(series), index=series.index)


def _attach_segment_bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["売上高帯"] = _make_quantile_band(out["nenshu"], "売上高")
    out["売上総利益帯"] = _make_quantile_band(out["gross_profit"], "売上総利益")
    out["営業利益帯"] = _make_quantile_band(out["op_profit"], "営業利益")
    out["金利帯"] = _make_quantile_band(out["final_rate"], "金利")
    return out


def _misjudge_breakdown_table(df: pd.DataFrame, column: str, top_n: int = 8) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame()
    tmp = df[df["kind"].isin(["FP", "FN"])].copy()
    if tmp.empty:
        return pd.DataFrame()
    grp = (
        tmp.groupby(column)
        .agg(誤判定件数=("id", "count"), FP=("kind", lambda s: int((s == "FP").sum())), FN=("kind", lambda s: int((s == "FN").sum())))
        .reset_index()
        .sort_values(["誤判定件数", "FP", "FN"], ascending=False)
        .head(top_n)
    )
    grp = grp.rename(columns={column: "区分"})
    return grp


def _table_to_text(table: pd.DataFrame) -> str:
    if table.empty:
        return "データなし"
    return table.to_string(index=False)


def _call_misjudge_gemini(prompt: str) -> str:
    api_key = (st.session_state.get("gemini_api_key", "").strip() or get_gemini_api_key() or "").strip()
    if not api_key:
        return "Gemini APIキーが設定されていません。サイドバーの AIモデル設定で入力してください。"
    model = st.session_state.get("gemini_model", GEMINI_MODEL_DEFAULT) or GEMINI_MODEL_DEFAULT
    if "1.5" in model:
        model = GEMINI_MODEL_DEFAULT
    resp = _gemini_chat(
        api_key=api_key,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        timeout_seconds=120,
        max_output_tokens=1400,
    )
    return (resp.get("message", {}) or {}).get("content", "") or "Gemini から空の応答が返されました。"


def _import_umap_safely():
    try:
        import numba  # type: ignore
    except Exception:
        return None

    orig_jit = getattr(numba, "jit", None)
    orig_njit = getattr(numba, "njit", None)
    if orig_jit is None or orig_njit is None:
        return None

    def _disable_cache(decorator):
        def wrapper(*args, **kwargs):
            patched_kwargs = dict(kwargs)
            patched_kwargs["cache"] = False
            try:
                return decorator(*args, **patched_kwargs)
            except RuntimeError as exc:
                if "no locator available" in str(exc):
                    patched_kwargs["cache"] = False
                    return decorator(*args, **patched_kwargs)
                raise

        return wrapper

    patched_jit = _disable_cache(orig_jit)
    patched_njit = _disable_cache(orig_njit)
    numba.jit = patched_jit
    numba.njit = patched_njit
    nb_core_decorators = nb_ufuncbuilder = nb_dufunc = None
    try:
        import numba.core.decorators as nb_core_decorators  # type: ignore
        import numba.np.ufunc.ufuncbuilder as nb_ufuncbuilder  # type: ignore
        import numba.np.ufunc.dufunc as nb_dufunc  # type: ignore
        nb_core_decorators.jit = patched_jit
        nb_ufuncbuilder.jit = patched_jit
        nb_dufunc.jit = patched_jit
        import contextlib
        import io
        import os
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            import umap  # type: ignore
        return umap
    except Exception:
        return None
    finally:
        numba.jit = orig_jit
        numba.njit = orig_njit
        if nb_core_decorators is not None and orig_jit is not None:
            nb_core_decorators.jit = orig_jit
        if nb_ufuncbuilder is not None and orig_jit is not None:
            nb_ufuncbuilder.jit = orig_jit
        if nb_dufunc is not None and orig_jit is not None:
            nb_dufunc.jit = orig_jit


def _run_misjudge_embedding(work: pd.DataFrame, method: str) -> tuple[np.ndarray, str, dict[str, object]]:
    feat_df, feature_cols = _get_misjudge_features(work)
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(feat_df.to_numpy(dtype=float))
    meta: dict[str, object] = {"feature_cols": feature_cols}

    if method == "UMAP":
        try:
            umap = _import_umap_safely()
            if umap is None:
                raise ImportError("umap unavailable")
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, max(5, len(work) // 4)),
                min_dist=0.15,
                metric="euclidean",
                random_state=42,
            )
            meta["reducer"] = reducer
            return reducer.fit_transform(X), "UMAP", meta
        except Exception:
            pass
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2, random_state=42).fit(X)
    meta["reducer"] = reducer
    meta["explained_variance_ratio"] = getattr(reducer, "explained_variance_ratio_", None)
    meta["components"] = getattr(reducer, "components_", None)
    return reducer.transform(X), "PCA", meta


def _build_misjudge_frame(df: pd.DataFrame, threshold: float = 71.0, method: str = "UMAP", only_misjudge: bool = False, month_key: str | None = None) -> pd.DataFrame:
    work = df[df["final_status"].isin(_VALID_STATUSES)].copy()
    if month_key:
        work = work[work["month_key"] == month_key].copy()
    if len(work) < 10:
        return pd.DataFrame()

    embedding, reducer_name, reducer_meta = _run_misjudge_embedding(work, method)
    actual = work["label"].astype(int).to_numpy()
    pred = (work["score"].astype(float).to_numpy() >= threshold).astype(int)
    kind = np.where((pred == 1) & (actual == 1), "TP",
             np.where((pred == 0) & (actual == 0), "TN",
             np.where((pred == 1) & (actual == 0), "FP", "FN")))

    out = work[[
        "id", "timestamp", "review_date", "month_key", "final_status", "score",
        "industry", "sales_dept", "nenshu", "gross_profit", "op_profit", "final_rate",
        "acquisition_cost",
        "売上高帯", "売上総利益帯", "営業利益帯", "金利帯",
    ]].copy()
    out["x"] = embedding[:, 0]
    out["y"] = embedding[:, 1]
    out["actual"] = actual
    out["pred"] = pred
    out["kind"] = kind
    out["reducer"] = reducer_name
    out["threshold"] = threshold
    if only_misjudge:
        out = out[out["kind"].isin(["FP", "FN"])].copy()
    out.attrs["reducer_meta"] = reducer_meta
    return out


# ── Tab 3: UMAP 誤判定マップ ───────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def _compute_misjudge_map(df: pd.DataFrame, method: str = "UMAP", only_misjudge: bool = False, month_key: str | None = None):
    if df.empty:
        return None
    out = _build_misjudge_frame(df, method=method, only_misjudge=only_misjudge, month_key=month_key)
    if out.empty:
        return None
    return out


def _render_misjudge_map(df: pd.DataFrame) -> None:
    st.subheader("🧭 UMAP誤判定マップ")
    st.caption("成約/失注案件を2次元に配置し、誤判定の偏りを可視化します。赤=偽陽性、橙=偽陰性。")

    work_df = df[df["final_status"].isin(_VALID_STATUSES)].copy()
    work_df = _attach_segment_bands(work_df) if not work_df.empty else work_df
    segment_options = {
        "なし": None,
        "営業部": "sales_dept",
        "業種": "industry",
        "売上高帯": "売上高帯",
        "売上総利益帯": "売上総利益帯",
        "営業利益帯": "営業利益帯",
        "金利帯": "金利帯",
    }

    control1, control2, control3 = st.columns([1, 1, 2])
    with control1:
        method = st.selectbox("次元圧縮", ["UMAP", "PCA"], index=0, key="misjudge_method")
    with control2:
        mode = st.selectbox("表示範囲", ["全件", "誤判定のみ"], index=0, key="misjudge_only")
    with control3:
        month_keys = ["（全期間）"]
        if "month_key" in work_df.columns:
            month_keys.extend(sorted([m for m in work_df["month_key"].dropna().astype(str).unique().tolist() if m]))
        selected_month = st.selectbox("年月比較", month_keys, index=0, key="misjudge_month")

    seg1, seg2, seg3 = st.columns([1, 1, 1])
    with seg1:
        segment_key = st.selectbox("分割キー", list(segment_options.keys()), index=1, key="misjudge_segment_key")
    with seg2:
        segment_value = "全件"
        segment_col = segment_options.get(segment_key)
        if segment_col and segment_col in work_df.columns:
            seg_values = ["全件"] + sorted([v for v in work_df[segment_col].dropna().astype(str).unique().tolist() if v])
            segment_value = st.selectbox("分割値", seg_values, index=0, key="misjudge_segment_value")
    with seg3:
        st.caption("色は選択した分割キーで付けます。売上高= `nenshu`、利益は `売上総利益 / 営業利益`、金利は `final_rate` を帯分け")

    only_misjudge = mode == "誤判定のみ"
    month_key = None if selected_month == "（全期間）" else selected_month

    filtered_df = work_df
    segment_col = segment_options.get(segment_key)
    if segment_col and segment_col in filtered_df.columns and segment_value != "全件":
        filtered_df = filtered_df[filtered_df[segment_col].astype(str) == str(segment_value)].copy()

    map_df = _compute_misjudge_map(filtered_df, method=method, only_misjudge=only_misjudge, month_key=month_key)
    if map_df is None or map_df.empty:
        st.info("誤判定マップを作るには、成約/失注データが10件以上必要です。")
        return

    selected_method = method
    reducer_name = str(map_df["reducer"].iloc[0])
    threshold = float(map_df["threshold"].iloc[0])
    if selected_method == "UMAP" and reducer_name == "PCA":
        st.warning("UMAPを選択しましたが、環境上の都合でPCAにフォールバックしています。")
    if reducer_name == "PCA":
        st.caption(
            "PCA1 は元の案件データのばらつきを最も強く表す第1主成分、PCA2 はその次に強い独立したばらつきを表す第2主成分です。"
            " 2軸は元の項目を要約した見取り図なので、近い点ほど特徴が似た案件として見ます。"
        )
        reducer_meta = map_df.attrs.get("reducer_meta") or {}
        feature_cols = list(reducer_meta.get("feature_cols") or [])
        components = reducer_meta.get("components")
        explained = reducer_meta.get("explained_variance_ratio")
        if isinstance(components, np.ndarray) and len(feature_cols) == components.shape[1]:
            st.markdown("#### PCAの中身")
            if isinstance(explained, np.ndarray) and len(explained) >= 2:
                st.caption(
                    f"PCA1の寄与率: {float(explained[0]) * 100:.1f}% / "
                    f"PCA2の寄与率: {float(explained[1]) * 100:.1f}%"
                )
            rows = []
            for pc_idx, pc_name in enumerate(["PCA1", "PCA2"]):
                coefs = components[pc_idx]
                top_idx = np.argsort(np.abs(coefs))[::-1][:5]
                for rank, feat_idx in enumerate(top_idx, start=1):
                    coef = float(coefs[feat_idx])
                    feat = feature_cols[feat_idx]
                    rows.append({
                        "主成分": pc_name,
                        "順位": rank,
                        "項目": feat,
                        "係数": coef,
                        "影響": "正方向" if coef >= 0 else "負方向",
                    })
            if rows:
                coef_df = pd.DataFrame(rows)
                st.dataframe(
                    coef_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "係数": st.column_config.NumberColumn(format="%.3f"),
                    },
                )
    st.caption(f"選択: {selected_method} / 実行: {reducer_name}")
    counts = map_df["kind"].value_counts().to_dict()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TP", int(counts.get("TP", 0)))
    c2.metric("TN", int(counts.get("TN", 0)))
    c3.metric("FP", int(counts.get("FP", 0)))
    c4.metric("FN", int(counts.get("FN", 0)))

    color_map = {
        "TP": "#2563eb",
        "TN": "#16a34a",
        "FP": "#ef4444",
        "FN": "#f59e0b",
    }
    label_map = {
        "TP": "正解:成約",
        "TN": "正解:失注",
        "FP": "誤判定:過大評価",
        "FN": "誤判定:過小評価",
    }
    if segment_col:
        color_field = segment_col if segment_col in map_df.columns else "kind"
        color_discrete_map = None
        if color_field == "sales_dept":
            color_discrete_map = {
                "宇都宮営業部": "#2563eb",
                "小山営業部": "#f97316",
                "足利営業部": "#16a34a",
                "埼玉営業部": "#7c3aed",
                "未設定": "#64748b",
                "未読取": "#64748b",
                "0": "#64748b",
            }
        elif color_field == "industry":
            color_discrete_map = {
                "D 建設業": "#2563eb",
                "E 製造業": "#f97316",
                "H 運輸業・郵便業": "#16a34a",
                "I 卸売業・小売業": "#7c3aed",
                "J 金融業・保険業": "#db2777",
                "K 不動産業・物品賃貸業": "#0891b2",
                "M 宿泊業・飲食サービス業": "#e11d48",
                "N 生活関連サービス業・娯楽業": "#0f766e",
                "O 教育・学習支援業": "#ca8a04",
                "P 医療・福祉": "#dc2626",
                "R サービス業": "#334155",
            }
        fig = px.scatter(
            map_df,
            x="x",
            y="y",
            color=color_field,
            symbol="kind",
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=px.colors.qualitative.Dark24 + px.colors.qualitative.Light24,
            hover_data={
                "id": True,
                "sales_dept": True,
                "industry": True,
                "売上高帯": True,
                "売上総利益帯": True,
                "営業利益帯": True,
                "金利帯": True,
                "score": ":.1f",
                "final_status": True,
                "nenshu": ":,.1f",
                "op_profit": ":,.1f",
                "final_rate": ":.2f",
                "x": False,
                "y": False,
                "kind": False,
                "month_key": False,
                "review_date": False,
            },
            title=f"{reducer_name} 誤判定マップ（閾値 {threshold:.0f}）",
        )
        fig.update_traces(marker=dict(size=10, opacity=0.9, line=dict(width=1.5, color="#111827")))
    else:
        fig = go.Figure()
        for kind in ["TP", "TN", "FP", "FN"]:
            sub = map_df[map_df["kind"] == kind]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["x"],
                y=sub["y"],
                mode="markers",
                name=label_map[kind],
                marker=dict(size=10 if kind in ("FP", "FN") else 8, color=color_map[kind], opacity=0.8, line=dict(width=1, color="#0f172a")),
                text=[
                    f"ID:{r.id}<br>営業部:{r.sales_dept}<br>業種:{r.industry}<br>スコア:{r.score:.1f}<br>判定:{r.final_status}<br>売上高:{r.nenshu:,.1f}<br>利益:{r.op_profit:,.1f}<br>金利:{r.final_rate:.2f}%"
                    for r in sub.itertuples(index=False)
                ],
                hovertemplate="%{text}<extra></extra>",
            ))
    fig.update_layout(
        title=f"{reducer_name} 誤判定マップ（閾値 {threshold:.0f}）",
        xaxis_title=f"{reducer_name} 1",
        yaxis_title=f"{reducer_name} 2",
        height=560,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Gemini説明")
    st.caption("今見ている条件の誤判定の塊を、業種・規模・利益・金利の観点で要約します。")
    gem_key = f"misjudge_gemini::{segment_key}::{segment_value}::{method}::{month_key or 'all'}::{only_misjudge}"
    if st.button("🤖 Geminiで説明", key=f"misjudge_gemini_btn::{segment_key}::{segment_value}::{method}::{month_key or 'all'}"):
        industry_table = _misjudge_breakdown_table(map_df, "industry")
        rate_table = _misjudge_breakdown_table(map_df, "金利帯")
        sales_table = _misjudge_breakdown_table(map_df, "売上高帯")
        profit_table = _misjudge_breakdown_table(map_df, "営業利益帯")
        gross_table = _misjudge_breakdown_table(map_df, "売上総利益帯")
        pca_summary = ""
        reducer_meta = map_df.attrs.get("reducer_meta") or {}
        if reducer_name == "PCA":
            feature_cols = list(reducer_meta.get("feature_cols") or [])
            components = reducer_meta.get("components")
            explained = reducer_meta.get("explained_variance_ratio")
            if isinstance(components, np.ndarray) and len(feature_cols) == components.shape[1]:
                parts = []
                if isinstance(explained, np.ndarray) and len(explained) >= 2:
                    parts.append(
                        f"PCA1寄与率={float(explained[0]) * 100:.1f}%、PCA2寄与率={float(explained[1]) * 100:.1f}%"
                    )
                for pc_idx, pc_name in enumerate(["PCA1", "PCA2"]):
                    coefs = components[pc_idx]
                    top_idx = np.argsort(np.abs(coefs))[::-1][:3]
                    top_terms = []
                    for feat_idx in top_idx:
                        coef = float(coefs[feat_idx])
                        feat = feature_cols[feat_idx]
                        top_terms.append(f"{feat}({coef:+.3f})")
                    if top_terms:
                        parts.append(f"{pc_name}: " + " / ".join(top_terms))
                pca_summary = "\n".join(parts)
        segment_desc = segment_key if segment_value == "全件" else f"{segment_key}={segment_value}"
        prompt = f"""あなたはリース審査の実績分析担当です。以下は誤判定マップの分析です。

## 条件
- 対象: {segment_desc}
- 次元圧縮: {selected_method}
- 実行結果: {reducer_name}
- 表示範囲: {"誤判定のみ" if only_misjudge else "全件"}
- 対象月: {month_key or "全期間"}
- 閾値: {threshold:.0f}
- TP: {int(counts.get("TP", 0))}, TN: {int(counts.get("TN", 0))}, FP: {int(counts.get("FP", 0))}, FN: {int(counts.get("FN", 0))}

## 業種別誤判定上位
{_table_to_text(industry_table)}

## 売上高帯別誤判定上位
{_table_to_text(sales_table)}

## 売上総利益帯別誤判定上位
{_table_to_text(gross_table)}

## 営業利益帯別誤判定上位
{_table_to_text(profit_table)}

## 金利帯別誤判定上位
{_table_to_text(rate_table)}

## PCAの中身（PCAのときのみ参考）
{pca_summary or "PCAではない、または係数情報がありません。"}

## 依頼
1. この条件で誤判定が固まる理由を、業種・規模・利益・金利の観点で要約してください。
2. FP と FN のどちらが多いかを踏まえて、モデルが過大評価しやすいか過小評価しやすいか述べてください。
3. 現場が次に確認すべき項目を3つに絞ってください。
4. 断定しすぎず、データから言える範囲で簡潔にまとめてください。
"""
        with st.spinner("Gemini に分析を依頼中…"):
            result = _call_misjudge_gemini(prompt)
        st.session_state[gem_key] = result
        st.rerun()
    result = st.session_state.get(gem_key)
    if result:
        st.markdown("##### Geminiコメント")
        st.markdown(result)

    fp_fn = map_df[map_df["kind"].isin(["FP", "FN"])].copy()
    if not fp_fn.empty:
        st.markdown("#### 誤判定案件一覧")
        show = fp_fn.sort_values(["kind", "score"], ascending=[True, False]).head(20).copy()
        show["kind"] = show["kind"].map(label_map)
        show = show.rename(columns={
            "kind": "判定",
            "final_status": "実績",
            "score": "スコア",
            "sales_dept": "営業部",
            "industry": "業種",
            "nenshu": "売上高",
            "op_profit": "利益",
            "final_rate": "金利",
        })
        st.dataframe(
            show[["判定", "id", "営業部", "業種", "実績", "スコア", "売上高", "利益", "金利", "timestamp"]],
            width="stretch",
            hide_index=True,
        )

    if month_key is None and "month_key" in filtered_df.columns:
        st.markdown("#### 年月比較")
        months = sorted([m for m in filtered_df["month_key"].dropna().astype(str).unique().tolist() if m])
        compare_months = months[-3:] if len(months) >= 3 else months
        if len(compare_months) >= 2:
            compare_fig = go.Figure()
            for month in compare_months:
                sub = _compute_misjudge_map(filtered_df, method=method, only_misjudge=True, month_key=month)
                if sub is None or sub.empty:
                    continue
                compare_fig.add_trace(go.Scatter(
                    x=sub["x"],
                    y=sub["y"],
                    mode="markers",
                    name=month,
                    text=[f"{r.id}<br>{r.industry}<br>{r.final_status}<br>スコア:{r.score:.1f}" for r in sub.itertuples(index=False)],
                    hovertemplate="%{text}<extra></extra>",
                    marker=dict(size=9, opacity=0.75),
                ))
            compare_fig.update_layout(
                title=f"誤判定のみの年月比較（{method}）",
                xaxis_title=f"{method} 1",
                yaxis_title=f"{method} 2",
                height=420,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(compare_fig, use_container_width=True)

    if segment_col == "sales_dept" and segment_value != "全件":
        st.markdown(f"#### {segment_value} の誤判定内訳")
        st.caption("FP = 実際は失注なのに成約と予測した件。FN = 実際は成約なのに失注と予測した件。")
        breakdown_cols = [
            ("業種", "industry"),
            ("売上高帯", "売上高帯"),
            ("売上総利益帯", "売上総利益帯"),
            ("営業利益帯", "営業利益帯"),
            ("金利帯", "金利帯"),
        ]
        for title, col in breakdown_cols:
            table = _misjudge_breakdown_table(map_df, col)
            if table.empty:
                continue
            st.markdown(f"##### {title}")
            st.dataframe(table, width="stretch", hide_index=True)


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


# ── Tab 5: 特徴量重要度 ───────────────────────────────────────────────────────

_LGBM_PKL_PATH = os.path.join(_REPO_ROOT, "data", "lgbm_contract_model.pkl")


@st.cache_data(show_spinner=False, ttl=600)
def _load_lgbm_bundle() -> dict | None:
    """lgbm_contract_model.pkl を読み込んで辞書を返す。失敗時は None。"""
    if not os.path.exists(_LGBM_PKL_PATH):
        return None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle = joblib.load(_LGBM_PKL_PATH)
        return bundle if isinstance(bundle, dict) else {"model": bundle, "feature_names": [], "n_cases": 0, "auc": 0, "accuracy": 0, "importance": []}
    except Exception:
        return None


def _render_feature_importance() -> None:
    bundle = _load_lgbm_bundle()
    if bundle is None:
        st.warning(f"モデルファイルが見つかりません: {_LGBM_PKL_PATH}")
        return

    model_obj = bundle.get("model")
    if model_obj is None or not hasattr(model_obj, "feature_importances_"):
        st.warning("feature_importances_ を持つモデルが見つかりません。")
        return

    feature_names: list[str] = bundle.get("feature_names") or []
    importances = model_obj.feature_importances_

    # feature_names が空または長さ不一致のときはダミー名を使う
    if len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    n_cases: int = bundle.get("n_cases", 0)
    auc: float = bundle.get("auc", 0.0)
    accuracy: float = bundle.get("accuracy", 0.0)
    model_class: str = model_obj.__class__.__name__
    n_estimators = getattr(model_obj, "n_estimators", "?")
    max_depth = getattr(model_obj, "max_depth", "?")

    # ── モデル情報メトリクス ──────────────────────────────────────────────────
    st.subheader("本体モデル情報")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("モデル種別", model_class)
    c2.metric("学習件数", f"{n_cases:,} 件")
    c3.metric("AUC", f"{auc:.4f}")
    c4.metric("Accuracy", f"{accuracy:.3f}")
    c5.metric("n_estimators / max_depth", f"{n_estimators} / {max_depth if max_depth else 'None'}")

    st.divider()

    # 上位 N 件のスライダー
    top_n = st.slider("表示する特徴量の数", min_value=5, max_value=min(39, len(importances)), value=20, step=1, key="feat_imp_top_n")

    # 降順ソートして上位 top_n を取得
    sorted_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    top_pairs = sorted_pairs[:top_n]
    names_all = [p[0] for p in sorted_pairs]
    vals_all = [float(p[1]) for p in sorted_pairs]
    names_top = [p[0] for p in top_pairs]
    vals_top = [float(p[1]) for p in top_pairs]

    # ── 2カラム表示: RF重要度 | LR係数 ───────────────────────────────────────
    col_rf, col_lr = st.columns([3, 2])

    with col_rf:
        st.markdown(f"#### {model_class} 特徴量重要度（上位 {top_n} 件）")
        # 横棒グラフ: 上位が上に来るよう逆順にする
        fig_rf = go.Figure(go.Bar(
            x=vals_top[::-1],
            y=names_top[::-1],
            orientation="h",
            marker=dict(
                color=vals_top[::-1],
                colorscale="Blues",
                showscale=False,
            ),
            text=[f"{v:.1f}" for v in vals_top[::-1]],
            textposition="outside",
        ))
        fig_rf.update_layout(
            height=max(350, top_n * 22),
            margin=dict(l=10, r=60, t=10, b=20),
            xaxis_title="重要度スコア",
            yaxis_title="",
        )
        st.plotly_chart(fig_rf, use_container_width=True)

    with col_lr:
        st.markdown("#### ロジスティック回帰 係数（参照用）")
        st.caption("scoring_core のロジスティック回帰係数（全体_既存先）。RFと比べて重視する特徴量の差異を確認できます。")
        try:
            from data_cases import get_effective_coeffs
            lr_coeffs: dict = get_effective_coeffs("全体_既存先") or {}
            if lr_coeffs:
                # RF の特徴量名と重なる係数のみ抽出し、絶対値で降順
                lr_rows = []
                for feat in names_all:
                    # 係数キーは feature_names と同名のものを探す
                    coef_val = lr_coeffs.get(feat)
                    if coef_val is not None:
                        lr_rows.append({"特徴量": feat, "係数": float(coef_val)})
                # RF になくても lr_coeffs に存在するキーも追加
                for k, v in lr_coeffs.items():
                    if k not in names_all and k != "intercept":
                        lr_rows.append({"特徴量": k, "係数": float(v)})

                if lr_rows:
                    lr_df = pd.DataFrame(lr_rows).sort_values("係数", key=abs, ascending=False)
                    # 横棒グラフ（正負で色分け）
                    fig_lr = go.Figure(go.Bar(
                        x=lr_df["係数"].tolist()[::-1],
                        y=lr_df["特徴量"].tolist()[::-1],
                        orientation="h",
                        marker_color=[
                            "#2563eb" if v >= 0 else "#ef4444"
                            for v in lr_df["係数"].tolist()[::-1]
                        ],
                        text=[f"{v:+.3f}" for v in lr_df["係数"].tolist()[::-1]],
                        textposition="outside",
                    ))
                    fig_lr.update_layout(
                        height=max(350, len(lr_df) * 22),
                        margin=dict(l=10, r=60, t=10, b=20),
                        xaxis_title="係数値",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig_lr, use_container_width=True)
                else:
                    st.info("RF の特徴量名と一致するLR係数が見つかりませんでした。")
                    # 全 LR 係数をテーブルで表示
                    raw_df = pd.DataFrame([{"係数キー": k, "値": v} for k, v in lr_coeffs.items() if k != "intercept"])
                    raw_df = raw_df.sort_values("値", key=abs, ascending=False)
                    st.dataframe(raw_df, hide_index=True, use_container_width=True)
            else:
                st.info("ロジスティック回帰係数が見つかりませんでした（DB 未設定の可能性）。")
        except Exception as exc:
            st.warning(f"LR係数の読み込みに失敗しました: {exc}")

    st.divider()

    # ── 全特徴量の重要度テーブル ──────────────────────────────────────────────
    with st.expander("全特徴量の重要度一覧", expanded=False):
        all_df = pd.DataFrame({
            "順位": range(1, len(names_all) + 1),
            "特徴量": names_all,
            "重要度": [round(v, 2) for v in vals_all],
            "累積寄与率(%)": [round(sum(vals_all[:i+1]) / sum(vals_all) * 100, 1) for i in range(len(vals_all))],
        })
        st.dataframe(
            all_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "重要度": st.column_config.NumberColumn(format="%.2f"),
                "累積寄与率(%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            },
        )


# ── メインエントリ ────────────────────────────────────────────────────────────

def render_model_diagnostics() -> None:
    st.title("🔭 モデル診断 — Phase 0 棚卸し・可視化")
    st.caption("学習データの分布・予測誤差の偏り・損失地形を俯瞰し、改善優先箇所を特定する。")

    df = _load_df()
    if df.empty:
        st.error("DBが見つかりません。")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 データ分布", "⚖️ 予測誤差の偏り", "🧭 UMAP誤判定", "🗺️ 損失地形", "🎯 特徴量重要度"])

    with tab1:
        _render_distribution(df)
    with tab2:
        _render_bias(df)
    with tab3:
        _render_misjudge_map(df)
    with tab4:
        _render_landscape()
    with tab5:
        _render_feature_importance()
