"""
CliffordNet 幾何代数ビジュアル分析。既存のスコアロジックには一切触れない。
"""
import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from runtime_paths import get_data_path

try:
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

try:
    from scipy.spatial import ConvexHull

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_BASE = Path(__file__).parent.parent
_DB_PATH = Path(get_data_path("screening_db.sqlite"))
_LEASE_DB_PATH = Path(get_data_path("lease_data.db"))
_VIS_LIMIT = 500

# 6 indicators available from both DB and last_result
_IND_LABELS = [
    "営業利益率(%)",
    "自己資本比率(%)",
    "資産回転率",
    "スコア",
    "成約確率(%)",
    "ROA(%)",
]

# reasonable normalization ranges for radar
_IND_RANGES = [
    (-20.0, 30.0),   # 営業利益率
    (0.0, 80.0),     # 自己資本比率
    (0.0, 3.0),      # 資産回転率
    (0.0, 100.0),    # スコア
    (0.0, 100.0),    # 成約確率
    (-5.0, 20.0),    # ROA
]


# ── CliffordNet 計算クラス ──────────────────────────────────────────────────────

class CliffordVisualizer:
    def compute_inner_product(self, u: np.ndarray, v: np.ndarray) -> float:
        """内積：指標間の整合性スコア"""
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8))

    def compute_wedge_product(self, u: np.ndarray, v: np.ndarray) -> float:
        """外積（ウェッジ積）：指標間の構造的ポテンシャル（面積）"""
        u, v = np.asarray(u, float), np.asarray(v, float)
        if len(u) == 2:
            raw = abs(float(u[0] * v[1] - u[1] * v[0]))
        else:
            raw = float(np.linalg.norm(np.cross(u[:3], v[:3])))
        return raw / (float(np.linalg.norm(u)) * float(np.linalg.norm(v)) + 1e-8)

    def compute_ggr(self, base_score: float, geo_update: float, alpha: float = 0.0) -> float:
        """Gated Geometric Residual（α=0でスコア変更なし、可視化のみ）"""
        gate = 1.0 / (1.0 + np.exp(-base_score / 100.0))
        return base_score + alpha * gate * geo_update


# ── データ取得・加工 ────────────────────────────────────────────────────────────

def _count_rows(db_path: Path, table: str) -> int:
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
        n = int(cur.fetchone()[0] or 0)
        conn.close()
        return n
    except Exception:
        return 0


def _load_past_cases(limit: int = _VIS_LIMIT) -> pd.DataFrame:
    """lease_data.db の past_cases から screening_records 互換 DataFrame を構築"""
    import json as _json
    if not _LEASE_DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(_LEASE_DB_PATH))
        query = "SELECT id, timestamp, score, user_eq, final_status, data FROM past_cases ORDER BY timestamp DESC"
        if limit and limit > 0:
            query += f" LIMIT {int(limit)}"
        df_raw = pd.read_sql_query(query, conn)
        conn.close()
        if len(df_raw) == 0:
            return pd.DataFrame()
        rows = []
        for _, row in df_raw.iterrows():
            try:
                d = _json.loads(row["data"]) if isinstance(row["data"], str) else {}
                result = d.get("result", {})
                fin = result.get("financials", {})
                rows.append({
                    "created_at": row["timestamp"],
                    "revenue_m": fin.get("nenshu", 0),
                    "op_profit_m": fin.get("op_profit", 0),
                    "total_assets_m": fin.get("assets", 0),
                    "net_assets_m": fin.get("net_assets", 0),
                    "equity_ratio": result.get("user_eq", row.get("user_eq", 0)),
                    "score": result.get("score", row["score"]),
                    "contract_prob": result.get("contract_prob", 0) / 100.0,
                    "judgment": result.get("hantei", ""),
                    "industry_major": result.get("industry_major", d.get("industry_major", "")),
                    "industry_sub": row.get("industry_sub") or result.get("industry_sub", ""),
                })
            except Exception:
                continue
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _load_db() -> pd.DataFrame:
    if not _DB_PATH.exists():
        return _load_past_cases()
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        query = "SELECT * FROM screening_records ORDER BY created_at DESC"
        query += f" LIMIT {_VIS_LIMIT}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if len(df) == 0:
            return _load_past_cases()
        return df
    except Exception:
        return _load_past_cases()


def _db_to_indicators(df: pd.DataFrame) -> np.ndarray:
    """DB 行から 6 次元指標行列 (N×6) を計算"""
    rev = df["revenue_m"].replace(0, np.nan)
    assets = df["total_assets_m"].replace(0, np.nan)
    net = df.get("net_assets_m", pd.Series(np.zeros(len(df)))).fillna(0)
    score = df["score"].fillna(0)
    cp = df["contract_prob"].fillna(0) * 100

    op_margin = (df["op_profit_m"] / rev * 100).fillna(0).clip(-100, 100)
    eq_ratio = df["equity_ratio"].fillna(0).clip(0, 100)
    asset_turn = (rev / assets).fillna(0).clip(0, 10)
    roa = (net / assets * 100).fillna(0).clip(-50, 50)

    return np.column_stack([
        op_margin,
        eq_ratio,
        asset_turn,
        score,
        cp.clip(0, 100),
        roa,
    ])


def _classify(df: pd.DataFrame) -> np.ndarray:
    """1=成約圏, 0=その他"""
    labels = np.zeros(len(df), dtype=int)
    if "judgment" in df.columns:
        labels[df["judgment"].fillna("").str.contains("承認")] = 1
    if "contract_prob" in df.columns:
        labels[df["contract_prob"].fillna(0) >= 0.65] = 1
    return labels


def _current_vec(res: dict) -> np.ndarray | None:
    """last_result から 6 次元指標ベクトルを抽出"""
    try:
        return np.array([
            float(np.clip(res.get("user_op") or 0, -100, 100)),
            float(np.clip(res.get("user_eq") or 0, 0, 100)),
            float(np.clip(res.get("user_asset_turnover") or 0, 0, 10)),
            float(res.get("score") or 0),
            float(np.clip((res.get("contract_prob") or 0) * 100, 0, 100)),
            float(np.clip(res.get("user_roa") or 0, -50, 50)),
        ])
    except Exception:
        return None


def _normalize(vec: np.ndarray) -> np.ndarray:
    """各指標を _IND_RANGES で [0,1] に正規化"""
    out = np.zeros(len(_IND_RANGES))
    for i, (lo, hi) in enumerate(_IND_RANGES):
        out[i] = float(np.clip((vec[i] - lo) / (hi - lo + 1e-8), 0.0, 1.0))
    return out


# ── タブ描画 ───────────────────────────────────────────────────────────────────

def _tab_golden_area(
    df: pd.DataFrame,
    ind: np.ndarray,
    labels: np.ndarray,
    res: dict | None,
) -> None:
    st.subheader("🟡 成約の黄金面積マップ")
    st.caption(
        "営業利益率 × 自己資本比率 の 2D 空間に各案件をプロット。"
        "成約圏の凸包（黄金ゾーン）を半透明ポリゴンで描画します。"
    )

    if len(df) == 0:
        st.info("案件データがありません。審査を実行して蓄積してからお使いください。")
        return

    if not _HAS_PLOTLY:
        st.warning("plotly がインストールされていません。")
        return

    x = ind[:, 0]  # 営業利益率
    y = ind[:, 1]  # 自己資本比率
    viz = CliffordVisualizer()

    fig = go.Figure()

    # ── 成約圏 ──
    mask_ok = labels == 1
    if mask_ok.any():
        fig.add_trace(go.Scatter(
            x=x[mask_ok], y=y[mask_ok],
            mode="markers",
            name="成約圏",
            marker=dict(color="royalblue", size=9, opacity=0.75, symbol="circle"),
            hovertemplate="<b>成約圏</b><br>営業利益率: %{x:.1f}%<br>自己資本比率: %{y:.1f}%<extra></extra>",
        ))

        # 凸包（黄金ゾーン）
        pts = np.column_stack([x[mask_ok], y[mask_ok]])
        pts = pts[np.isfinite(pts).all(axis=1)]
        if _HAS_SCIPY and len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                vx = list(pts[hull.vertices, 0]) + [pts[hull.vertices[0], 0]]
                vy = list(pts[hull.vertices, 1]) + [pts[hull.vertices[0], 1]]
                fig.add_trace(go.Scatter(
                    x=vx, y=vy,
                    fill="toself",
                    fillcolor="rgba(255,215,0,0.18)",
                    line=dict(color="gold", width=2, dash="dash"),
                    mode="lines",
                    name="黄金ゾーン（成約凸包）",
                    hoverinfo="skip",
                ))
            except Exception:
                pass

        # 成約群の平均ウェッジ積
        pts_finite = pts[np.isfinite(pts).all(axis=1)]
        n_sample = min(60, len(pts_finite))
        if n_sample >= 2:
            idx = np.random.choice(len(pts_finite), n_sample, replace=False)
            w_vals = [
                viz.compute_wedge_product(pts_finite[i], pts_finite[j])
                for i, j in combinations(idx, 2)
                if i != j
            ]
            if w_vals:
                st.metric(
                    "成約群 平均ウェッジ積（黄金面積指数）",
                    f"{np.mean(w_vals):.4f}",
                    help="成約圏の指標ベクトル間の平均外積。大きいほど財務構造の多様性が高い。",
                )

    # ── 要審議圏 ──
    mask_ng = labels == 0
    if mask_ng.any():
        fig.add_trace(go.Scatter(
            x=x[mask_ng], y=y[mask_ng],
            mode="markers",
            name="要審議圏",
            marker=dict(color="salmon", size=7, opacity=0.55, symbol="circle"),
            hovertemplate="<b>要審議圏</b><br>営業利益率: %{x:.1f}%<br>自己資本比率: %{y:.1f}%<extra></extra>",
        ))

    # ── 現在案件 ──
    cv = _current_vec(res) if res else None
    if cv is not None:
        fig.add_trace(go.Scatter(
            x=[cv[0]], y=[cv[1]],
            mode="markers+text",
            name="現在案件",
            marker=dict(color="gold", size=20, symbol="star", line=dict(color="black", width=1.5)),
            text=["現在案件"],
            textposition="top right",
        ))

    fig.update_layout(
        xaxis_title="営業利益率 (%)",
        yaxis_title="自己資本比率 (%)",
        height=500,
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=40, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _tab_ggr_arrows(
    df: pd.DataFrame,
    ind: np.ndarray,
    labels: np.ndarray,
    res: dict | None,
) -> None:
    st.subheader("🏹 GGR 補正ベクトルアロー")
    st.caption(
        "各指標ペアのウェッジ積から算出した幾何学的補正ポテンシャル。"
        "**現行 α=0 のためスコアは変更されません**。α=0.2 の場合の参考値として表示します。"
    )

    cv = _current_vec(res) if res else None
    if cv is None:
        st.info("審査を実行してから開いてください。")
        return

    if not _HAS_PLOTLY:
        st.warning("plotly がインストールされていません。")
        return

    viz = CliffordVisualizer()
    base_score = float(cv[3])

    # ── 指標ペアのウェッジ積 ──
    # DB の全指標平均をリファレンスとして使い、現在案件とのウェッジを計算
    if len(ind) > 0:
        db_mean = ind.mean(axis=0)
    else:
        db_mean = np.zeros(len(_IND_LABELS))

    pair_labels, wedge_vals = [], []
    for i, j in combinations(range(len(_IND_LABELS)), 2):
        u = np.array([cv[i], db_mean[i]])
        v = np.array([cv[j], db_mean[j]])
        w = viz.compute_wedge_product(u, v)
        pair_labels.append(f"{_IND_LABELS[i][:5]} × {_IND_LABELS[j][:5]}")
        wedge_vals.append(w)

    wedge_arr = np.array(wedge_vals)
    geo_update = float(np.mean(wedge_arr)) * 100.0  # スコア空間にスケール

    ggr_ref = viz.compute_ggr(base_score, geo_update, alpha=0.2)
    delta = ggr_ref - base_score

    # ── アロー図（スコア軸） ──
    fig_arrow = go.Figure()

    # 背景: DB 案件の (スコア, 成約確率) 散点
    if len(df) > 0:
        fig_arrow.add_trace(go.Scatter(
            x=ind[:, 3], y=ind[:, 4],
            mode="markers",
            name="DB 案件",
            marker=dict(
                color=labels.tolist(),
                colorscale=[[0, "salmon"], [1, "royalblue"]],
                size=6,
                opacity=0.4,
            ),
            hoverinfo="skip",
        ))

    # 現在案件
    fig_arrow.add_trace(go.Scatter(
        x=[base_score], y=[cv[4]],
        mode="markers+text",
        name=f"現在 ({base_score:.1f}点)",
        marker=dict(color="gold", size=18, symbol="star", line=dict(color="black", width=1.5)),
        text=[f"現在<br>{base_score:.1f}点"],
        textposition="top right",
    ))

    # GGR 参考点（α=0.2）
    fig_arrow.add_trace(go.Scatter(
        x=[ggr_ref], y=[cv[4]],
        mode="markers+text",
        name=f"GGR参考 α=0.2 ({ggr_ref:.1f}点)",
        marker=dict(color="darkorange", size=14, symbol="diamond"),
        text=[f"GGR参考<br>{ggr_ref:.1f}点"],
        textposition="top left",
    ))

    # アロー
    fig_arrow.add_annotation(
        x=ggr_ref, y=cv[4],
        ax=base_score, ay=cv[4],
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=3, arrowsize=1.5,
        arrowcolor="darkorange",
        arrowwidth=2.5,
    )

    fig_arrow.update_layout(
        xaxis_title="スコア (点)",
        yaxis_title="成約確率 (%)",
        height=380,
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=40, r=20, t=20, b=20),
        annotations=[dict(
            x=0.02, y=0.97, xref="paper", yref="paper",
            text=f"<b>GGR 補正幅 (α=0.2 参考): {delta:+.2f} 点</b>",
            showarrow=False,
            font=dict(size=13, color="darkorange"),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="darkorange",
            borderwidth=1,
        )],
    )
    st.plotly_chart(fig_arrow, use_container_width=True)

    # ── ペアごとのウェッジ積 棒グラフ ──
    sorted_idx = np.argsort(wedge_arr)[::-1]
    fig_bar = go.Figure(go.Bar(
        x=[pair_labels[i] for i in sorted_idx],
        y=[wedge_vals[i] for i in sorted_idx],
        marker_color=[
            f"rgba(255,165,0,{0.4 + 0.6 * (1 - k / len(sorted_idx))})"
            for k in range(len(sorted_idx))
        ],
        hovertemplate="%{x}<br>ウェッジ積: %{y:.4f}<extra></extra>",
    ))
    fig_bar.update_layout(
        title_text="指標ペア別 ウェッジ積（幾何ポテンシャル）",
        xaxis_tickangle=-45,
        yaxis_title="ウェッジ積",
        height=340,
        margin=dict(l=40, r=20, t=40, b=100),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


def _tab_heatmap(df: pd.DataFrame, ind: np.ndarray) -> None:
    st.subheader("🌡️ 指標ペア整合性ヒートマップ")
    st.caption(
        "全 DB 案件の財務指標ベクトルについて、各ペアの内積（整合性スコア）を表示します。"
        "緑=整合・赤=逆行。逆行しているペアが財務上の矛盾点を示します。"
    )

    if len(df) < 3:
        st.info("ヒートマップ表示には 3 件以上のデータが必要です。")
        return

    if not _HAS_PLOTLY:
        st.warning("plotly がインストールされていません。")
        return

    viz = CliffordVisualizer()
    n = len(_IND_LABELS)
    mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
            else:
                u = ind[:, i]
                v = ind[:, j]
                mat[i, j] = viz.compute_inner_product(u, v)

    short = [lbl.replace("(%)", "").replace("率", "率\n") for lbl in _IND_LABELS]

    fig = go.Figure(go.Heatmap(
        z=mat,
        x=short,
        y=short,
        colorscale=[
            [0.0, "#d62728"],   # 赤（逆行）
            [0.5, "#f0f0f0"],   # 白（中立）
            [1.0, "#2ca02c"],   # 緑（整合）
        ],
        zmin=-1.0,
        zmax=1.0,
        text=np.round(mat, 3),
        texttemplate="%{text}",
        hovertemplate="<b>%{x} × %{y}</b><br>整合性: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=480,
        margin=dict(l=60, r=20, t=20, b=60),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 逆行ペアのハイライト
    threshold = -0.3
    rev_pairs = [
        (f"{_IND_LABELS[i]} ↔ {_IND_LABELS[j]}", round(mat[i, j], 3))
        for i in range(n) for j in range(i + 1, n)
        if mat[i, j] < threshold
    ]
    if rev_pairs:
        st.warning("**逆行ペア（整合性スコア < −0.3）**")
        for name, val in sorted(rev_pairs, key=lambda x: x[1]):
            st.markdown(f"- {name}: **{val}**")
    else:
        st.success("逆行している指標ペアはありません（整合性良好）。")


def _tab_radar(
    df: pd.DataFrame,
    ind: np.ndarray,
    labels: np.ndarray,
    res: dict | None,
) -> None:
    st.subheader("🎯 成約クラスター近接度")
    st.caption(
        "現在案件の各指標が成約成功群の「黄金面積」にどれだけ近いかをレーダーチャートで表示します。"
    )

    cv = _current_vec(res) if res else None
    if cv is None:
        st.info("審査を実行してから開いてください。")
        return

    if not _HAS_PLOTLY:
        st.warning("plotly がインストールされていません。")
        return

    mask_ok = labels == 1
    if mask_ok.sum() < 2:
        st.info("成約圏のデータが 2 件以上必要です。審査案件を蓄積してからお使いください。")
        return

    approved_ind = ind[mask_ok]
    centroid = approved_ind.mean(axis=0)

    cv_norm = _normalize(cv)
    centroid_norm = _normalize(centroid)

    cats = _IND_LABELS + [_IND_LABELS[0]]  # 閉じるため先頭を末尾に追加
    cv_vals = list(cv_norm) + [cv_norm[0]]
    ct_vals = list(centroid_norm) + [centroid_norm[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=ct_vals, theta=cats,
        fill="toself",
        fillcolor="rgba(65,105,225,0.15)",
        line=dict(color="royalblue", width=2, dash="dash"),
        name="成約圏 平均",
    ))
    fig.add_trace(go.Scatterpolar(
        r=cv_vals, theta=cats,
        fill="toself",
        fillcolor="rgba(255,215,0,0.25)",
        line=dict(color="goldenrod", width=2.5),
        name="現在案件",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        legend=dict(orientation="h", y=-0.15),
        height=450,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 改善ガイダンス ──
    gaps = centroid_norm - cv_norm
    needs_improvement = [
        (i, _IND_LABELS[i], gaps[i], _IND_RANGES[i])
        for i in range(len(_IND_LABELS))
        if gaps[i] > 0.05
    ]

    if not needs_improvement:
        st.success("🎉 すべての指標で成約圏と同等以上のレベルです！")
    else:
        st.markdown("**改善ガイダンス（成約ゾーンとのギャップ）**")
        for _, label, gap, (lo, hi) in sorted(needs_improvement, key=lambda x: -x[2]):
            improve_pct = gap * (hi - lo)
            pct_bar = int(gap * 100)
            st.markdown(
                f"- **{label}**: あと **{improve_pct:.1f}** 改善すると成約ゾーンに近づきます "
                f"{'🔴' if gap > 0.3 else '🟡' if gap > 0.15 else '🟢'}"
            )
            st.progress(max(0, min(100, int((1 - gap) * 100))))


# ── メイン UI ─────────────────────────────────────────────────────────────────

def render_clifford_visual() -> None:
    """CliffordNet 幾何代数ビジュアル分析ページ。スコアへの書き込みは行わない。"""
    st.title("🔷 幾何代数（CliffordNet）ビジュアル分析")
    st.caption(
        "財務指標をウェッジ積・内積で解析し、成約ゾーンを幾何学的に可視化します。"
        "**既存の審査スコアは変更しません（α=0）**。"
    )

    # 現在案件（読み取り専用）
    res = st.session_state.get("last_result")

    # DB 読み込み
    with st.spinner("データを読み込んでいます…"):
        df = _load_db()

    total_rows = _count_rows(_DB_PATH, "screening_records")
    if total_rows == 0:
        total_rows = _count_rows(_LEASE_DB_PATH, "past_cases")
    shown_rows = len(df)

    if shown_rows > 0:
        ind = _db_to_indicators(df)
        labels = _classify(df)
        n_approved = int(labels.sum())
        col1, col2, col3 = st.columns(3)
        col1.metric("総案件数", f"{total_rows} 件" if total_rows else f"{shown_rows} 件")
        col2.metric("表示件数", f"{shown_rows} 件")
        col3.metric("成約圏", f"{n_approved} 件")
        st.caption(f"図に使っているのは直近 {shown_rows} 件です。総案件数はDB全件数を表示しています。")
    else:
        ind = np.zeros((0, len(_IND_LABELS)))
        labels = np.zeros(0, dtype=int)
        st.info(
            "screening_records にデータがありません。「審査・分析」で審査を実行してから開いてください。"
        )

    if res:
        score = res.get("score", 0)
        hantei = res.get("hantei", "—")
        st.info(f"📋 現在案件: スコア **{score:.1f}点** / 判定 **{hantei}**（読み取り専用）")
    else:
        st.warning("現在案件がありません。「審査・分析」で審査を実行してから開いてください。")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🟡 成約の黄金面積マップ",
        "🏹 GGR 補正ベクトルアロー",
        "🌡️ 指標ペア整合性ヒートマップ",
        "🎯 成約クラスター近接度",
    ])

    with tab1:
        _tab_golden_area(df, ind, labels, res)

    with tab2:
        _tab_ggr_arrows(df, ind, labels, res)

    with tab3:
        _tab_heatmap(df, ind)

    with tab4:
        _tab_radar(df, ind, labels, res)
