"""
3D マハラノビス距離可視化コンポーネント。
過去案件データをPCAで3次元に圧縮し、信頼楕円体（D=1,2,3）と共にプロットする。
"""
import os
import sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.decomposition import PCA

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from mahalanobis_engine import MahalanobisScorer
from train_mahalanobis import FEATURES, _extract_val

MODEL_PATH = "data/mahalanobis_model.joblib"

FEAT_LABELS = {
    "nenshu":       "売上高",
    "op_profit":    "営業利益",
    "ord_profit":   "経常利益",
    "net_income":   "当期純利益",
    "net_assets":   "純資産",
    "total_assets": "総資産",
    "contracts":    "取引件数",
}

STATUS_COLOR = {
    "成約":  "#00d4ff",
    "失注":  "#ff4b4b",
    "否決":  "#ff4b4b",
    "未登録": "#aaaaaa",
}
STATUS_DEFAULT = "#aaaaaa"

ELLIPSOID_LEVELS = [
    (1.0, "rgba(0,255,100,0.10)", "rgba(0,255,100,0.60)", "D=1"),
    (2.0, "rgba(255,200,0,0.07)", "rgba(255,200,0,0.55)", "D=2"),
    (3.0, "rgba(255,80,80,0.05)",  "rgba(255,80,80,0.50)",  "D=3"),
]


@st.cache_resource
def _load_model() -> MahalanobisScorer | None:
    if os.path.exists(MODEL_PATH):
        return MahalanobisScorer.load(MODEL_PATH)
    return None


@st.cache_data(ttl=300)
def _load_cases() -> list[dict]:
    try:
        from data_cases import load_all_cases
        return load_all_cases()
    except Exception:
        return []


def _build_feature_matrix(cases: list[dict]) -> tuple[np.ndarray, list[str]]:
    """過去案件から特徴量行列と最終ステータスリストを返す。"""
    rows, statuses = [], []
    for c in cases:
        row = [_extract_val(c, f) for f in FEATURES]
        rows.append(row)
        statuses.append(c.get("final_status", "未登録") or "未登録")
    return np.array(rows, dtype=float), statuses


def _make_ellipsoid_traces(
    cov_pca: np.ndarray,
    mu_pca: np.ndarray,
    resolution: int = 24,
) -> list[go.Mesh3d]:
    """PCA空間における信頼楕円体トレースを生成する（D=1,2,3）。"""
    # 楕円体の軸方向と長さを固有値分解で求める
    eigvals, eigvecs = np.linalg.eigh(cov_pca)
    eigvals = np.maximum(eigvals, 1e-12)
    L = eigvecs * np.sqrt(eigvals)  # 3x3: 各列が軸方向×長さ

    # 球面上の点を生成
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    uu, vv = np.meshgrid(u, v)
    sphere = np.stack([
        np.sin(vv) * np.cos(uu),
        np.sin(vv) * np.sin(uu),
        np.cos(vv),
    ], axis=-1)  # (res, res, 3)

    traces = []
    for d_val, color_face, color_line, label in ELLIPSOID_LEVELS:
        pts = (sphere * d_val) @ L.T + mu_pca  # (res, res, 3)
        x = pts[:, :, 0].flatten()
        y = pts[:, :, 1].flatten()
        z = pts[:, :, 2].flatten()
        traces.append(go.Mesh3d(
            x=x, y=y, z=z,
            alphahull=0,
            color=color_face,
            opacity=0.15,
            name=label,
            showlegend=True,
            hoverinfo="skip",
            contour=dict(show=True, color=color_line, width=2),
        ))
    return traces


def _get_current_case_vec() -> np.ndarray | None:
    """セッション状態から現在審査中の案件特徴量ベクトルを取得する。"""
    ss = st.session_state

    # 審査フォームの直近入力値（lease_logic_sumaho12.py の session_state キー）
    inputs = ss.get("form_inputs") or ss.get("_last_inputs") or {}
    if inputs:
        row = [float(inputs.get(f, 0) or 0) for f in FEATURES]
        if any(v != 0 for v in row):
            return np.array(row, dtype=float)

    # mahalanobis_ui.py のスライダー値
    keys_present = [f for f in FEATURES if f in ss]
    if keys_present:
        row = [float(ss.get(f, 0) or 0) for f in FEATURES]
        if any(v != 0 for v in row):
            return np.array(row, dtype=float)

    return None


def _retrain() -> tuple[bool, str]:
    """train_mahalanobis.py を実行してモデルを再学習する。(成否, ログ) を返す。"""
    import subprocess
    result = subprocess.run(
        [sys.executable, "train_mahalanobis.py"],
        capture_output=True, text=True,
        cwd=_DIR,
    )
    _load_model.clear()
    ok = result.returncode == 0
    log = (result.stdout + result.stderr).strip()
    return ok, log


def render_mahalanobis_3d() -> None:
    st.title("📊 マハラノビス距離 3D 可視化")
    st.caption(
        "過去案件をPCA圧縮した3次元空間で表示。信頼楕円体（D=1/2/3）が合格圏の目安を示す。"
        "現在審査中の案件がある場合は★で強調表示される。"
    )

    with st.sidebar:
        st.divider()
        st.markdown("### 🛡️ マハラノビス3D")
        if st.button("🔄 モデル再学習", use_container_width=True, key="mah3d_retrain"):
            with st.spinner("学習中…"):
                ok, log = _retrain()
            if ok:
                st.success("学習完了")
            else:
                st.error(f"失敗: {log[-300:]}")
            st.rerun()

    model = _load_model()
    if model is None:
        st.error("モデルが未生成です。下のボタンで学習してください。")
        if st.button("🔄 モデルを今すぐ学習する", type="primary"):
            import subprocess
            with st.spinner("学習中… (数秒かかります)"):
                result = subprocess.run(
                    [sys.executable, "train_mahalanobis.py"],
                    capture_output=True, text=True,
                    cwd=_DIR,
                )
            _load_model.clear()
            if result.returncode == 0:
                st.success("学習完了！ページを再読み込みします。")
                st.rerun()
            else:
                st.error(f"学習失敗:\n{result.stderr[-500:]}")
        return

    cases = _load_cases()

    # ── 特徴量行列の構築 ──────────────────────────────────────────────────────
    if len(cases) < 3:
        st.warning("過去案件が3件未満のため合成データを使用します。")
        from train_mahalanobis import _synth_df
        df_synth = _synth_df(80)
        X_raw = df_synth[FEATURES].values
        statuses = ["合成データ"] * len(X_raw)
    else:
        X_raw, statuses = _build_feature_matrix(cases)

    # StandardScaler で変換 → モデルのスケーラーを流用
    X_scaled = model.scaler.transform(
        model._preprocess(X_raw)
    )

    # ── PCA 3次元 ─────────────────────────────────────────────────────────────
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    mu_pca = pca.transform(model.mu_.reshape(1, -1)).flatten()

    # PCA空間での共分散行列（マハラノビス距離の精度行列から逆算）
    S_scaled = np.linalg.inv(model.precision_)            # 元空間の共分散
    W = pca.components_                                     # (3, n_feat)
    cov_pca = W @ S_scaled @ W.T                           # (3, 3)

    # ── 現在案件 ──────────────────────────────────────────────────────────────
    cur_vec = _get_current_case_vec()
    cur_pca = None
    cur_score = None
    cur_d = None
    if cur_vec is not None:
        cur_scaled = model.scaler.transform(model._preprocess(cur_vec))
        cur_pca = pca.transform(cur_scaled).flatten()
        cur_score, cur_d, _, _ = model.get_analysis(cur_vec)

    # ── Plotly トレース組み立て ───────────────────────────────────────────────
    fig = go.Figure()

    # 楕円体
    for trace in _make_ellipsoid_traces(cov_pca, mu_pca):
        fig.add_trace(trace)

    # 過去案件（ステータス別）
    unique_statuses = list(dict.fromkeys(statuses))
    for s in unique_statuses:
        idx = [i for i, st in enumerate(statuses) if st == s]
        color = STATUS_COLOR.get(s, STATUS_DEFAULT)
        case_names = []
        for i in idx:
            c = cases[i] if i < len(cases) else {}
            name = c.get("company_name") or c.get("inputs", {}).get("company_name") or f"案件{i+1}"
            _, d_i, _, _ = model.get_analysis(X_raw[i])
            case_names.append(f"{name}<br>D={d_i:.2f}")
        fig.add_trace(go.Scatter3d(
            x=X_pca[idx, 0],
            y=X_pca[idx, 1],
            z=X_pca[idx, 2],
            mode="markers",
            marker=dict(size=5, color=color, opacity=0.75, line=dict(width=0.3, color="white")),
            name=s,
            text=case_names,
            hovertemplate="%{text}<extra></extra>",
        ))

    # 重心
    fig.add_trace(go.Scatter3d(
        x=[mu_pca[0]], y=[mu_pca[1]], z=[mu_pca[2]],
        mode="markers",
        marker=dict(symbol="diamond", size=10, color="gold", line=dict(width=1, color="white")),
        name="重心 ◆",
        hovertemplate="重心（合格典型）<extra></extra>",
    ))

    # 現在審査中の案件
    if cur_pca is not None:
        fig.add_trace(go.Scatter3d(
            x=[cur_pca[0]], y=[cur_pca[1]], z=[cur_pca[2]],
            mode="markers+text",
            marker=dict(symbol="cross", size=14, color="#ff00ff",
                        line=dict(width=2, color="white")),
            text=["★ 現在案件"],
            textposition="top center",
            name=f"★ 現在案件 (D={cur_d:.2f})" if cur_d is not None else "★ 現在案件",
            hovertemplate=(
                f"★ 現在審査中<br>D={cur_d:.3f}<br>スコア={cur_score:.1f}pt<extra></extra>"
                if cur_d is not None else "★ 現在審査中<extra></extra>"
            ),
        ))

    # 軸ラベル
    axis_labels = [
        f"PC1 ({explained[0]*100:.1f}%)",
        f"PC2 ({explained[1]*100:.1f}%)",
        f"PC3 ({explained[2]*100:.1f}%)",
    ]
    fig.update_layout(
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            bgcolor="rgba(10,10,20,1)",
            xaxis=dict(gridcolor="#333", zerolinecolor="#555"),
            yaxis=dict(gridcolor="#333", zerolinecolor="#555"),
            zaxis=dict(gridcolor="#333", zerolinecolor="#555"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.05, font=dict(size=11)),
        margin=dict(l=0, r=0, t=30, b=0),
        height=620,
        title=dict(
            text="マハラノビス距離 3D — 信頼楕円体（D=1/2/3）",
            font=dict(size=15),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── 補足情報 ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**PCA 寄与率**")
        for i, r in enumerate(explained):
            st.caption(f"PC{i+1}: {r*100:.1f}%")
        st.caption(f"累積: {explained.sum()*100:.1f}%")

    with col2:
        st.markdown("**PCA 主成分の解釈（上位寄与特徴量）**")
        for i, comp in enumerate(pca.components_):
            top_idx = np.argsort(np.abs(comp))[::-1][:3]
            parts = ", ".join(
                f"{FEAT_LABELS.get(FEATURES[j], FEATURES[j])}({comp[j]:+.2f})"
                for j in top_idx
            )
            st.caption(f"PC{i+1}: {parts}")

    if cur_d is not None:
        st.divider()
        status_label = "✅ 合格圏内 (D<2)" if cur_d < 2 else ("⚠️ 要注意 (D<3)" if cur_d < 3 else "❌ 外れ値 (D≥3)")
        st.metric("現在案件のマハラノビス距離", f"{cur_d:.3f}", delta=status_label)
    else:
        st.info("現在審査中の案件がありません。審査・分析画面で審査を実行すると★がプロットされます。")
