import subprocess
import sys
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from mahalanobis_engine import MahalanobisScorer

st.set_page_config(layout="wide", page_title="マハラノビス距離審査", page_icon="🛡️")

MODEL_PATH = "data/mahalanobis_model.joblib"

FEAT_LABELS = {
    "nenshu": "売上高 (千円)",
    "op_profit": "営業利益 (千円)",
    "ord_profit": "経常利益 (千円)",
    "net_income": "当期純利益 (千円)",
    "net_assets": "純資産 (千円)",
    "total_assets": "総資産 (千円)",
    "contracts": "取引件数",
}
LOG_FEATS = {"nenshu", "net_assets", "total_assets"}
DEFAULTS = {
    "nenshu": 500_000, "op_profit": 20_000, "ord_profit": 18_000,
    "net_income": 10_000, "net_assets": 150_000, "total_assets": 600_000, "contracts": 3,
}
SLIDER_CFG = {
    "nenshu":       (0, 5_000_000, 10_000),
    "op_profit":    (-500_000, 500_000, 1_000),
    "ord_profit":   (-500_000, 500_000, 1_000),
    "net_income":   (-500_000, 500_000, 1_000),
    "net_assets":   (-500_000, 2_000_000, 10_000),
    "total_assets": (0, 5_000_000, 10_000),
    "contracts":    (0, 100, 1),
}


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return MahalanobisScorer.load(MODEL_PATH)
    return None


def retrain():
    result = subprocess.run(
        [sys.executable, "train_mahalanobis.py"],
        capture_output=True, text=True
    )
    st.cache_resource.clear()
    return result.stdout + result.stderr


# --- サイドバー ---
with st.sidebar:
    st.header("🛡️ マハラノビス距離審査")
    if st.button("🔄 モデル再学習", use_container_width=True):
        with st.spinner("学習中..."):
            log = retrain()
        st.text(log)

    model = load_model()
    if model:
        st.info(f"学習件数: {model.train_size} 件\n\n更新: {model.last_updated}")
    else:
        st.error("モデル未生成。再学習ボタンを押してください。")

    st.divider()
    st.subheader("入力値")
    inputs = {}
    for feat in (model.feature_names if model else list(FEAT_LABELS.keys())):
        lo, hi, step = SLIDER_CFG.get(feat, (0, 1_000_000, 1_000))
        default = DEFAULTS.get(feat, 0)
        if feat == "contracts":
            inputs[feat] = float(st.slider(FEAT_LABELS.get(feat, feat), lo, hi, default, step))
        else:
            inputs[feat] = float(st.slider(FEAT_LABELS.get(feat, feat), lo, hi, default, step))

if model is None:
    st.stop()

x_vec = [inputs[f] for f in model.feature_names]
score, d, grad, contribs = model.get_analysis(x_vec)
advices = model.advise_improvement(x_vec, top_k=3)

st.title("🛡️ マハラノビス距離審査コックピット")

col_gauge, col_bar, col_advice = st.columns([1, 1, 1])

# --- ゲージ ---
with col_gauge:
    color = "#00ff00" if score >= 70 else ("#ffa500" if score >= 40 else "#ff4b4b")
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " pt", "font": {"size": 40}},
        title={"text": f"幾何学的信頼スコア<br>D={d:.3f}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#2d0000"},
                {"range": [40, 70], "color": "#2d1800"},
                {"range": [70, 100], "color": "#002d00"},
            ],
        },
    ))
    fig_g.update_layout(height=300, margin=dict(t=60, b=0, l=20, r=20),
                        paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_g, use_container_width=True)

# --- 貢献度バー ---
with col_bar:
    st.subheader("属性別寄与度")
    labels = [FEAT_LABELS.get(f, f) for f in model.feature_names]
    vals = [float(c) * 100 for c in contribs]
    colors = ["#00d4ff" if v >= 0 else "#ff4b4b" for v in vals]
    fig_b = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
    ))
    fig_b.update_layout(height=300, margin=dict(t=20, b=0, l=0, r=60),
                        xaxis_title="寄与%", paper_bgcolor="rgba(0,0,0,0)",
                        yaxis={"autorange": "reversed"})
    st.plotly_chart(fig_b, use_container_width=True)

# --- 改善アドバイス ---
with col_advice:
    st.subheader("測地線アドバイス（最短改善）")
    for a in advices:
        label = FEAT_LABELS.get(a["feat"], a["feat"])
        delta = a["delta"]
        sign = "+" if delta > 0 else ""
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.markdown(f"{a['direction']} **{sign}{delta:,.0f}千円**")
            st.caption(f"現在: {a['current']:,.0f} → 推奨: {a['recommended']:,.0f}")

st.divider()

# --- 等高線マップ ---
st.subheader("2D 等高線マップ（合格圏可視化）")
feat_options = [FEAT_LABELS.get(f, f) for f in model.feature_names]
feat_map = {FEAT_LABELS.get(f, f): f for f in model.feature_names}

c1, c2 = st.columns(2)
with c1:
    sel_x = st.selectbox("X軸", feat_options, index=0)
with c2:
    remaining = [f for f in feat_options if f != sel_x]
    sel_y = st.selectbox("Y軸", remaining, index=min(4, len(remaining)-1))

fx = feat_map[sel_x]
fy = feat_map[sel_y]
ix = model.feature_names.index(fx)
iy = model.feature_names.index(fy)

x_cur, y_cur = x_vec[ix], x_vec[iy]
x_range = np.linspace(max(0, x_cur * 0.1), x_cur * 3 + 1, 60)
y_range = np.linspace(y_cur * 0.1 if y_cur > 0 else -abs(y_cur)*3, y_cur * 3 + 1, 60)

Z = np.zeros((len(y_range), len(x_range)))
for i, yv in enumerate(y_range):
    for j, xv in enumerate(x_range):
        pt = list(x_vec)
        pt[ix] = xv
        pt[iy] = yv
        s, _, _, _ = model.get_analysis(pt)
        Z[i, j] = s

mu_orig = model.scaler.inverse_transform(model.mu_.reshape(1, -1)).flatten()

fig_c = go.Figure()
fig_c.add_trace(go.Contour(
    x=x_range, y=y_range, z=Z,
    colorscale="RdYlGn", zmin=0, zmax=100,
    contours=dict(showlabels=True, labelfont=dict(size=10)),
    colorbar=dict(title="スコア"),
))
fig_c.add_trace(go.Scatter(
    x=[x_cur], y=[y_cur], mode="markers",
    marker=dict(symbol="circle", size=14, color="red", line=dict(width=2, color="white")),
    name="現在位置",
))
fig_c.add_trace(go.Scatter(
    x=[np.expm1(mu_orig[ix]) if fx in model.log_features else mu_orig[ix]],
    y=[np.expm1(mu_orig[iy]) if fy in model.log_features else mu_orig[iy]],
    mode="markers",
    marker=dict(symbol="star", size=18, color="gold"),
    name="重心 ★",
))
fig_c.update_layout(
    xaxis_title=sel_x, yaxis_title=sel_y,
    height=450, paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.05),
)
st.plotly_chart(fig_c, use_container_width=True)
