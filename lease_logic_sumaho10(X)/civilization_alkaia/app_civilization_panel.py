"""
アルカイアの慟哭 — 文明パラメータ・リアルタイム・シミュレーター

選択肢ゲームではなく、全パラメータをスライダーで操作し、
リアルタイムで審判・リタ・文明の形を表示。1億年シミュレーションで最終ログを生成。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback

import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# ダークモード・ハッカー風フォント・危険域フラッシュ
# ---------------------------------------------------------------------------
DARK_CSS = """
<style>
  /* ダーク背景・モノスペース */
  .stApp, [data-testid="stAppViewContainer"], main { background: #0d1117 !important; }
  .stApp * { font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Mono', monospace !important; }
  /* サイドバー */
  section[data-testid="stSidebar"] { background: #161b22 !important; }
  section[data-testid="stSidebar"] * { font-family: 'SF Mono', 'Monaco', monospace !important; }
  /* メインエリアのテキスト色 */
  .stMarkdown, p, span, label, .stAlert { color: #c9d1d9 !important; }
  h1, h2, h3 { color: #58a6ff !important; }
  /* 危険域フラッシュ */
  .danger-flash { animation: flash 1.5s ease-in-out infinite; }
  @keyframes flash {
    0%, 100% { background: #0d1117; }
    50% { background: #3d1f1f; box-shadow: 0 0 40px rgba(248,81,73,0.4); }
  }
  /* アルカイア・リタの表示枠 */
  .arcaia-box { background: #21262d; border-left: 4px solid #58a6ff; padding: 1rem; margin: 0.5rem 0; border-radius: 4px; }
  .rita-box { background: #21262d; border-left: 4px solid #7ee787; padding: 1rem; margin: 0.5rem 0; border-radius: 4px; font-size: 1.4rem; }
  /* レーダーコンテナ */
  .radar-container { background: #161b22; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
</style>
"""

# civilization(alkaia).py をロード
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.dirname(_SCRIPT_DIR)
_CIV_NAME = "civilization(alkaia).py"
_CIV_PATH = os.path.join(_SCRIPT_DIR, _CIV_NAME)
if not os.path.isfile(_CIV_PATH):
    _CIV_PATH = os.path.join(os.getcwd(), _CIV_NAME)
if not os.path.isfile(_CIV_PATH):
    _CIV_PATH = os.path.join(os.getcwd(), "civilization_alkaia", _CIV_NAME)

try:
    if not os.path.isfile(_CIV_PATH):
        raise FileNotFoundError(f"ゲームモジュールが見つかりません: {_SCRIPT_DIR}, {os.getcwd()}")
    spec = importlib.util.spec_from_file_location("civilization_alkaia", _CIV_PATH)
    _civ = importlib.util.module_from_spec(spec)
    sys.modules["civilization_alkaia"] = _civ
    spec.loader.exec_module(_civ)
    infer_decay_reasons = _civ.infer_decay_reasons
    JUDGMENT_THRESHOLDS = _civ.JUDGMENT_THRESHOLDS
except Exception:
    st.error("ゲームモジュールの読み込みに失敗しました。")
    st.code(traceback.format_exc())
    st.stop()

HABITABILITY_PATH = os.path.join(_DATA_DIR, "solar_system_habitability.json")
YEARS_1E8 = 100_000_000  # 1億年


def _load_habitability() -> list:
    if not os.path.isfile(HABITABILITY_PATH):
        return []
    with open(HABITABILITY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("events") or []


def _habitability_at_years(events: list, years: int) -> dict | None:
    """指定年数後に最も近いタイムラインイベントを返す。"""
    if not events:
        return None
    best = None
    best_diff = float("inf")
    for ev in events:
        y = ev.get("years_from_now", 0)
        if y <= years and (years - y) < best_diff:
            best_diff = years - y
            best = ev
    if best is not None:
        return best
    return events[0]


def _is_danger(total_score: float, happiness: float, violence: float, sustainability: float) -> bool:
    return (
        total_score < JUDGMENT_THRESHOLDS["monitor"]
        or happiness < 25
        or violence > 70
        or sustainability < 30
    )


def _rita_expression(happiness: float, violence: float) -> tuple[str, str]:
    """(テキスト, 顔文字) を返す。"""
    if violence > 70:
        return "暴力の指数が、リタを怯えさせている。", "😨"
    if happiness <= 0:
        return "リタは、もう動かない。記録を閉じる。", "💀"
    if happiness >= 80 and violence < 30:
        return "リタは穏やかに眠っている。君の文明を、彼女は信じている。", "🐕✨"
    if happiness >= 60:
        return "リタは落ち着いている。まだ、見届ける理由がある。", "🐕"
    if happiness >= 40:
        return "リタがこちらを見ている。何かを待っている。", "🐕"
    if happiness >= 20:
        return "リタが震えている。この数値の先に、彼女は何を見ている？", "😢"
    return "リタは悲しんでいる。君のパラメータが、彼女を傷つけた。", "😭"


def _radar_chart_7(
    tech: float, ethical: float, sustainability: float,
    happiness: float, env: float, violence: float, resource: float
) -> go.Figure:
    """5〜7軸のレーダーチャート。技術だけ突出すると歪になる。"""
    categories = ["技術", "倫理", "持続", "幸福", "環境負荷(逆)", "暴力(逆)", "資源(逆)"]
    # 環境・暴力・資源は「低いほど良い」なので 100-x で表示（高い＝危険を「突出」で示す）
    env_inv = 100 - env
    violence_inv = 100 - violence
    resource_inv = 100 - resource
    values = [tech, ethical, sustainability, happiness, env_inv, violence_inv, resource_inv]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="文明の形",
            line=dict(color="rgba(88,166,255,0.95)", width=2),
            fillcolor="rgba(88,166,255,0.35)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        paper_bgcolor="rgba(13,17,23,0)",
        plot_bgcolor="rgba(22,27,34,0.5)",
        font=dict(color="#c9d1d9", family="monospace"),
        title=dict(text="文明の形状（技術だけ突出＝歪な文明）", font=dict(size=14)),
        showlegend=False,
        height=420,
        margin=dict(t=40),
    )
    return fig


def _run_1e8_simulation(
    tech: float, ethical: float, sustainability: float,
    env: float, violence: float, resource: float, happiness: float,
    env_impact: float, violence_idx: float, inequality: float,
    space: float, tech_prog: float, ai_dev: float, energy: float,
) -> str:
    """1億年放置後のアルカイアの最終ログを生成。"""
    events = _load_habitability()
    ev = _habitability_at_years(events, YEARS_1E8)
    total_score = tech * 0.30 + ethical * 0.35 + sustainability * 0.35
    reasons = infer_decay_reasons(
        tech, ethical, sustainability,
        environmental_impact=env_impact, violence_index=violence_idx, inequality=inequality,
        space_exploration=space, tech_progress=tech_prog,
        ai_development=ai_dev, energy_utilization=energy,
    )
    earth_status = "不明"
    if ev:
        earth_status = (ev.get("habitability") or {}).get("地球", "不明")
        if ev.get("unhabitable_note"):
            earth_status += " — " + ev["unhabitable_note"]

    lines = [
        f"[ 経過: +{YEARS_1E8 // 10_000_000}千万年 ]",
        f"[ 地球の状態: {earth_status} ]",
        "",
    ]
    if total_score >= JUDGMENT_THRESHOLDS["preserve"]:
        lines.append("アルカイア: 「プロトコル適合。この文明は、記録に残す価値がある。」")
    elif total_score >= JUDGMENT_THRESHOLDS["monitor"]:
        lines.append("アルカイア: 「要観測。1億年後も存続しているが、次の1億年は保証しない。」")
    else:
        lines.append("アルカイア: 「排除・再生失敗。このパラメータでは、文明は既に滅んでいる。」")
    if reasons:
        lines.append("")
        lines.append("滅びの理由（推定）: " + " / ".join(reasons))
    if happiness <= 0:
        lines.append("")
        lines.append("リタは、この文明を認めなかった。")
    lines.append("")
    lines.append("—— 記録を閉じる ——")
    return "\n".join(lines)


# ---------- ページ設定・ダークモード ----------
st.set_page_config(page_title="アルカイア — 文明シミュレーター", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)
st.title("アルカイアの慟哭 — 文明パラメータ・リアルタイム・シミュレーター")

# ---------- 1. サイドバー: 全変数スライダー ----------
st.sidebar.header("文明パラメータ（すべてリアルタイム反映）")

tech_level = st.sidebar.slider("技術レベル (tech_level)", 0.0, 100.0, 50.0, 0.5)
ethical_dev = st.sidebar.slider("倫理発達 (ethical_dev)", 0.0, 100.0, 50.0, 0.5)
sustainability = st.sidebar.slider("持続可能性 (sustainability)", 0.0, 100.0, 50.0, 0.5)
st.sidebar.markdown("---")
environmental_impact = st.sidebar.slider("環境負荷 (environmental_impact)", 0.0, 100.0, 50.0, 0.5)
violence_index = st.sidebar.slider("暴力指数 (violence_index)", 0.0, 100.0, 50.0, 0.5)
resource_consumption = st.sidebar.slider("資源消費 (resource_consumption)", 0.0, 100.0, 50.0, 0.5)
happiness_index = st.sidebar.slider("幸福度 (happiness_index)", 0.0, 100.0, 50.0, 0.5)

# 1億年シミュレーション用の補助パラメータ（滅び理由に使用）
st.sidebar.markdown("---")
st.sidebar.caption("滅び推定の補助指標")
inequality = st.sidebar.slider("不平等度 (inequality)", 0.0, 100.0, 50.0, 0.5)
space_exploration = st.sidebar.slider("宇宙進出 (space_exploration)", 0.0, 100.0, 50.0, 0.5)
tech_progress = st.sidebar.slider("技術進歩 (tech_progress)", 0.0, 100.0, 50.0, 0.5)
ai_development = st.sidebar.slider("AI発達 (ai_development)", 0.0, 100.0, 50.0, 0.5)
energy_utilization = st.sidebar.slider("エネルギー利用 (energy_utilization)", 0.0, 100.0, 50.0, 0.5)

total_score = tech_level * 0.30 + ethical_dev * 0.35 + sustainability * 0.35
danger = _is_danger(total_score, happiness_index, violence_index, sustainability)

# ---------- 2. リアルタイム・フィードバック（メインパネル） ----------
if danger:
    st.markdown('<div class="danger-flash">', unsafe_allow_html=True)
    st.error("⚠ 危険域: スコア・幸福度・持続可能性の低下、または暴力指数の上昇。アルカイアの排除が近い。")
    st.markdown('</div>', unsafe_allow_html=True)

# 【アルカイアの審判】
reasons = infer_decay_reasons(
    tech_level, ethical_dev, sustainability,
    environmental_impact=environmental_impact, violence_index=violence_index, inequality=inequality,
    space_exploration=space_exploration, tech_progress=tech_progress,
    ai_development=ai_development, energy_utilization=energy_utilization,
)
st.markdown("### 【アルカイアの審判】")
judge_text = "この瞬間のパラメータで想定される滅びの理由: **" + " / ".join(reasons) + "**" if reasons else "特定の滅び理由は検出されていません（「その他」のみ）。"
st.markdown(f'<div class="arcaia-box">{judge_text}</div>', unsafe_allow_html=True)

# 【リタの表情】
rita_text, rita_emoji = _rita_expression(happiness_index, violence_index)
st.markdown("### 【リタの表情】")
st.markdown(f'<div class="rita-box">{rita_emoji} {rita_text}</div>', unsafe_allow_html=True)

# 【文明の形状】レーダーチャート（5〜7パラメータ）
st.markdown("### 【文明の形状】")
fig = _radar_chart_7(
    tech_level, ethical_dev, sustainability,
    happiness_index, environmental_impact, violence_index, resource_consumption,
)
st.plotly_chart(fig, use_container_width=True)
st.caption("技術だけが突出した歪な形＝排除に近い文明。環境・暴力・資源は「低いほど良い」で逆表示。")

# ---------- 3. 1億年後のシミュレーション ----------
st.markdown("---")
st.markdown("### 1億年後のシミュレーション")
if st.button("このパラメータで1億年放置する"):
    log = _run_1e8_simulation(
        tech_level, ethical_dev, sustainability,
        environmental_impact, violence_index, resource_consumption, happiness_index,
        environmental_impact, violence_index, inequality,
        space_exploration, tech_progress, ai_development, energy_utilization,
    )
    st.session_state["sim_1e8_log"] = log
if st.session_state.get("sim_1e8_log"):
    st.markdown("**アルカイアの最終ログ**")
    st.code(st.session_state["sim_1e8_log"], language=None)
