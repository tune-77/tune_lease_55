# -*- coding: utf-8 -*-
"""
novel_simulation_view.py
========================
文明年代記 — アルカイアの記録
太陽系を舞台にした文明シミュレーション ビジュアライゼーション。
MiroFish スタイルの D3.js グラフ（星空背景・グロー・アニメーション）を使用。
"""
from __future__ import annotations

import json
import time
import math
import streamlit as st
import streamlit.components.v1 as components

# ─── 太陽系 D3.js テンプレート ────────────────────────────────────────────────
_SOLAR_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0d1117; overflow:hidden; font-family:'SF Mono','Monaco','Inconsolata',monospace; }
  canvas#stars { position:absolute; top:0; left:0; pointer-events:none; }
  svg#solar { position:absolute; top:0; left:0; }

  @keyframes pulse-ring {
    0%,100% { transform:scale(1);    opacity:0.8; }
    50%     { transform:scale(1.15); opacity:0.3; }
  }
  @keyframes beacon-pulse {
    0%,100% { opacity:0.75; }
    50%     { opacity:1.0; }
  }
  @keyframes beacon-collapse {
    from { opacity:1; }
    to   { opacity:0; transform:scaleY(0); }
  }
  @keyframes beacon-ascend {
    0%   { opacity:1;   transform:scaleY(1)   translateY(0); }
    60%  { opacity:0.8; transform:scaleY(2.5) translateY(-80px); }
    100% { opacity:0;   transform:scaleY(3)   translateY(-200px); }
  }
  @keyframes flow-dash {
    from { stroke-dashoffset:20; }
    to   { stroke-dashoffset:0; }
  }
  @keyframes arc-war {
    from { stroke-dashoffset:40; }
    to   { stroke-dashoffset:0; }
  }
  .pulse       { animation:pulse-ring 2.4s ease-in-out infinite; transform-origin:center; }
  .b-active    { animation:beacon-pulse 2.2s ease-in-out infinite; }
  .b-collapse  { animation:beacon-collapse 3s ease-in forwards; }
  .b-ascend    { animation:beacon-ascend 2.5s ease-out forwards; }
  .arc-war     { animation:arc-war 0.7s linear infinite; }
  .arc-ally    { animation:flow-dash 1.5s linear infinite; }

  .tooltip {
    position:absolute; pointer-events:none;
    background:rgba(13,17,23,0.92); border:1px solid #334155;
    color:#c9d1d9; padding:8px 12px; border-radius:6px;
    font-size:11px; line-height:1.6; max-width:220px;
    display:none; z-index:999;
  }
  .badge {
    position:absolute; bottom:12px; left:14px;
    background:rgba(13,17,23,0.85); border:1px solid #1e293b;
    color:#64748b; padding:6px 10px; border-radius:6px;
    font-size:10px; line-height:1.8;
  }
  .badge span { color:#c9d1d9; }
</style>
</head>
<body>
<canvas id="stars"></canvas>
<svg id="solar"></svg>
<div class="tooltip" id="tip"></div>
<div class="badge" id="badge"></div>

<script>
const DATA   = __SOLAR_DATA__;
const W      = window.innerWidth;
const H      = __HEIGHT__;
const CX     = W / 2;
const CY     = H / 2;
const SUN_R  = 36;
const MAX_BEACON_H = Math.min(H * 0.38, 220);
const HZ_R_INNER   = Math.min(W, H) * 0.22;
const HZ_R_OUTER   = Math.min(W, H) * 0.32;

// ── 星空 Canvas ──────────────────────────────────────────────────────
const canvas = document.getElementById("stars");
canvas.width = W; canvas.height = H;
const ctx = canvas.getContext("2d");
for (let i = 0; i < 200; i++) {
  const x = Math.random() * W;
  const y = Math.random() * H;
  const r = Math.random() * 1.1 + 0.2;
  const a = Math.random() * 0.55 + 0.1;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = `rgba(200,220,255,${a})`;
  ctx.fill();
}

// ── SVG 初期化 ────────────────────────────────────────────────────────
const svg = document.getElementById("solar");
svg.setAttribute("width", W);
svg.setAttribute("height", H);

function el(tag, attrs = {}) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
  return e;
}
function appendTo(parent, child) { parent.appendChild(child); return child; }

// ── defs: グロー・グラデーション ───────────────────────────────────
const defs = appendTo(svg, el("defs"));

// 太陽グロー
const glowF = appendTo(defs, el("filter", {id:"glow-sun", x:"-50%", y:"-50%", width:"200%", height:"200%"}));
appendTo(glowF, el("feGaussianBlur", {stdDeviation:"8", result:"blur"}));
const glowMerge = appendTo(glowF, el("feMerge"));
appendTo(glowMerge, el("feMergeNode", {in:"blur"}));
appendTo(glowMerge, el("feMergeNode", {in:"SourceGraphic"}));

// ビーコングロー
const glowB = appendTo(defs, el("filter", {id:"glow-beacon", x:"-100%", y:"-50%", width:"300%", height:"200%"}));
appendTo(glowB, el("feGaussianBlur", {stdDeviation:"4", result:"blur"}));
const glowBMerge = appendTo(glowB, el("feMerge"));
appendTo(glowBMerge, el("feMergeNode", {in:"blur"}));
appendTo(glowBMerge, el("feMergeNode", {in:"SourceGraphic"}));

// 太陽 radial gradient
const sunGrad = appendTo(defs, el("radialGradient", {id:"sun-grad", cx:"50%", cy:"50%", r:"50%"}));
[["0%","#fff9c4",1],["35%","#fbbf24",1],["70%","#f97316",0.7],["100%","#92400e",0]].forEach(([o,c,a])=>{
  const s = appendTo(sunGrad, el("stop", {offset:o}));
  s.setAttribute("stop-color", c);
  s.setAttribute("stop-opacity", a);
});

// ── ハビタブルゾーン ─────────────────────────────────────────────────
[HZ_R_INNER, HZ_R_OUTER].forEach((r, i) => {
  appendTo(svg, el("circle", {
    cx: CX, cy: CY, r,
    fill: "none",
    stroke: "#22c55e",
    "stroke-width": 0.8,
    "stroke-dasharray": "4,6",
    opacity: i === 0 ? 0.3 : 0.15
  }));
});
// HZ 帯塗り
const hzBand = appendTo(svg, el("path", {}));
hzBand.setAttribute("d", `
  M ${CX - HZ_R_OUTER} ${CY}
  A ${HZ_R_OUTER} ${HZ_R_OUTER} 0 1 1 ${CX + HZ_R_OUTER} ${CY}
  A ${HZ_R_OUTER} ${HZ_R_OUTER} 0 1 1 ${CX - HZ_R_OUTER} ${CY}
  M ${CX - HZ_R_INNER} ${CY}
  A ${HZ_R_INNER} ${HZ_R_INNER} 0 1 0 ${CX + HZ_R_INNER} ${CY}
  A ${HZ_R_INNER} ${HZ_R_INNER} 0 1 0 ${CX - HZ_R_INNER} ${CY}
`);
hzBand.setAttribute("fill", "rgba(34,197,94,0.04)");
hzBand.setAttribute("fill-rule", "evenodd");

// ── 太陽本体 ────────────────────────────────────────────────────────
const sunCircle = appendTo(svg, el("circle", {
  cx: CX, cy: CY, r: SUN_R,
  fill: "url(#sun-grad)",
  filter: "url(#glow-sun)",
  class: "pulse"
}));

// 太陽フェーズラベル
const solPhase = DATA.solar ? DATA.solar.phase : "主系列";
const phaseText = appendTo(svg, el("text", {
  x: CX, y: CY + SUN_R + 14,
  "text-anchor": "middle",
  fill: "#fbbf24",
  "font-size": "10",
  opacity: "0.7"
}));
phaseText.textContent = solPhase;

// ── 文明ビーコン ─────────────────────────────────────────────────────
const civs = DATA.civilizations || [];
const n = civs.length;
const baseRadius = Math.min(W, H) * 0.13;  // ビーコン根元の太陽からの距離

// イベントマップ (source → event_type)
const eventMap = {};
(DATA.events || []).forEach(ev => {
  if (ev.source) eventMap[ev.source] = ev.event_type;
  if (ev.civ)    eventMap[ev.civ]    = ev.event_type;
});

// ビーコン線形グラデーション (per-civ)
const beaconPositions = [];  // [{cx, cy, tipX, tipY, civ}] for arc drawing

civs.forEach((civ, i) => {
  const angleDeg = civ.angle_deg !== undefined ? civ.angle_deg : (i * 360 / n);
  const angleRad = (angleDeg - 90) * Math.PI / 180;
  const beaconH  = Math.max(20, (civ.tech_level || 0.1) * MAX_BEACON_H);
  const color    = civ.epoch_color || "#64748b";
  const status   = civ.status || "active";
  const evType   = eventMap[civ.id] || null;

  // ビーコン根元 (太陽から baseRadius 離れた点)
  const rootX = CX + (baseRadius + SUN_R) * Math.cos(angleRad);
  const rootY = CY + (baseRadius + SUN_R) * Math.sin(angleRad);

  // ビーコン先端
  const tipX = CX + (baseRadius + SUN_R + beaconH) * Math.cos(angleRad);
  const tipY = CY + (baseRadius + SUN_R + beaconH) * Math.sin(angleRad);

  beaconPositions.push({rootX, rootY, tipX, tipY, civ, color, angleDeg});

  // per-civ linearGradient
  const gradId = `bg-${i}`;
  const grad = appendTo(defs, el("linearGradient", {
    id: gradId,
    gradientUnits: "userSpaceOnUse",
    x1: rootX, y1: rootY, x2: tipX, y2: tipY
  }));
  const s1 = appendTo(grad, el("stop", {offset:"0%"}));
  s1.setAttribute("stop-color", color); s1.setAttribute("stop-opacity", "0.9");
  const s2 = appendTo(grad, el("stop", {offset:"100%"}));
  s2.setAttribute("stop-color", color); s2.setAttribute("stop-opacity", "0");

  // ビーコン本体 (line)
  let animClass = "b-active";
  if (status === "collapsed" || evType === "collapse") animClass = "b-collapse";
  else if (status === "ascended" || evType === "ascension") animClass = "b-ascend";

  const beacon = appendTo(svg, el("line", {
    x1: rootX, y1: rootY, x2: tipX, y2: tipY,
    stroke: `url(#${gradId})`,
    "stroke-width": Math.max(3, (civ.tech_level || 0.2) * 7),
    "stroke-linecap": "round",
    filter: "url(#glow-beacon)",
    class: animClass,
  }));

  // ビーコン先端の光点
  const dot = appendTo(svg, el("circle", {
    cx: tipX, cy: tipY, r: 4,
    fill: color,
    filter: "url(#glow-beacon)",
    opacity: status === "active" ? 1 : 0.3,
  }));

  // 文明名ラベル
  const labelDist = baseRadius + SUN_R + beaconH + 14;
  const labelX = CX + labelDist * Math.cos(angleRad);
  const labelY = CY + labelDist * Math.sin(angleRad);
  const label = appendTo(svg, el("text", {
    x: labelX, y: labelY,
    "text-anchor": angleDeg < 180 ? "start" : "end",
    fill: color,
    "font-size": "9",
    opacity: "0.85",
  }));
  label.textContent = civ.id || `CIV-${i}`;

  // ホバーツールチップ
  const hitArea = appendTo(svg, el("line", {
    x1: rootX, y1: rootY, x2: tipX, y2: tipY,
    stroke: "transparent",
    "stroke-width": 14,
    class: "civ-hover",
    "data-id": civ.id || "",
    "data-tech": (civ.tech_level || 0).toFixed(2),
    "data-epoch": civ.epoch || 1,
    "data-status": status,
  }));
  hitArea.style.cursor = "pointer";
  hitArea.addEventListener("mouseenter", (e) => {
    const tip = document.getElementById("tip");
    tip.innerHTML = `<b>${civ.id || "?"}</b><br>
      技術水準: ${((civ.tech_level||0)*100).toFixed(0)}%<br>
      エポック: ${civ.epoch || 1}<br>
      状態: ${status}`;
    tip.style.display = "block";
    tip.style.left = (e.clientX + 12) + "px";
    tip.style.top  = (e.clientY - 10) + "px";
  });
  hitArea.addEventListener("mouseleave", () => {
    document.getElementById("tip").style.display = "none";
  });
});

// ── イベントアーク (文明間の攻防) ────────────────────────────────────
(DATA.events || []).forEach(ev => {
  if (!ev.source || !ev.target) return;
  const si = beaconPositions.findIndex(b => b.civ.id === ev.source);
  const ti = beaconPositions.findIndex(b => b.civ.id === ev.target);
  if (si === -1 || ti === -1) return;

  const sp = beaconPositions[si];
  const tp = beaconPositions[ti];

  const isWar   = ev.event_type === "war" || ev.event_type === "betrayal";
  const isAlly  = ev.event_type === "alliance";
  const color   = isWar ? "#ef4444" : isAlly ? "#22c55e" : "#fde68a";
  const arcClass= isWar ? "arc-war" : "arc-ally";

  // 制御点: 太陽より少し手前 (中間点の 60% 位置)
  const mx = (sp.tipX + tp.tipX) / 2;
  const my = (sp.tipY + tp.tipY) / 2;
  const cpx = CX + (mx - CX) * 0.5;
  const cpy = CY + (my - CY) * 0.5;

  const pathD = `M ${sp.tipX} ${sp.tipY} Q ${cpx} ${cpy} ${tp.tipX} ${tp.tipY}`;
  const arcLen = Math.hypot(tp.tipX - sp.tipX, tp.tipY - sp.tipY);

  appendTo(svg, el("path", {
    d: pathD,
    fill: "none",
    stroke: color,
    "stroke-width": isWar ? 2 : 1.2,
    "stroke-dasharray": isWar ? "6,4" : "4,5",
    "stroke-linecap": "round",
    opacity: 0.8,
    class: arcClass,
    filter: "url(#glow-beacon)",
  }));

  // イベントラベル (中間点)
  const evLabel = appendTo(svg, el("text", {
    x: cpx, y: cpy - 6,
    "text-anchor": "middle",
    fill: color,
    "font-size": "9",
    opacity: "0.9",
  }));
  const emojiMap = {war:"⚔️", alliance:"🤝", collapse:"💀", discovery:"🔭",
    growth:"📈", betrayal:"🗡️", revolution:"⚡", contact:"📡"};
  evLabel.textContent = (emojiMap[ev.event_type] || "●") + " " + (ev.title || ev.event_type);
});

// ── バッジ ───────────────────────────────────────────────────────────
const sol = DATA.solar || {};
document.getElementById("badge").innerHTML =
  `ラウンド: <span>${DATA.round_no || 0}</span>　` +
  `アルカイア暦: <span>A.${DATA.year || 0}</span>　` +
  `太陽フェーズ: <span>${sol.phase || "?"}</span>　` +
  `光度: <span>${(sol.luminosity || 1).toFixed(2)}L₀</span>　` +
  `文明数: <span>${civs.length}</span>`;
</script>
</body>
</html>"""


# ─── ヘルパー関数 ─────────────────────────────────────────────────────────────

def _get_latest_round() -> int:
    try:
        from novel_simulation import get_current_round
        return get_current_round()
    except Exception:
        return 0


def _get_active_civilizations() -> list[dict]:
    try:
        from novelist_agent import get_civilization_registry
        return get_civilization_registry()
    except Exception:
        return []


def _get_latest_events() -> list[dict]:
    try:
        from novel_simulation import get_round_history
        hist = get_round_history(limit=1)
        return hist[0]["events"] if hist else []
    except Exception:
        return []


def _get_solar(round_no: int) -> dict:
    try:
        from novel_simulation import solar_state, round_to_t_gyr
        from dataclasses import asdict
        t = round_to_t_gyr(round_no)
        return asdict(solar_state(t))
    except Exception:
        return {"luminosity": 1.0, "phase": "主系列", "hz_inner_au": 0.95,
                "hz_outer_au": 1.37, "radius_rsun": 1.0, "nature_epoch": 1, "t_gyr": 0.0}


def _civs_to_solar_data(civs: list[dict], round_no: int) -> list[dict]:
    """civilization_registry → __SOLAR_DATA__ 用 civilizations リスト変換"""
    from novel_simulation import EPOCH_COLORS
    n = len(civs)
    result = []
    for i, civ in enumerate(civs):
        epoch = int(civ.get("civ_era", "1").split("-")[0]) if civ.get("civ_era") else 1
        epoch = max(1, min(5, epoch))
        result.append({
            "id": civ.get("company_name", f"CIV-{i}"),
            "angle_deg": round((i * 360 / max(n, 1)) % 360, 1),
            "tech_level": _tech_level_from_stage(civ.get("civ_stage", "")),
            "epoch": epoch,
            "epoch_color": EPOCH_COLORS.get(epoch, "#64748b"),
            "status": civ.get("status", "active"),
        })
    return result


def _tech_level_from_stage(stage: str) -> float:
    """文明ステージ文字列から tech_level (0〜1) を推定"""
    stage = (stage or "").lower()
    mapping = [
        (["初期", "原始", "initial", "early"], 0.15),
        (["発展", "develop", "growth"], 0.35),
        (["成熟", "mature", "peak"], 0.6),
        (["高度", "advanced", "stellar"], 0.8),
        (["超越", "transcend", "ascend"], 0.95),
    ]
    for keywords, val in mapping:
        if any(k in stage for k in keywords):
            return val
    return 0.3  # デフォルト


def _render_solar_system(round_no: int) -> None:
    """太陽系ビジュアルをレンダリングする。"""
    civs     = _get_active_civilizations()
    events   = _get_latest_events()
    sol      = _get_solar(round_no)

    solar_data = json.dumps({
        "round_no": round_no,
        "year": round_no * 100,
        "solar": sol,
        "civilizations": _civs_to_solar_data(civs, round_no),
        "events": events,
    }, ensure_ascii=False)

    html = _SOLAR_TEMPLATE.replace("__SOLAR_DATA__", solar_data)
    html = html.replace("__HEIGHT__", "600")
    components.html(html, height=620, scrolling=False)


# ─── メインレンダリング関数 ───────────────────────────────────────────────────

def render_novel_simulation() -> None:
    """文明年代記ページのメインエントリポイント。"""
    st.title("🌌 文明年代記 — アルカイアの記録")

    # ── 自動実行フラグ初期化 ────────────────────────────────────────
    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False
    if "sim_round" not in st.session_state:
        st.session_state.sim_round = _get_latest_round()

    # ── コントロールUI ───────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    with col1:
        if st.button("▶ 自動実行", type="primary",
                     disabled=st.session_state.sim_running,
                     key="sim_start"):
            st.session_state.sim_running = True
            st.rerun()
    with col2:
        if st.button("⏹ 停止", key="sim_stop",
                     disabled=not st.session_state.sim_running):
            st.session_state.sim_running = False
    with col3:
        if st.button("⏭ 1ラウンド", key="sim_step",
                     disabled=st.session_state.sim_running):
            _run_one_round()

    with col4:
        st.caption(
            f"ラウンド **{st.session_state.sim_round}** | "
            f"アルカイア暦 **A.{st.session_state.sim_round * 100}**"
        )

    # ── 自動ループ ───────────────────────────────────────────────────
    if st.session_state.sim_running:
        result = _run_one_round()
        if result.get("error"):
            st.error(result["error"])
            st.session_state.sim_running = False
        else:
            time.sleep(1.2)
            st.rerun()

    # ── 太陽系ビジュアル ─────────────────────────────────────────────
    _render_solar_system(st.session_state.sim_round)

    # ── タブ ─────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📜 イベントログ", "☀️ 太陽の状態", "🤖 文豪ナラティブ"])

    with tab1:
        _render_event_log()

    with tab2:
        _render_solar_info()

    with tab3:
        _render_archaia_log()


def _run_one_round() -> dict:
    """1ラウンド実行してセッション状態を更新する。"""
    from novel_simulation import run_simulation_round
    with st.spinner(f"⚙️ ラウンド {st.session_state.sim_round + 1} シミュレーション中…"):
        result = run_simulation_round()
    if not result.get("error"):
        st.session_state.sim_round += 1
    return result


def _render_event_log() -> None:
    """最新イベントログを表示する。"""
    from novel_simulation import get_round_history, EVENT_TYPES
    history = get_round_history(limit=15)
    if not history:
        st.info("まだシミュレーションが実行されていません。「▶ 自動実行」で開始してください。")
        return

    for rnd in history:
        with st.expander(
            f"**第{rnd['round_no']}ラウンド** — A.{rnd['year']} 年  |  {rnd['summary'][:40] if rnd['summary'] else ''}",
            expanded=(rnd == history[0])
        ):
            sol_raw = None
            try:
                import sqlite3, os
                _db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "data", "novelist_agent.db")
                conn = sqlite3.connect(_db)
                row = conn.execute(
                    "SELECT solar_state_json, nature_epoch FROM simulation_rounds WHERE round_no=?",
                    (rnd["round_no"],)
                ).fetchone()
                conn.close()
                if row and row[0]:
                    sol_raw = json.loads(row[0])
            except Exception:
                pass

            if sol_raw:
                from novel_simulation import NATURE_EPOCHS
                ep = sol_raw.get("nature_epoch", 1)
                ep_info = NATURE_EPOCHS.get(ep, NATURE_EPOCHS[1])
                st.caption(
                    f"☀️ {sol_raw['phase']} / 光度 {sol_raw['luminosity']:.2f}L₀ "
                    f"/ HZ {sol_raw['hz_inner_au']:.2f}〜{sol_raw['hz_outer_au']:.2f} AU "
                    f"/ エポック: {ep_info['name']}"
                )

            for ev in rnd["events"]:
                et = ev.get("event_type", "contact")
                info = EVENT_TYPES.get(et, {"emoji": "●", "color": "#94a3b8"})
                st.markdown(
                    f"{info['emoji']} **{ev.get('title', et)}** — {ev.get('civ', '?')} — "
                    f"<span style='color:{info['color']}'>{et}</span> | {ev.get('description','')[:60]}",
                    unsafe_allow_html=True,
                )


def _render_solar_info() -> None:
    """現在の太陽状態とエポック情報を表示する。"""
    from novel_simulation import NATURE_EPOCHS, solar_state, round_to_t_gyr
    import plotly.graph_objects as go

    t_now = round_to_t_gyr(st.session_state.sim_round)
    sol   = _get_solar(st.session_state.sim_round)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("太陽フェーズ", sol["phase"])
    c2.metric("光度 (L₀)", f"{sol['luminosity']:.3f}")
    c3.metric("太陽半径 (R☉)", f"{sol['radius_rsun']:.2f}")
    c4.metric("経過時間", f"{t_now:.3f} Gyr")

    # L(t) グラフ
    from dataclasses import asdict
    t_vals = [i * 0.1 for i in range(101)]
    l_vals = [solar_state(t).luminosity for t in t_vals]
    r_vals = [min(solar_state(t).radius_rsun, 220) for t in t_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_vals, y=l_vals, name="光度 L(t)", line=dict(color="#fbbf24", width=2)))
    fig.add_trace(go.Scatter(x=t_vals, y=r_vals, name="半径 R(t)", line=dict(color="#f97316", width=1.5, dash="dot"),
                             yaxis="y2"))
    fig.add_vline(x=t_now, line_color="#ef4444", line_dash="dash", annotation_text="現在")

    fig.update_layout(
        xaxis_title="時間 (Gyr)",
        yaxis_title="光度 (L₀)",
        yaxis2=dict(title="半径 (R☉)", overlaying="y", side="right"),
        paper_bgcolor="rgba(13,17,23,0)", plot_bgcolor="rgba(22,27,34,0.5)",
        font=dict(color="#c9d1d9"), height=260, margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)

    # エポック一覧
    st.markdown("#### 「自然」の定義の変容")
    for ep, info in NATURE_EPOCHS.items():
        active = sol.get("nature_epoch", 1) == ep
        prefix = "► " if active else "　"
        color  = info["color"]
        st.markdown(
            f"<span style='color:{color}'>{prefix}**エポック{ep}「{info['name']}」**</span> "
            f"({info['t_range'][0]}〜{info['t_range'][1]} Gyr)　— {info['definition']}",
            unsafe_allow_html=True,
        )


def _render_archaia_log() -> None:
    """文豪ナラティブログを表示する。"""
    from novel_simulation import get_archaia_log
    logs = get_archaia_log(limit=20)

    if not logs:
        st.info("まだ文豪ナラティブが生成されていません。collapse/ascension イベントが発生すると Gemini が散文詩を生成します。")
        return

    from novel_prompts import BUNGO_STYLES
    # 文体フィルター
    all_styles = ["すべて"] + list(BUNGO_STYLES.keys())
    chosen = st.selectbox("文体フィルター", all_styles, key="bungo_filter")

    for log in logs:
        if chosen != "すべて" and log.get("bungo_style") != chosen:
            continue
        event_emoji = {"collapse": "💀", "ascension": "✨"}.get(log["event_type"], "●")
        with st.container(border=True):
            st.caption(
                f"{event_emoji} **{log['civ_name']}** — {log['event_type']} "
                f"| ラウンド {log['round_no']} | 文体: {log['bungo_style'] or '?'}"
            )
            st.markdown(
                f"<div style='color:#c9d1d9;line-height:1.9;font-family:serif;font-size:13px;'>"
                f"{log['narrative']}</div>",
                unsafe_allow_html=True,
            )
