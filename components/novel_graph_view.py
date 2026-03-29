# -*- coding: utf-8 -*-
"""
novel_graph_view.py
===================
小説AI「波乱丸」の登場人物・企業間の関係グラフを
D3.js IroFish風フォースグラフで描画するモジュール。

novel_graph.py からデータを取得し、Streamlit の components.html() で表示する。
"""
import json
import streamlit as st
import streamlit.components.v1 as components

from novel_graph import build_d3_graph_data, get_episode_history, REL_TYPES


_D3_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #00010d; font-family: 'Courier New', monospace; overflow: hidden; }

  /* ネオン・グロートゥールチップ */
  .tooltip {
    position: absolute;
    background: rgba(0,1,20,0.92);
    color: #a5f3fc;
    border: 1px solid rgba(56,189,248,0.5);
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 11px;
    pointer-events: none;
    display: none;
    line-height: 1.8;
    max-width: 240px;
    z-index: 10;
    box-shadow: 0 0 12px rgba(56,189,248,0.3);
  }

  /* 凡例 */
  .legend {
    position: absolute; bottom: 14px; left: 14px;
    color: #475569; font-size: 10px; font-family: 'Courier New', monospace;
  }
  .legend-item { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
  .legend-line { width: 18px; height: 2px; flex-shrink: 0; }

  /* 統計バッジ */
  .stats-badge {
    position: absolute; top: 12px; right: 14px;
    background: rgba(0,1,20,0.8);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 4px; padding: 5px 10px;
    color: #38bdf8; font-size: 10px;
    font-family: 'Courier New', monospace;
    letter-spacing: 0.05em;
  }

  /* スキャンライン */
  body::after {
    content: '';
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 2px,
      rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px
    );
    pointer-events: none; z-index: 5;
  }

  /* パルスアニメーション（エージェントノード） */
  @keyframes pulse-ring {
    0%   { transform: scale(1);   opacity: 0.8; }
    50%  { transform: scale(1.18); opacity: 0.3; }
    100% { transform: scale(1);   opacity: 0.8; }
  }
  .pulse { animation: pulse-ring 2.4s ease-in-out infinite; }

  /* 危険エッジ フロー（ダッシュアニメーション） */
  @keyframes flow-dash {
    from { stroke-dashoffset: 20; }
    to   { stroke-dashoffset: 0; }
  }
  .edge-danger  { animation: flow-dash 0.6s linear infinite; }
  .edge-warning { animation: flow-dash 1.4s linear infinite; }

  /* リスクバッジ */
  .risk-badge {
    position: absolute; top: 12px; left: 14px;
    background: rgba(0,1,20,0.8);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 4px; padding: 5px 10px;
    color: #fca5a5; font-size: 10px;
    font-family: 'Courier New', monospace;
    letter-spacing: 0.05em;
    display: none;
  }
</style>
</head>
<body>
<div class="tooltip" id="tooltip"></div>
<div class="stats-badge" id="stats">LOADING...</div>
<div class="risk-badge" id="risk-badge"></div>
<div class="legend" id="legend"></div>
<canvas id="stars"></canvas>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const GRAPH = __GRAPH_DATA__;
const REL_TYPES = __REL_TYPES__;
const W = window.innerWidth, H = __HEIGHT__;

// ── 星空背景 ──────────────────────────────────────────────
const canvas = document.getElementById("stars");
canvas.width = W; canvas.height = H;
canvas.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;";
const ctx = canvas.getContext("2d");
for (let i = 0; i < 180; i++) {
  const x = Math.random() * W, y = Math.random() * H;
  const r = Math.random() * 1.2;
  const a = Math.random() * 0.6 + 0.1;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = `rgba(200,220,255,${a})`;
  ctx.fill();
}

// ── 凡例 ──────────────────────────────────────────────────
const legend = document.getElementById("legend");
legend.innerHTML = Object.entries(REL_TYPES).map(([k, v]) =>
  `<div class="legend-item">
    <div class="legend-line" style="background:${v.color};box-shadow:0 0 4px ${v.color}"></div>
    <span style="color:${v.color}">${v.label}</span>
  </div>`
).join("") +
`<div class="legend-item" style="margin-top:6px">
  <div style="width:10px;height:10px;background:#38bdf8;clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);flex-shrink:0"></div>
  <span style="color:#38bdf8">エージェント</span>
</div>
<div class="legend-item">
  <div style="width:10px;height:10px;border-radius:50%;background:#475569;flex-shrink:0"></div>
  <span>企業・文明</span>
</div>`;

// ── 統計 ──────────────────────────────────────────────────
const agentN = GRAPH.nodes.filter(n => n.group === "agent").length;
const companyN = GRAPH.nodes.filter(n => n.group === "company").length;
const epStr = GRAPH.episode_no != null ? ` EP.${GRAPH.episode_no}` : " ALL";
const yearStr = GRAPH.sim_year > 0 ? `  ◈ A.${GRAPH.sim_year}` : "";
document.getElementById("stats").textContent =
  `AGENTS:${agentN}  CIVS:${companyN}  EDGES:${GRAPH.links.length}${epStr}${yearStr}`;

// リスクバッジ（高リスクエッジがある場合）
const highRisk = GRAPH.links.filter(l => (l.risk_level || 0) >= 0.7);
const riskBadge = document.getElementById("risk-badge");
if (highRisk.length > 0) {
  riskBadge.textContent = `⚠ HIGH RISK × ${highRisk.length}`;
  riskBadge.style.display = "block";
}

// ── SVG ───────────────────────────────────────────────────
const svg = d3.select("body").append("svg")
  .attr("width", W).attr("height", H)
  .style("position", "absolute").style("top", 0).style("left", 0)
  .call(d3.zoom().scaleExtent([0.2, 5]).on("zoom", e => g.attr("transform", e.transform)));

const defs = svg.append("defs");

// グローフィルター
const glow = defs.append("filter").attr("id", "glow").attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
glow.append("feGaussianBlur").attr("stdDeviation", "4").attr("result", "blur");
const merge = glow.append("feMerge");
merge.append("feMergeNode").attr("in", "blur");
merge.append("feMergeNode").attr("in", "SourceGraphic");

// 強グローフィルター（エージェント用）
const glowStrong = defs.append("filter").attr("id", "glow-strong").attr("x", "-80%").attr("y", "-80%").attr("width", "260%").attr("height", "260%");
glowStrong.append("feGaussianBlur").attr("stdDeviation", "8").attr("result", "blur");
const merge2 = glowStrong.append("feMerge");
merge2.append("feMergeNode").attr("in", "blur");
merge2.append("feMergeNode").attr("in", "SourceGraphic");

// 矢印マーカー
Object.entries(REL_TYPES).forEach(([key, info]) => {
  defs.append("marker")
    .attr("id", `arrow-${key}`)
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 24).attr("refY", 0)
    .attr("markerWidth", 5).attr("markerHeight", 5)
    .attr("orient", "auto")
    .append("path").attr("d", "M0,-5L10,0L0,5")
    .attr("fill", info.color).attr("opacity", 0.9);
});

const g = svg.append("g");

// ── シミュレーション ───────────────────────────────────────
const sim = d3.forceSimulation(GRAPH.nodes)
  .force("link", d3.forceLink(GRAPH.links).id(d => d.id).distance(__LINK_DISTANCE__).strength(0.2))
  .force("charge", d3.forceManyBody().strength(__CHARGE__))
  .force("center", d3.forceCenter(W / 2, H / 2))
  .force("collide", d3.forceCollide(d => d.size + __COLLIDE__));

// ── エッジ ─────────────────────────────────────────────────
const link = g.append("g").selectAll("line")
  .data(GRAPH.links).enter().append("line")
  .attr("stroke", d => {
    const r = d.risk_level || 0;
    if (!d.auto && r >= 0.7) return "#ef4444";   // 高リスク: 赤
    if (!d.auto && r >= 0.4) return "#f97316";   // 中リスク: オレンジ
    return d.color;
  })
  .attr("stroke-width", d => {
    const r = d.risk_level || 0;
    return d.auto ? d.width : d.width + (r >= 0.7 ? 1.5 : r >= 0.4 ? 0.7 : 0);
  })
  .attr("stroke-opacity", d => d.opacity)
  .attr("stroke-dasharray", d => {
    if (d.auto) return null;
    const r = d.risk_level || 0;
    if (r >= 0.7) return "6,3";
    if (r >= 0.4) return "4,4";
    return null;
  })
  .attr("class", d => {
    if (d.auto) return "";
    const r = d.risk_level || 0;
    if (r >= 0.7) return "edge-danger";
    if (r >= 0.4) return "edge-warning";
    return "";
  })
  .attr("filter", d => {
    if (d.auto) return null;
    const r = d.risk_level || 0;
    return r >= 0.7 ? "url(#glow-strong)" : "url(#glow)";
  })
  .attr("marker-end", d => `url(#arrow-${d.rel_type})`);

// ── エッジラベル ───────────────────────────────────────────
const linkLabel = g.append("g").selectAll("text")
  .data(GRAPH.links).enter().append("text")
  .attr("fill", d => {
    const r = d.risk_level || 0;
    if (!d.auto && r >= 0.7) return "#ef4444";
    if (!d.auto && r >= 0.4) return "#f97316";
    return d.color;
  })
  .attr("font-size", "9px")
  .attr("font-family", "'Courier New', monospace")
  .attr("text-anchor", "middle")
  .attr("opacity", d => d.auto ? 0.4 : 0.9)
  .style("filter", d => {
    if (d.auto) return null;
    const r = d.risk_level || 0;
    const c = r >= 0.7 ? "#ef4444" : r >= 0.4 ? "#f97316" : d.color;
    return `drop-shadow(0 0 4px ${c})`;
  })
  .text(d => {
    const raw = d.note ? d.note : d.rel_label;
    return raw.length > 16 ? raw.slice(0, 15) + "…" : raw;
  });

// ── ノード ─────────────────────────────────────────────────
const node = g.append("g").selectAll("g")
  .data(GRAPH.nodes).enter().append("g")
  .call(d3.drag()
    .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on("drag",  (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on("end",   (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
  );

node.each(function(d) {
  const el = d3.select(this);
  if (d.group === "agent") {
    // パルスリング
    el.append("polygon")
      .attr("points", hexPoints(0, 0, d.size * 1.6))
      .attr("fill", "none")
      .attr("stroke", d.color)
      .attr("stroke-width", 1)
      .attr("opacity", 0.3)
      .attr("class", "pulse");
    // 本体六角形
    el.append("polygon")
      .attr("points", hexPoints(0, 0, d.size))
      .attr("fill", d.color)
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
      .attr("filter", "url(#glow-strong)")
      .attr("opacity", 0.95);
  } else {
    // 企業: 外リング
    el.append("circle").attr("r", d.size + 4)
      .attr("fill", "none")
      .attr("stroke", d.color)
      .attr("stroke-width", 0.8)
      .attr("opacity", 0.4);
    // 本体
    el.append("circle").attr("r", d.size)
      .attr("fill", "#00010d")
      .attr("stroke", d.color)
      .attr("stroke-width", 2)
      .attr("filter", "url(#glow)")
      .attr("opacity", 0.92);
  }
});

// ノードラベル
node.append("text")
  .attr("text-anchor", "middle")
  .attr("dy", d => d.size + 14)
  .attr("fill", d => d.group === "agent" ? d.color : "#94a3b8")
  .attr("font-size", d => d.group === "agent" ? "11px" : "9px")
  .attr("font-family", "'Courier New', monospace")
  .attr("font-weight", "bold")
  .style("filter", d => d.group === "agent" ? `drop-shadow(0 0 4px ${d.color})` : null)
  .text(d => d.label);

// ── ツールチップ ───────────────────────────────────────────
const tooltip = document.getElementById("tooltip");

node
  .on("mouseover", (e, d) => {
    link.attr("stroke-opacity", l =>
      l.source.id === d.id || l.target.id === d.id ? 1.0 : 0.04);
    linkLabel.attr("opacity", l =>
      l.source.id === d.id || l.target.id === d.id ? 1.0 : 0.08);
    node.selectAll("circle, polygon").attr("opacity", n =>
      n.id === d.id || GRAPH.links.some(l =>
        (l.source.id === d.id && l.target.id === n.id) ||
        (l.target.id === d.id && l.source.id === n.id)) ? 1.0 : 0.15);

    let html = `<b style="color:#7dd3fc">${d.label}</b><br>`;
    html += `<span style="color:#475569">${d.group === "agent" ? "◆ AGENT" : "○ CIVILIZATION"}</span>`;
    if (d.personality) html += `<br><span style="color:#a78bfa">⬡ ${d.personality}</span>`;
    if (d.traits)  html += `<br><span style="color:#86efac">▸ ${d.traits}</span>`;
    if (d.goals)   html += `<br><span style="color:#fde68a">▸ ${d.goals}</span>`;
    if (d.ideology) html += `<br><span style="color:#a5f3fc;font-size:10px">${d.ideology}</span>`;
    tooltip.innerHTML = html;
    tooltip.style.display = "block";
  })
  .on("mousemove", e => {
    tooltip.style.left = (e.pageX + 14) + "px";
    tooltip.style.top  = (e.pageY - 30) + "px";
  })
  .on("mouseout", () => {
    tooltip.style.display = "none";
    link.attr("stroke-opacity", d => d.opacity);
    linkLabel.attr("opacity", d => d.auto ? 0.4 : 0.9);
    node.selectAll("circle, polygon").attr("opacity", d => d.group === "agent" ? 0.95 : 0.92);
  });

link
  .on("mouseover", (e, d) => {
    const epLabel = d.episode_no >= 0 ? `EP.${d.episode_no}` : "GENESIS";
    const r = d.risk_level || 0;
    const riskColor = r >= 0.7 ? "#ef4444" : r >= 0.4 ? "#f97316" : "#22c55e";
    const riskLabel = r >= 0.7 ? "🔴 HIGH RISK" : r >= 0.4 ? "🟡 WARNING" : r > 0 ? "🟢 STABLE" : "";
    let html =
      `<b style="color:#7dd3fc">${d.source.id}</b>` +
      ` <span style="color:#38bdf8">→</span> ` +
      `<b style="color:#7dd3fc">${d.target.id}</b><br>` +
      `${d.note ? `<span>${d.note}</span><br>` : ""}` +
      `<span style="color:#475569">${d.rel_label} / ${epLabel}</span>`;
    if (d.prediction) {
      html += `<br><span style="color:${riskColor};font-size:10px">${riskLabel}</span>`;
      html += `<br><span style="color:#fde68a;font-size:10px">🔮 ${d.prediction}</span>`;
    }
    tooltip.innerHTML = html;
    tooltip.style.display = "block";
  })
  .on("mousemove", e => {
    tooltip.style.left = (e.pageX + 14) + "px";
    tooltip.style.top  = (e.pageY - 30) + "px";
  })
  .on("mouseout", () => { tooltip.style.display = "none"; });

// ── Tick ──────────────────────────────────────────────────
sim.on("tick", () => {
  link
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  linkLabel
    .attr("x", d => (d.source.x + d.target.x) / 2)
    .attr("y", d => (d.source.y + d.target.y) / 2);
  node.attr("transform", d => `translate(${d.x},${d.y})`);
});

// 六角形ポイント生成
function hexPoints(cx, cy, r) {
  return Array.from({length: 6}, (_, i) => {
    const a = (Math.PI / 3) * i - Math.PI / 6;
    return `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
  }).join(" ");
}
</script>
</body>
</html>
"""


def render_novel_graph(
    episode_no: int | None = None,
    height: int = 820,
    link_distance: int = 320,
    charge: int = -1200,
    collide: int = 70,
) -> None:
    """
    小説AI関係性グラフを描画する。
    episode_no: Noneなら全エピソード累積、指定するとそのエピソードまでの状態を表示。
    """
    try:
        from novel_simulation import get_current_year
        sim_year = get_current_year()
    except Exception:
        sim_year = 0

    graph_data = build_d3_graph_data(episode_no=episode_no)
    graph_data["episode_no"] = episode_no  # D3側で使用
    graph_data["sim_year"]   = sim_year    # シミュレーション年

    rel_types_for_js = {
        k: {"color": v["color"], "label": v["label"]}
        for k, v in REL_TYPES.items()
    }

    html = (
        _D3_TEMPLATE
        .replace("__GRAPH_DATA__", json.dumps(graph_data, ensure_ascii=False))
        .replace("__LINK_DISTANCE__", str(link_distance))
        .replace("__CHARGE__", str(-abs(charge)))
        .replace("__COLLIDE__", str(collide))
        .replace("__REL_TYPES__", json.dumps(rel_types_for_js, ensure_ascii=False))
        .replace("__HEIGHT__", str(height))
    )
    components.html(html, height=height, scrolling=False)
