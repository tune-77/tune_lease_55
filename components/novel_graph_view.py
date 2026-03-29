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
  body { margin: 0; background: #0f172a; font-family: sans-serif; overflow: hidden; }
  .tooltip {
    position: absolute; background: rgba(15,23,42,0.95); color: #e2e8f0;
    border: 1px solid #334155; border-radius: 8px; padding: 10px 14px;
    font-size: 12px; pointer-events: none; display: none;
    line-height: 1.7; max-width: 220px; z-index: 10;
  }
  .legend {
    position: absolute; bottom: 12px; left: 12px; color: #94a3b8; font-size: 11px;
  }
  .legend-item { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .legend-line { width: 20px; height: 3px; flex-shrink: 0; }
  .stats-badge {
    position: absolute; top: 12px; right: 12px;
    background: rgba(15,23,42,0.8); border: 1px solid #334155;
    border-radius: 6px; padding: 6px 10px; color: #94a3b8; font-size: 11px;
  }
</style>
</head>
<body>
<div class="tooltip" id="tooltip"></div>
<div class="stats-badge" id="stats">読み込み中...</div>
<div class="legend" id="legend"></div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const GRAPH = __GRAPH_DATA__;
const REL_TYPES = __REL_TYPES__;
const W = window.innerWidth, H = __HEIGHT__;

// 凡例を生成
const legend = document.getElementById("legend");
const relEntries = Object.entries(REL_TYPES);
const legendHtml = relEntries.map(([key, info]) =>
  `<div class="legend-item">
    <div class="legend-line" style="background:${info.color}"></div>
    <span>${info.label}</span>
  </div>`
).join("") +
`<div class="legend-item" style="margin-top:6px">
  <div class="legend-dot" style="background:#3b82f6"></div><span>エージェント</span>
</div>
<div class="legend-item">
  <div class="legend-dot" style="background:#94a3b8"></div><span>企業・文明</span>
</div>
<div style="margin-top:6px;color:#64748b">線の太さ = |strength|</div>`;
legend.innerHTML = legendHtml;

// 統計バッジ
const agentCount = GRAPH.nodes.filter(n => n.group === "agent").length;
const companyCount = GRAPH.nodes.filter(n => n.group === "company").length;
const epInfo = GRAPH.episode_no != null ? `  ep.${GRAPH.episode_no}まで` : "  全エピソード";
document.getElementById("stats").textContent =
  `エージェント: ${agentCount}  企業: ${companyCount}  エッジ: ${GRAPH.links.length}${epInfo}`;

const svg = d3.select("body").append("svg")
  .attr("width", W).attr("height", H)
  .call(d3.zoom().scaleExtent([0.3, 4]).on("zoom", e => g.attr("transform", e.transform)));

const g = svg.append("g");

// マーカー定義（矢印）
const defs = svg.append("defs");
const relKeys = Object.keys(REL_TYPES);
relKeys.forEach(key => {
  defs.append("marker")
    .attr("id", `arrow-${key}`)
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 22)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", REL_TYPES[key].color)
    .attr("opacity", 0.7);
});

// シミュレーション
const sim = d3.forceSimulation(GRAPH.nodes)
  .force("link", d3.forceLink(GRAPH.links).id(d => d.id).distance(120).strength(0.5))
  .force("charge", d3.forceManyBody().strength(-200))
  .force("center", d3.forceCenter(W / 2, H / 2))
  .force("collide", d3.forceCollide(d => d.size + 10));

// エッジ
const link = g.append("g").selectAll("line")
  .data(GRAPH.links).enter().append("line")
  .attr("stroke", d => d.color)
  .attr("stroke-width", d => d.width)
  .attr("stroke-opacity", d => d.opacity)
  .attr("marker-end", d => `url(#arrow-${d.rel_type})`);

// エッジラベル
const linkLabel = g.append("g").selectAll("text")
  .data(GRAPH.links).enter().append("text")
  .attr("fill", d => d.color)
  .attr("font-size", "9px")
  .attr("text-anchor", "middle")
  .attr("opacity", 0.7)
  .text(d => d.rel_label);

// ノード
const node = g.append("g").selectAll("g")
  .data(GRAPH.nodes).enter().append("g")
  .call(d3.drag()
    .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
  );

// ノード形状（エージェント=六角形風、企業=円）
node.each(function(d) {
  const el = d3.select(this);
  if (d.group === "agent") {
    // 六角形ポリゴン
    el.append("polygon")
      .attr("points", hexPoints(0, 0, d.size))
      .attr("fill", d.color)
      .attr("stroke", "#1e293b")
      .attr("stroke-width", 2)
      .attr("opacity", 0.92);
  } else {
    el.append("circle")
      .attr("r", d.size)
      .attr("fill", d.color)
      .attr("stroke", "#1e293b")
      .attr("stroke-width", 1.5)
      .attr("opacity", 0.85);
  }
});

// ノードラベル
node.append("text")
  .attr("text-anchor", "middle")
  .attr("dy", d => d.size + 13)
  .attr("fill", "#e2e8f0")
  .attr("font-size", d => d.group === "agent" ? "11px" : "9px")
  .attr("font-weight", d => d.group === "agent" ? "bold" : "normal")
  .text(d => d.label);

// ツールチップ
const tooltip = document.getElementById("tooltip");
node
  .on("mouseover", (e, d) => {
    // そのノードに繋がるエッジを強調
    link.attr("stroke-opacity", l =>
      l.source.id === d.id || l.target.id === d.id ? 0.95 : 0.06);
    linkLabel.attr("opacity", l =>
      l.source.id === d.id || l.target.id === d.id ? 1.0 : 0);
    node.selectAll("circle, polygon").attr("opacity", n =>
      n.id === d.id || GRAPH.links.some(l =>
        (l.source.id === d.id && l.target.id === n.id) ||
        (l.target.id === d.id && l.source.id === n.id)) ? 1.0 : 0.2);
    tooltip.style.display = "block";
    tooltip.innerHTML = `<b>${d.label}</b><br>種別: ${d.group === "agent" ? "エージェント" : "企業・文明"}`;
  })
  .on("mousemove", e => {
    tooltip.style.left = (e.pageX + 14) + "px";
    tooltip.style.top = (e.pageY - 30) + "px";
  })
  .on("mouseout", () => {
    tooltip.style.display = "none";
    link.attr("stroke-opacity", d => d.opacity);
    linkLabel.attr("opacity", 0.7);
    node.selectAll("circle, polygon").attr("opacity", d => d.group === "agent" ? 0.92 : 0.85);
  });

// エッジホバー
link
  .on("mouseover", (e, d) => {
    const sign = d.strength >= 0 ? "+" : "";
    tooltip.innerHTML =
      `<b>${d.source.id} → ${d.target.id}</b><br>` +
      `関係: ${d.rel_label}<br>` +
      `強度: ${sign}${d.strength.toFixed(1)}<br>` +
      `${d.note ? "備考: " + d.note : ""}` +
      `<br><span style="color:#64748b">第${d.episode_no}話</span>`;
    tooltip.style.display = "block";
  })
  .on("mousemove", e => {
    tooltip.style.left = (e.pageX + 14) + "px";
    tooltip.style.top = (e.pageY - 30) + "px";
  })
  .on("mouseout", () => { tooltip.style.display = "none"; });

// Tick
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
  let pts = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    pts.push(`${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`);
  }
  return pts.join(" ");
}
</script>
</body>
</html>
"""


def render_novel_graph(episode_no: int | None = None, height: int = 500) -> None:
    """
    小説AI関係性グラフを描画する。
    episode_no: Noneなら全エピソード累積、指定するとそのエピソードまでの状態を表示。
    """
    graph_data = build_d3_graph_data(episode_no=episode_no)
    graph_data["episode_no"] = episode_no  # D3側で使用

    rel_types_for_js = {
        k: {"color": v["color"], "label": v["label"]}
        for k, v in REL_TYPES.items()
    }

    html = (
        _D3_TEMPLATE
        .replace("__GRAPH_DATA__", json.dumps(graph_data, ensure_ascii=False))
        .replace("__REL_TYPES__", json.dumps(rel_types_for_js, ensure_ascii=False))
        .replace("__HEIGHT__", str(height))
    )
    components.html(html, height=height, scrolling=False)
