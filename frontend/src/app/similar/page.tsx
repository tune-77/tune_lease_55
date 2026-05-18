"use client";
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Network, Activity, Users, FileCheck, FileX, Link as LinkIcon, SlidersHorizontal, RotateCcw } from 'lucide-react';

const D3_TEMPLATE = `
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; width: 100%; height: 100%; background: #0f172a; font-family: sans-serif; overflow: hidden; }
  #root { width: 100%; height: 100%; position: relative; overflow: hidden; }
  .tooltip {
    position: absolute; background: rgba(15,23,42,0.95); color: #e2e8f0;
    border: 1px solid #334155; border-radius: 8px; padding: 10px 14px;
    font-size: 12px; pointer-events: none; display: none;
    line-height: 1.6; max-width: 200px;
  }
  .legend { position: absolute; bottom: 12px; left: 12px; color: #94a3b8; font-size: 11px; }
  .legend-item { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .hint {
    position: absolute; top: 12px; right: 12px; color: #cbd5e1; font-size: 11px;
    background: rgba(15,23,42,0.82); border: 1px solid rgba(71,85,105,0.7);
    border-radius: 999px; padding: 6px 10px; pointer-events: none;
  }
</style>
</head>
<body>
<div id="root">
<div class="tooltip" id="tooltip"></div>
<div class="hint">ドラッグで移動・ホイールで拡大縮小</div>
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>現在の案件</div>
  <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>成約</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>失注</div>
  <div class="legend-item"><div class="legend-dot" style="background:#94a3b8"></div>未登録</div>
  <div style="margin-top:6px; color:#64748b; font-weight:bold;">線の太さ = 類似度の強さ</div>
</div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
function initGraph() {
    // GRAPH_DATA_PLACEHOLDER will be replaced string-wise with the actual JSON payload
    const GRAPH_DATA = /*GRAPH_DATA_PLACEHOLDER*/ null;
    if (!GRAPH_DATA || !GRAPH_DATA.nodes) return;
    const CONFIG = /*CONFIG_PLACEHOLDER*/ null;
    const W = window.innerWidth, H = window.innerHeight;
    const chargeStrength = CONFIG?.chargeStrength ?? -120;
    const linkBaseDistance = CONFIG?.linkBaseDistance ?? 85;
    const collisionPadding = CONFIG?.collisionPadding ?? 6;
    const fitOnLoad = CONFIG?.fitOnLoad ?? true;
    const zoomMax = CONFIG?.zoomMax ?? 4;
    const nodeScale = CONFIG?.nodeScale ?? 1;

    // Clear previous
    d3.select("#root").selectAll("svg").remove();

    const zoom = d3.zoom().scaleExtent([0.25, zoomMax]).on("zoom", e => g.attr("transform", e.transform));
    const svg = d3.select("#root").append("svg")
    .attr("width", W).attr("height", H)
    .call(zoom);

    const g = svg.append("g");

    // シミュレーション
    const sim = d3.forceSimulation(GRAPH_DATA.nodes)
    .force("link", d3.forceLink(GRAPH_DATA.edges).distance(d => Math.max(35, linkBaseDistance - d.similarity * 60)).strength(0.8))
    .force("charge", d3.forceManyBody().strength(chargeStrength))
    .force("center", d3.forceCenter(W / 2, H / 2))
    .force("collide", d3.forceCollide(d => d.radius + collisionPadding));

    // エッジ
    const link = g.append("g").selectAll("line")
    .data(GRAPH_DATA.edges).enter().append("line")
    .attr("stroke", "#475569")
    .attr("stroke-width", d => Math.max(1, d.width))
    .attr("stroke-opacity", d => d.opacity);

    // ノード
    const node = g.append("g").selectAll("g")
    .data(GRAPH_DATA.nodes).enter().append("g")
    .call(d3.drag()
        .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
    );

    // 円形（現在の案件用の星形ロジックは簡略化して円＋エフェクトで代用）
    node.each(function(d) {
    const el = d3.select(this);
    if (nodeScale !== 1) {
        d.radius = Math.max(8, d.radius * nodeScale);
    }
    if (d.is_current) {
        el.append("circle")
        .attr("r", d.radius + 4)
        .attr("fill", "transparent")
        .attr("stroke", d.color)
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "2,2");
    }
    el.append("circle")
        .attr("r", d.radius)
        .attr("fill", d.color)
        .attr("stroke", d.is_current ? "#fff" : "#1e293b")
        .attr("stroke-width", d.is_current ? 2 : 1.5)
        .attr("opacity", 0.9);
    });

    // ラベル
    node.append("text")
    .attr("text-anchor", "middle")
    .attr("dy", d => d.radius + 12)
    .attr("fill", "#cbd5e1")
    .attr("font-size", "9px")
    .text(d => d.industry_sub ? d.industry_sub.slice(0, 8) : "");

    // ツールチップ
    const tooltip = document.getElementById("tooltip");
    node
    .on("mouseover", (e, d) => {
        const rate = d.final_rate > 0 ? d.final_rate.toFixed(2) + "%" : "—";
        tooltip.innerHTML =
        '<b>' + d.industry_sub + '</b><br>' +
        'スコア: ' + d.score.toFixed(0) + '<br>' +
        '状態: ' + d.status + '<br>' +
        '獲得金利: ' + rate + '<br>' +
        '競合: ' + (d.competitor_name || "なし");
        tooltip.style.display = "block";
        link.attr("stroke-opacity", l =>
        l.source.id === d.id || l.target.id === d.id ? 0.9 : 0.05);
        node.selectAll("circle").attr("opacity", n =>
        n.id === d.id || GRAPH_DATA.edges.some(l =>
            (l.source.id === d.id && l.target.id === n.id) ||
            (l.target.id === d.id && l.source.id === n.id)) ? 1.0 : 0.25);
    })
    .on("mousemove", e => {
        tooltip.style.left = (e.pageX + 12) + "px";
        tooltip.style.top = (e.pageY - 28) + "px";
    })
    .on("mouseout", () => {
        tooltip.style.display = "none";
        link.attr("stroke-opacity", d => d.opacity);
        node.selectAll("circle").attr("opacity", 0.9);
    });

    sim.on("tick", () => {
    link
        .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("transform", d => 'translate(' + d.x + ',' + d.y + ')');
    });

    if (fitOnLoad) {
      sim.on("end", () => {
        const xs = GRAPH_DATA.nodes.map(n => n.x || 0);
        const ys = GRAPH_DATA.nodes.map(n => n.y || 0);
        const minX = Math.min(...xs), maxX = Math.max(...xs);
        const minY = Math.min(...ys), maxY = Math.max(...ys);
        const boundsW = Math.max(1, maxX - minX);
        const boundsH = Math.max(1, maxY - minY);
        const padding = 60;
        const scale = Math.min((W - padding * 2) / boundsW, (H - padding * 2) / boundsH, 1.8);
        const tx = W / 2 - ((minX + maxX) / 2) * scale;
        const ty = H / 2 - ((minY + maxY) / 2) * scale;
        svg.transition().duration(500).call(
          zoom.transform,
          d3.zoomIdentity.translate(tx, ty).scale(scale)
        );
      });
    }
}
initGraph();
</script>
</div>
</body>
</html>
`;


export default function SimilarPage() {
  const [graphData, setGraphData] = useState<any>(null);
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [chargeStrength, setChargeStrength] = useState(-120);
  const [linkBaseDistance, setLinkBaseDistance] = useState(85);
  const [collisionPadding, setCollisionPadding] = useState(6);
  const [nodeScale, setNodeScale] = useState(0.9);
  const [zoomMax, setZoomMax] = useState(4);
  const [fitOnLoad, setFitOnLoad] = useState(true);
  const [viewKey, setViewKey] = useState(0);

  useEffect(() => {
    triggerMebuki('guide', '案件類似ネットワーク画面ですね！\n過去の案件から似たパターンのものを可視化します！');
    
    const fetchData = async () => {
      try {
        const res = await axios.get(`/api/similar/data`);
        setGraphData(res.data);
        setSummary(res.data.summary);
      } catch (err) {
        console.error("Failed to load similar network data", err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[calc(100vh-2rem)]">
        <div className="flex flex-col items-center">
          <Activity className="w-12 h-12 text-teal-500 animate-spin mb-4" />
          <h2 className="text-xl font-bold text-slate-500">ネットワーク構築中...</h2>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
          <Network className="w-8 h-8 text-teal-600" />
          案件類似ネットワーク (D3.js)
        </h1>
        <p className="text-slate-500 font-bold mt-2">過去案件をノードとし、類似度（業種・スコア・競合）に応じてエッジで繋ぎます。</p>
      </div>

      <div className="mb-6 bg-white border border-slate-200 rounded-2xl p-4 shadow-sm">
        <div className="flex items-center gap-2 mb-4 text-slate-700 font-bold">
          <SlidersHorizontal className="w-4 h-4 text-teal-600" />
          調整画面
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          <label className="space-y-2">
            <div className="flex justify-between text-sm text-slate-600">
              <span>反発の強さ</span>
              <span className="font-bold text-slate-800">{chargeStrength}</span>
            </div>
            <input type="range" min={-600} max={-40} step={10} value={chargeStrength} onChange={(e) => setChargeStrength(Number(e.target.value))} className="w-full" />
          </label>
          <label className="space-y-2">
            <div className="flex justify-between text-sm text-slate-600">
              <span>リンク距離</span>
              <span className="font-bold text-slate-800">{linkBaseDistance}</span>
            </div>
            <input type="range" min={50} max={220} step={5} value={linkBaseDistance} onChange={(e) => setLinkBaseDistance(Number(e.target.value))} className="w-full" />
          </label>
          <label className="space-y-2">
            <div className="flex justify-between text-sm text-slate-600">
              <span>衝突余白</span>
              <span className="font-bold text-slate-800">{collisionPadding}</span>
            </div>
            <input type="range" min={0} max={40} step={1} value={collisionPadding} onChange={(e) => setCollisionPadding(Number(e.target.value))} className="w-full" />
          </label>
          <label className="space-y-2">
            <div className="flex justify-between text-sm text-slate-600">
              <span>ノードサイズ</span>
              <span className="font-bold text-slate-800">{nodeScale.toFixed(2)}x</span>
            </div>
            <input type="range" min={0.7} max={1.6} step={0.05} value={nodeScale} onChange={(e) => setNodeScale(Number(e.target.value))} className="w-full" />
          </label>
          <label className="space-y-2">
            <div className="flex justify-between text-sm text-slate-600">
              <span>ズーム上限</span>
              <span className="font-bold text-slate-800">{zoomMax.toFixed(1)}x</span>
            </div>
            <input type="range" min={1.5} max={8} step={0.5} value={zoomMax} onChange={(e) => setZoomMax(Number(e.target.value))} className="w-full" />
          </label>
          <label className="flex items-center gap-3 pt-7 text-sm font-bold text-slate-700">
            <input type="checkbox" checked={fitOnLoad} onChange={(e) => setFitOnLoad(e.target.checked)} className="w-4 h-4 rounded border-slate-300" />
            初期表示で全体をフィット
          </label>
        </div>
        <div className="mt-4 flex items-center gap-3">
          <button
            type="button"
            onClick={() => setViewKey((v) => v + 1)}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-teal-600 text-white font-bold shadow-sm hover:bg-teal-500 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            再配置
          </button>
          <div className="text-xs text-slate-500">広がりすぎるときは反発を弱め、リンク距離を短くすると詰まります。</div>
        </div>
      </div>

      {summary && summary.total < 2 ? (
        <div className="bg-amber-50 border border-amber-200 p-6 rounded-2xl flex items-start gap-4">
          <Activity className="w-8 h-8 text-amber-500 shrink-0" />
          <div>
            <h3 className="font-bold text-amber-800 text-lg">案件データが不足しています</h3>
            <p className="text-amber-700 mt-1">ネットワークを形成するためには、2件以上の案件が登録されている必要があります。</p>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* メトリクス */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 flex flex-col justify-between">
              <div className="text-sm font-bold text-slate-500 flex items-center gap-2"><Users className="w-4 h-4" /> 総ノード数</div>
              <div className="text-3xl font-black text-slate-800 mt-2">{summary?.total || 0}</div>
            </div>
            <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 flex flex-col justify-between">
              <div className="text-sm font-bold text-green-600 flex items-center gap-2"><FileCheck className="w-4 h-4" /> 成約数</div>
              <div className="text-3xl font-black text-green-600 mt-2">{summary?.won || 0}</div>
            </div>
            <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 flex flex-col justify-between">
              <div className="text-sm font-bold text-rose-600 flex items-center gap-2"><FileX className="w-4 h-4" /> 失注数</div>
              <div className="text-3xl font-black text-rose-600 mt-2">{summary?.lost || 0}</div>
            </div>
            <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 flex flex-col justify-between">
              <div className="text-sm font-bold text-teal-600 flex items-center gap-2"><LinkIcon className="w-4 h-4" /> エッジ接続数</div>
              <div className="text-3xl font-black text-teal-600 mt-2">{graphData?.edges?.length || 0}</div>
            </div>
          </div>

          <div className="bg-slate-900 rounded-2xl shadow-xl overflow-hidden border border-slate-700 relative h-[78vh] min-h-[760px] w-full group">
            {/* ネットワーク本体 */}
            <iframe
              key={viewKey}
              srcDoc={D3_TEMPLATE
                .replace('/*GRAPH_DATA_PLACEHOLDER*/ null', JSON.stringify(graphData || {}))
                .replace('/*CONFIG_PLACEHOLDER*/ null', JSON.stringify({
                  chargeStrength,
                  linkBaseDistance,
                  collisionPadding,
                  nodeScale,
                  zoomMax,
                  fitOnLoad,
                }))}
              className="w-full h-full border-none"
              title="Cases D3 Network"
              sandbox="allow-scripts"
            />
          </div>

        </div>
      )}
    </div>
  );
}
