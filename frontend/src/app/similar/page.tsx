"use client";
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Network, Activity, Users, FileCheck, FileX, Link as LinkIcon } from 'lucide-react';

const D3_TEMPLATE = `
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
    line-height: 1.6; max-width: 200px;
  }
  .legend { position: absolute; bottom: 12px; left: 12px; color: #94a3b8; font-size: 11px; }
  .legend-item { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
</style>
</head>
<body>
<div class="tooltip" id="tooltip"></div>
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
    const W = window.innerWidth, H = window.innerHeight;

    // Clear previous
    d3.select("body").selectAll("svg").remove();

    const svg = d3.select("body").append("svg")
    .attr("width", W).attr("height", H)
    .call(d3.zoom().scaleExtent([0.3, 4]).on("zoom", e => g.attr("transform", e.transform)));

    const g = svg.append("g");

    // シミュレーション
    const sim = d3.forceSimulation(GRAPH_DATA.nodes)
    .force("link", d3.forceLink(GRAPH_DATA.edges).distance(d => 120 - d.similarity * 60).strength(0.6))
    .force("charge", d3.forceManyBody().strength(-200))
    .force("center", d3.forceCenter(W / 2, H / 2))
    .force("collide", d3.forceCollide(d => d.radius + 10));

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
}
initGraph();
</script>
</body>
</html>
`;


export default function SimilarPage() {
  const [graphData, setGraphData] = useState<any>(null);
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    triggerMebuki('guide', '案件類似ネットワーク画面ですね！\n過去の案件から似たパターンのものを可視化します！');
    
    const fetchData = async () => {
      try {
        const res = await axios.get(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/similar/data`);
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

          <div className="bg-slate-900 rounded-2xl shadow-xl overflow-hidden border border-slate-700 relative h-[600px] w-full group">
            {/* ネットワーク本体 */}
            <iframe
              srcDoc={D3_TEMPLATE.replace('/*GRAPH_DATA_PLACEHOLDER*/ null', JSON.stringify(graphData || {}))}
              className="w-full h-full border-none"
              title="Cases D3 Network"
              sandbox="allow-scripts"
            />
          </div>

          {/* 類似ペアランキングリスト */}
          {graphData && graphData.edges && graphData.edges.length > 0 && (
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
              <h3 className="font-bold text-slate-800 text-lg mb-4 flex items-center gap-2">
                <LinkIcon className="w-5 h-5 text-teal-600" /> 最も類似しているペア・ランキング (Top 10)
              </h3>
              <div className="overflow-x-auto rounded-xl border border-slate-200">
                <table className="w-full text-sm text-left">
                  <thead className="bg-slate-100/80 sticky top-0 font-bold text-slate-600">
                    <tr>
                      <th className="px-4 py-3 border-b">ランク</th>
                      <th className="px-4 py-3 border-b">案件 A</th>
                      <th className="px-4 py-3 border-b">案件 B</th>
                      <th className="px-4 py-3 border-b text-right">類似度</th>
                    </tr>
                  </thead>
                  <tbody>
                    {graphData.edges
                      .sort((a: any, b: any) => b.similarity - a.similarity)
                      .slice(0, 10)
                      .map((e: any, i: number) => {
                        const na = graphData.nodes[e.source];
                        const nb = graphData.nodes[e.target];
                        // naやnbが存在しない場合の防御
                        if(!na || !nb) return null;
                        return (
                          <tr key={i} className="border-b last:border-b-0 hover:bg-slate-50">
                            <td className="px-4 py-3 text-slate-500 font-bold">#{i + 1}</td>
                            <td className="px-4 py-3 text-slate-800 font-bold">
                                {na.industry_sub.slice(0, 10)} <span className="text-slate-400 font-normal ml-2">({na.score.toFixed(0)}pt)</span>
                            </td>
                            <td className="px-4 py-3 text-slate-800 font-bold border-l">
                                {nb.industry_sub.slice(0, 10)} <span className="text-slate-400 font-normal ml-2">({nb.score.toFixed(0)}pt)</span>
                            </td>
                            <td className="px-4 py-3 text-right text-teal-600 font-black">
                                {e.similarity.toFixed(2)}
                            </td>
                          </tr>
                        );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}