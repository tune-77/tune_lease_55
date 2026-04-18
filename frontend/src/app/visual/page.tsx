"use client";
import React, { useEffect, useState, useMemo, useRef } from 'react';
import axios from 'axios';
import * as d3 from 'd3';
import { sankey as d3Sankey, sankeyLinkHorizontal } from 'd3-sankey';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Eye, Activity, ChartPie, Grid, GitMerge, MousePointer2 } from 'lucide-react';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

export default function VisualPage() {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'bubble' | 'heatmap' | 'sankey'>('bubble');
  const sankeyRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    triggerMebuki('guide', 'ビジュアルインサイト画面ですね！\n過去の案件をマッピングして成約の傾向を探ります！');
    
    const fetchData = async () => {
      try {
        const res = await axios.get(`${process.env.NEXT_PUBLIC_API_URL}/api/visual/data`);
        setData(res.data.cases || []);
      } catch (err) {
        console.error("Failed to load visual data", err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  // Sankeyデータ構築
  const sankeyData = useMemo(() => {
    if (!data.length || activeTab !== 'sankey') return null;

    const bands = [
      { min: 0, max: 50, label: "～50" },
      { min: 50, max: 65, label: "50〜65" },
      { min: 65, max: 80, label: "65〜80" },
      { min: 80, max: 200, label: "80〜" }
    ];

    const industries = Array.from(new Set(data.map(c => c.industry_major || "不明"))).slice(0, 8);
    const results = ["成約", "失注"];
    const bandLabels = bands.map(b => b.label);

    const nodes: any[] = [
      ...industries.map(name => ({ name, category: 'industry', color: '#6366f1' })),
      ...bandLabels.map(name => ({ name, category: 'band', color: '#8b5cf6' })),
      ...results.map(name => ({ name, category: 'result', color: name === '成約' ? '#22c55e' : '#ef4444' }))
    ];

    const nodeIndexMap = new Map(nodes.map((n, i) => [n.name, i]));
    const links: any[] = [];

    // 業種 -> スコア帯
    industries.forEach(ind => {
      bandLabels.forEach(bandLabel => {
        const band = bands.find(b => b.label === bandLabel)!;
        const count = data.filter(c => (c.industry_major || "不明") === ind && c.score >= band.min && c.score < band.max).length;
        if (count > 0) {
          links.push({
            source: nodeIndexMap.get(ind),
            target: nodeIndexMap.get(bandLabel),
            value: count,
            color: 'rgba(99, 102, 241, 0.2)'
          });
        }
      });
    });

    // スコア帯 -> 結果
    bandLabels.forEach(bandLabel => {
      const band = bands.find(b => b.label === bandLabel)!;
      results.forEach(res => {
        const count = data.filter(c => c.score >= band.min && c.score < band.max && c.status === res).length;
        if (count > 0) {
          links.push({
            source: nodeIndexMap.get(bandLabel),
            target: nodeIndexMap.get(res),
            value: count,
            color: res === '成約' ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)'
          });
        }
      });
    });

    return { nodes, links };
  }, [data, activeTab]);

  // Sankey描画用
  useEffect(() => {
    if (!sankeyData || !sankeyRef.current || activeTab !== 'sankey') return;

    const width = 900;
    const height = 500;
    const svg = d3.select(sankeyRef.current);

    svg.selectAll('*').remove();

    const generator = d3Sankey()
      .nodeWidth(20)
      .nodePadding(20)
      .extent([[1, 1], [width - 1, height - 5]]);

    const { nodes, links } = generator({
      nodes: sankeyData.nodes.map(d => Object.assign({}, d)),
      links: sankeyData.links.map(d => Object.assign({}, d))
    });

    // Links
    svg.append('g')
      .attr('fill', 'none')
      .selectAll('path')
      .data(links)
      .join('path')
      .attr('d', sankeyLinkHorizontal())
      .attr('stroke', (d: any) => d.color)
      .attr('stroke-width', (d: any) => Math.max(1, d.width))
      .style('mix-blend-mode', 'multiply')
      .attr('opacity', 0.6)
      .on('mouseover', function() { d3.select(this).attr('opacity', 0.9).attr('stroke', (d: any) => d.target.color); })
      .on('mouseout', function() { d3.select(this).attr('opacity', 0.6).attr('stroke', (d: any) => d.color); });

    // Nodes
    const nodeNodes = svg.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g');

    nodeNodes.append('rect')
      .attr('x', (d: any) => d.x0)
      .attr('y', (d: any) => d.y0)
      .attr('height', (d: any) => d.y1 - d.y0)
      .attr('width', (d: any) => d.x1 - d.x0)
      .attr('fill', (d: any) => d.color)
      .attr('rx', 4);

    nodeNodes.append('text')
      .attr('x', (d: any) => d.x0 < width / 2 ? d.x1 + 6 : d.x0 - 6)
      .attr('y', (d: any) => (d.y1 + d.y0) / 2)
      .attr('dy', '0.35em')
      .attr('text-anchor', (d: any) => d.x0 < width / 2 ? 'start' : 'end')
      .attr('font-size', '11px')
      .attr('font-weight', 'bold')
      .attr('fill', '#475569')
      .text((d: any) => d.name);

  }, [sankeyData, activeTab]);

  // バブルチャート用データの整形
  const { wonCases, lostCases } = useMemo(() => {
    const won = data.filter(c => c.status === '成約');
    const lost = data.filter(c => c.status === '失注');
    return { wonCases: won, lostCases: lost };
  }, [data]);

  // ヒートマップ用のデータ整形
  const heatmapData = useMemo(() => {
    if (!data.length) return { matrix: [], bands: [], industries: [] };
    const bandsList = [
      { min: 0, max: 50, label: "～50" },
      { min: 50, max: 65, label: "50〜65" },
      { min: 65, max: 80, label: "65〜80" },
      { min: 80, max: 200, label: "80〜" }
    ];
    const industryCounts:Record<string, number> = {};
    data.forEach(c => {
       const ind = c.industry_major || "不明";
       industryCounts[ind] = (industryCounts[ind] || 0) + 1;
    });
    const industries = Object.keys(industryCounts).sort((a, b) => industryCounts[b] - industryCounts[a]).slice(0, 10);
    const matrix = industries.map(ind => ({
      industry: ind,
      cells: bandsList.map(b => {
        const matching = data.filter(c => (c.industry_major || "不明") === ind && c.score >= b.min && c.score < b.max);
        const won = matching.filter(c => c.status === '成約').length;
        const total = matching.length;
        const rate = total > 0 ? won / total : null;
        return { label: b.label, rate, won, total };
      })
    }));
    return { matrix, bands: bandsList.map(b => b.label), industries };
  }, [data]);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload;
      return (
        <div className="bg-slate-900/90 text-white p-3 rounded-xl shadow-2xl border border-white/10 text-xs backdrop-blur-md">
          <p className="font-black text-indigo-300 mb-1">{d.industry_sub || d.industry_major}</p>
          <p className="flex justify-between gap-4">結果: <span className={d.status === '成約' ? 'text-emerald-400 font-black' : 'text-rose-400 font-black'}>{d.status}</span></p>
          <p className="flex justify-between gap-4">スコア: <span className="font-mono text-white/80">{d.score?.toFixed(1)}</span></p>
          <p className="flex justify-between gap-4">価格: <span className="font-mono text-white/80">{d.acquisition_cost?.toLocaleString()}</span></p>
        </div>
      );
    }
    return null;
  };

  if (loading) return (
    <div className="flex items-center justify-center min-h-[calc(100vh-2rem)]">
      <Activity className="w-12 h-12 text-indigo-500 animate-spin" />
    </div>
  );

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8 flex justify-between items-start">
        <div>
          <h1 className="text-4xl font-black text-slate-800 flex items-center gap-4">
            <Eye className="w-10 h-10 text-indigo-600" />
            ビジュアルインサイト
          </h1>
          <p className="text-slate-500 font-bold mt-2">成約・失注の境界線、業種ごとのフローを一目で把握します。</p>
        </div>
      </div>
      
      <div className="flex gap-2 mb-8 bg-slate-100 p-1.5 rounded-2xl w-fit shadow-inner">
        {[
          {id: 'bubble', label: '案件マップ', icon: ChartPie},
          {id: 'heatmap', label: '熱分布分析', icon: Grid},
          {id: 'sankey', label: '案件フロー', icon: GitMerge}
        ].map(tab => (
          <button 
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl font-black transition-all ${activeTab === tab.id ? 'bg-white text-indigo-600 shadow-md transform scale-[1.02]' : 'text-slate-500 hover:text-slate-700'}`}
          >
            <tab.icon className="w-5 h-5" /> {tab.label}
          </button>
        ))}
      </div>

      <div className="bg-white p-8 rounded-[2.5rem] shadow-xl border border-slate-200 relative overflow-hidden">
         {activeTab === 'bubble' && (
           <div className="animate-in fade-in duration-500">
             <div className="mb-6">
                <h3 className="text-2xl font-black text-slate-800">🫧 案件ポジショニング分析</h3>
                <p className="text-slate-400 text-sm font-bold">スコアと収益性の相関をプロット。現在の成功パターンを特定します。</p>
             </div>
             <div className="w-full h-[500px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis type="number" dataKey="score" name="スコア" domain={[0, 105]} stroke="#94a3b8" tick={{fill: '#64748b', fontWeight: 'bold'}} />
                    <YAxis type="number" dataKey="spread" name="スプレッド" stroke="#94a3b8" tick={{fill: '#64748b', fontWeight: 'bold'}} />
                    <ZAxis type="number" dataKey="acquisition_cost" range={[40, 800]} />
                    <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
                    <Legend iconType="circle" />
                    <Scatter name="成約" data={wonCases} fill="#10b981" fillOpacity={0.6} stroke="#059669" strokeWidth={2} />
                    <Scatter name="失注" data={lostCases} fill="#f43f5e" fillOpacity={0.6} stroke="#e11d48" strokeWidth={2} />
                  </ScatterChart>
                </ResponsiveContainer>
             </div>
           </div>
         )}

         {activeTab === 'heatmap' && (
           <div className="animate-in fade-in duration-500 overflow-x-auto">
             <div className="mb-6">
                <h3 className="text-2xl font-black text-slate-800">🌡️ 業種 × スコア 成約確率分布</h3>
                <p className="text-slate-400 text-sm font-bold">どの業種のどのスコア帯が「勝ち筋」かを色付けして特定します。</p>
             </div>
              <table className="w-full text-left border-collapse rounded-2xl overflow-hidden shadow-inner">
                <thead>
                  <tr className="bg-slate-50 text-slate-500 font-black text-xs uppercase tracking-widest">
                    <th className="p-5 border-b border-slate-100">業種区分</th>
                    {heatmapData.bands.map(b => <th key={b} className="p-5 text-center border-b border-slate-100">{b}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {heatmapData.matrix.map((row) => (
                    <tr key={row.industry} className="border-b border-slate-50 last:border-0 group">
                      <td className="p-5 font-black text-slate-700 bg-slate-50/30 group-hover:bg-slate-50 transition-colors">{row.industry}</td>
                      {row.cells.map((cell, i) => {
                        const h = (cell.rate ?? 0) * 120;
                        return (
                          <td key={i} className="p-5 text-center transition-all" style={cell.rate !== null ? {backgroundColor: `hsla(${h}, 70%, 60%, 0.1)`, border: `2px solid hsla(${h}, 70%, 60%, 0.2)`} : {}}>
                            {cell.rate !== null ? (
                              <div>
                                <div className="text-lg font-black" style={{color: `hsla(${h}, 70%, 40%, 1)`}}>{(cell.rate * 100).toFixed(0)}%</div>
                                <div className="text-[10px] font-bold text-slate-400">{cell.won}/{cell.total} <span className="opacity-50">cases</span></div>
                              </div>
                            ) : <span className="text-slate-300 italic text-xs">NO DATA</span>}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
           </div>
         )}

         {activeTab === 'sankey' && (
           <div className="animate-in fade-in duration-500">
             <div className="mb-6 flex justify-between items-end">
                <div>
                  <h3 className="text-2xl font-black text-slate-800">🌊 審査案件フロー・サンキー図</h3>
                  <p className="text-slate-400 text-sm font-bold">業種からスコア帯を経て最終結果へ。案件の流動性を可視化します。</p>
                </div>
                <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 p-2 px-4 rounded-xl text-[10px] font-black text-slate-400 uppercase tracking-widest">
                   <MousePointer2 className="w-3 h-3 text-emerald-400" /> Hover to Highlight
                </div>
             </div>
             <div className="w-full flex justify-center bg-slate-50/50 p-6 rounded-3xl border border-slate-100 shadow-inner">
               <svg ref={sankeyRef} className="w-full max-w-full" viewBox="0 0 900 500" preserveAspectRatio="xMidYMid meet" />
             </div>
           </div>
         )}
      </div>
    </div>
  );
}