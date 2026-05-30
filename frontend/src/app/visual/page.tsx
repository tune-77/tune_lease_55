"use client";
import React, { useEffect, useState, useMemo, useRef } from 'react';
import axios from 'axios';
import * as d3 from 'd3';
import { sankey as d3Sankey, sankeyLinkHorizontal } from 'd3-sankey';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Eye, Activity, ChartPie, Grid, GitMerge, MousePointer2, Lightbulb, ChevronDown, ChevronUp } from 'lucide-react';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

// REV-109: **bold** マーカーを JSX に変換
function formatInsight(text: string): React.ReactNode {
  const parts = text.split(/\*\*(.*?)\*\*/g);
  return parts.map((p, i) => i % 2 === 1 ? <strong key={i} className="text-slate-800">{p}</strong> : p);
}

export default function VisualPage() {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'bubble' | 'heatmap' | 'sankey'>('bubble');
  const [insightOpen, setInsightOpen] = useState(true);
  const sankeyRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    triggerMebuki('guide', 'ビジュアルインサイト画面ですね！\n過去の案件をマッピングして成約の傾向を探ります！');
    
    const fetchData = async () => {
      try {
        const res = await axios.get(`/api/visual/data`);
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

    const width = 640;
    const height = 300;
    const svg = d3.select(sankeyRef.current);

    svg.selectAll('*').remove();

    const generator = d3Sankey()
      .nodeWidth(14)
      .nodePadding(10)
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
      .attr('font-size', '9px')
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

  // REV-109: タブごとのデータ駆動インサイト自動生成
  const aiInsights = useMemo((): string[] => {
    if (!data.length) return [];

    const won = data.filter(c => c.status === '成約');
    const lost = data.filter(c => c.status === '失注');

    if (activeTab === 'bubble') {
      const avgWon = won.length ? won.reduce((s, c) => s + (c.score || 0), 0) / won.length : 0;
      const avgLost = lost.length ? lost.reduce((s, c) => s + (c.score || 0), 0) / lost.length : 0;
      const over70 = data.filter(c => c.score >= 70);
      const over70Rate = over70.length ? (over70.filter(c => c.status === '成約').length / over70.length * 100).toFixed(0) : null;
      const insights = [
        `成約案件の平均スコアは **${avgWon.toFixed(1)}pt**、失注は **${avgLost.toFixed(1)}pt**。差分 **${(avgWon - avgLost).toFixed(1)}pt** が現在の成否境界線です。`,
        over70Rate ? `スコア **70pt以上** の案件の成約率は **${over70Rate}%** — 高スコア案件の優先クロージングが成約率向上に直結します。` : null,
        `右上エリア（高スコア×高スプレッド）の緑クラスターが理想的な案件プロファイルです。この領域を狙う業種・物件選定を優先してください。`,
      ];
      return insights.filter(Boolean) as string[];
    }

    if (activeTab === 'heatmap') {
      const bands = [
        { min: 0, max: 50, label: '～50' },
        { min: 50, max: 65, label: '50〜65' },
        { min: 65, max: 80, label: '65〜80' },
        { min: 80, max: 200, label: '80〜' },
      ];
      const industryCounts: Record<string, { won: number; total: number }> = {};
      data.forEach(c => {
        const ind = c.industry_major || '不明';
        if (!industryCounts[ind]) industryCounts[ind] = { won: 0, total: 0 };
        industryCounts[ind].total++;
        if (c.status === '成約') industryCounts[ind].won++;
      });
      let hotInd = '', hotRate = 0, hotTotal = 0, hotWon = 0;
      Object.entries(industryCounts).forEach(([ind, cnt]) => {
        if (cnt.total >= 3 && cnt.won / cnt.total > hotRate) {
          hotRate = cnt.won / cnt.total; hotInd = ind; hotTotal = cnt.total; hotWon = cnt.won;
        }
      });
      let bestInd = '', bestBand = '', bestCellRate = 0;
      Object.keys(industryCounts).forEach(ind => {
        bands.forEach(b => {
          const matching = data.filter(c => (c.industry_major || '不明') === ind && c.score >= b.min && c.score < b.max);
          if (matching.length >= 2) {
            const r = matching.filter(c => c.status === '成約').length / matching.length;
            if (r > bestCellRate) { bestCellRate = r; bestInd = ind; bestBand = b.label; }
          }
        });
      });
      return [
        hotInd ? `**${hotInd}** 業種の成約率が **${(hotRate * 100).toFixed(0)}%** で最高（${hotTotal}件中 ${hotWon}件成約）。重点営業ターゲットです。` : '',
        bestInd ? `最パフォーマンスセルは **${bestInd} × ${bestBand}pt帯** で成約率 **${(bestCellRate * 100).toFixed(0)}%**。このゾーンの案件獲得を優先してください。` : '',
        `濃い緑のセルが「勝ち筋」です。赤・NO DATAのセルへのリソース配分を減らし、緑ゾーンに集中することで成約率の底上げが期待できます。`,
      ].filter(Boolean);
    }

    if (activeTab === 'sankey') {
      const total = data.length;
      const wonCount = won.length;
      const overallRate = (wonCount / total * 100).toFixed(1);
      const hi = data.filter(c => c.score >= 65);
      const hiRate = hi.length ? (hi.filter(c => c.status === '成約').length / hi.length * 100).toFixed(0) : '0';
      const mid = data.filter(c => c.score >= 50 && c.score < 65);
      const midRate = mid.length ? (mid.filter(c => c.status === '成約').length / mid.length * 100).toFixed(0) : '0';
      return [
        `全 **${total}件** のうち **${wonCount}件** が成約（全体成約率 **${overallRate}%**）。`,
        `スコア **65pt以上** の成約率は **${hiRate}%**、50〜65ptは **${midRate}%** — スコア帯が成否の最大分岐点です。`,
        `フローの太さが案件数を表します。細くなるフローの手前（業種・スコア帯）を特定し、そのステージでの改善施策を検討してください。`,
      ];
    }

    return [];
  }, [data, activeTab]);

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

      {/* REV-109: AI 状況説明コメントパネル */}
      {aiInsights.length > 0 && (
        <div className="mb-5 bg-indigo-50 border border-indigo-200 rounded-2xl overflow-hidden">
          <button
            onClick={() => setInsightOpen(o => !o)}
            className="w-full flex items-center gap-2.5 px-5 py-3 text-left hover:bg-indigo-100/60 transition-colors"
          >
            <Lightbulb className="w-4 h-4 text-indigo-500 flex-shrink-0" />
            <span className="font-black text-indigo-700 text-sm">AI 分析コメント</span>
            <span className="text-[10px] font-bold text-indigo-400 bg-indigo-100 px-2 py-0.5 rounded-full ml-1">データ自動解析</span>
            <span className="ml-auto text-indigo-400">
              {insightOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </span>
          </button>
          {insightOpen && (
            <div className="px-5 pb-4 space-y-2">
              {aiInsights.map((insight, i) => (
                <div key={i} className="flex items-start gap-2.5">
                  <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-indigo-400 flex-shrink-0" />
                  <p className="text-sm text-slate-600 leading-relaxed">{formatInsight(insight)}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

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
             <div className="mb-5 max-w-5xl">
                <h3 className="text-xl font-black text-slate-800 flex items-center gap-2">
                  <span className="inline-flex w-7 h-7 items-center justify-center rounded-lg bg-gradient-to-br from-fuchsia-500 to-amber-400 text-white shadow-sm">🌡️</span>
                  業種 × スコア 成約確率分布
                </h3>
                <p className="text-slate-500 text-xs font-semibold mt-1">勝ち筋の濃い帯だけを、見やすく濃い色で出しています。</p>
                <div className="mt-3 flex flex-wrap gap-2 text-[10px] font-bold text-slate-500">
                  <span className="px-2 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-100">高成約率</span>
                  <span className="px-2 py-1 rounded-full bg-amber-50 text-amber-700 border border-amber-100">中間帯</span>
                  <span className="px-2 py-1 rounded-full bg-rose-50 text-rose-700 border border-rose-100">低成約率</span>
                </div>
              </div>
              <div className="max-w-5xl">
              <table className="w-full text-left border-collapse rounded-2xl overflow-hidden shadow-inner border border-slate-200">
                <thead>
                  <tr className="bg-slate-900 text-slate-200 font-black text-[10px] uppercase tracking-[0.25em]">
                    <th className="p-3 border-b border-slate-800">業種区分</th>
                    {heatmapData.bands.map(b => <th key={b} className="p-3 text-center border-b border-slate-800">{b}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {heatmapData.matrix.map((row) => (
                    <tr key={row.industry} className="border-b border-slate-100 last:border-0 group">
                      <td className="p-3 font-black text-slate-700 bg-slate-50/70 group-hover:bg-slate-50 transition-colors text-sm whitespace-nowrap">
                        {row.industry}
                      </td>
                      {row.cells.map((cell, i) => {
                        const h = (cell.rate ?? 0) * 120;
                        const bg = cell.rate !== null
                          ? `linear-gradient(180deg, hsla(${h}, 85%, 58%, 0.95) 0%, hsla(${h}, 85%, 48%, 0.75) 100%)`
                          : 'linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%)';
                        return (
                          <td key={i} className="p-2 md:p-3 text-center transition-all" style={{ background: bg }}>
                            {cell.rate !== null ? (
                              <div className="flex flex-col items-center justify-center gap-1 min-h-[58px]">
                                <div className="text-lg md:text-xl font-black text-white drop-shadow-[0_1px_1px_rgba(0,0,0,0.35)]">
                                  {(cell.rate * 100).toFixed(0)}%
                                </div>
                                <div className="w-full max-w-[84px] h-1.5 rounded-full bg-white/25 overflow-hidden">
                                  <div
                                    className="h-full rounded-full bg-white/90"
                                    style={{ width: `${Math.max(12, Math.min(100, (cell.rate || 0) * 100))}%` }}
                                  />
                                </div>
                                <div className="text-[10px] font-bold text-white/85">
                                  {cell.won}/{cell.total} cases
                                </div>
                              </div>
                            ) : <span className="text-slate-400 italic text-[10px] font-bold">NO DATA</span>}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              </div>
           </div>
         )}

         {activeTab === 'sankey' && (
           <div className="animate-in fade-in duration-500">
             <div className="mb-4 flex justify-between items-end">
                <div>
                  <h3 className="text-lg font-black text-slate-800">🌊 審査案件フロー・サンキー図</h3>
                  <p className="text-slate-400 text-[11px] font-bold">業種からスコア帯を経て最終結果へ。</p>
                </div>
                <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 p-2 px-3 rounded-xl text-[9px] font-black text-slate-400 uppercase tracking-widest">
                   <MousePointer2 className="w-3 h-3 text-emerald-400" /> Hover to Highlight
                </div>
             </div>
             <div className="w-full flex justify-center bg-slate-50/50 p-3 rounded-3xl border border-slate-100 shadow-inner">
               <svg ref={sankeyRef} className="w-full max-w-full" viewBox="0 0 640 300" preserveAspectRatio="xMidYMid meet" />
             </div>
           </div>
         )}
      </div>
    </div>
  );
}
