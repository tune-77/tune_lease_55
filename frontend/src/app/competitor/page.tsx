"use client";
import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import * as d3 from 'd3';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Share2, Users, Activity, Target, Zap, MousePointer2 } from 'lucide-react';

export default function CompetitorPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    triggerMebuki('guide', '競合他社との勢力図ですね。\\n業種ごとにどの会社が強いか、力関係をネットワーク状に可視化して分析します！');
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const res = await axios.get("http://localhost:8000/api/analysis/competitor_graph");
      setData(res.data);
    } catch (err) {
      console.error(err);
      triggerMebuki('reject', 'データの取得に失敗しました。');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!data || !svgRef.current || !containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = 600;
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    svg.selectAll('*').remove();
    const g = svg.append('g');

    // Zoom
    svg.call(d3.zoom<SVGSVGElement, any>()
      .scaleExtent([0.2, 5])
      .on('zoom', (event: any) => g.attr('transform', event.transform))
    );

    // Arrow markers
    const defs = svg.append('defs');
    defs.selectAll('marker')
      .data(['competed', 'belongs'])
      .join('marker')
      .attr('id', d => `arrow-${d}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 22)
      .attr('refY', 0)
      .attr('markerWidth', 5)
      .attr('markerHeight', 5)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', d => d === 'competed' ? '#94a3b8' : '#cbd5e1');

    // Force Simulation
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.edges)
        .id((d: any) => d.id)
        .distance((d: any) => d.type === 'belongs' ? 90 : 180)
        .strength((d: any) => d.type === 'belongs' ? 0.4 : 0.7)
      )
      .force('charge', d3.forceManyBody().strength((d: any) => d.type === 'industry' ? -500 : -200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide().radius((d: any) => (d.radius || 15) + 15))
      .force('x', d3.forceX(width / 2).strength(0.05))
      .force('y', d3.forceY(height / 2).strength(0.05));

    // Links
    const link = g.append('g').selectAll('line')
      .data(data.edges)
      .join('line')
      .attr('stroke', (d: any) => d.type === 'competed' ? '#94a3b8' : '#cbd5e1')
      .attr('stroke-width', (d: any) => d.width || 1)
      .attr('stroke-dasharray', (d: any) => d.type === 'belongs' ? '4,4' : null)
      .attr('opacity', 0.6)
      .attr('marker-end', (d: any) => d.type === 'competed' ? 'url(#arrow-competed)' : null);

    // Nodes
    const node = g.append('g').selectAll('g')
      .data(data.nodes)
      .join('g')
      .attr('cursor', 'grab')
      .call(d3.drag<SVGGElement, any>()
        .on('start', (event: any, d: any) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on('drag', (event: any, d: any) => { d.fx = event.x; d.fy = event.y; })
        .on('end', (event: any, d: any) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null; d.fy = null;
        }) as any
      );

    // Node Shape
    node.each(function(d: any) {
      const el = d3.select(this);
      if (d.type === 'competitor') {
        const r = d.radius || 20;
        el.append('polygon')
          .attr('points', `0,${-r*1.2} ${r*1.2},0 0,${r*1.2} ${-r*1.2},0`)
          .attr('fill', '#f8fafc')
          .attr('stroke', '#cbd5e1')
          .attr('stroke-width', 3);
        el.append('text')
          .attr('text-anchor', 'middle')
          .attr('dy', 5)
          .attr('font-size', 9)
          .attr('font-weight', 'black')
          .attr('fill', '#64748b')
          .text('COMP');
      } else if (d.type === 'case') {
        el.append('circle')
          .attr('r', 8)
          .attr('fill', d.color)
          .attr('stroke', '#fff')
          .attr('stroke-width', 2);
      } else {
        el.append('circle')
          .attr('r', d.radius)
          .attr('fill', d.color)
          .attr('stroke', '#fff')
          .attr('stroke-width', 4)
          .attr('filter', 'drop-shadow(0 4px 6px rgba(0,0,0,0.1))');
      }
    });

    // Labels
    node.filter((d: any) => d.type !== 'case').append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', (d: any) => (d.radius || 20) + 20)
      .attr('font-size', 11)
      .attr('font-weight', 'black')
      .attr('fill', '#1e293b')
      .text((d: any) => d.label);

    // Win Rate in Industrial Nodes
    node.filter((d: any) => d.type === 'industry').append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', 4)
      .attr('font-size', 10)
      .attr('font-weight', 'black')
      .attr('fill', '#fff')
      .text((d: any) => `${Math.round(d.win_rate * 100)}%`);

    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => { simulation.stop(); };
  }, [data]);

  if (loading) return (
    <div className="p-8 flex items-center justify-center min-h-screen">
      <div className="flex flex-col items-center gap-4">
        <Activity className="w-12 h-12 text-orange-500 animate-spin" />
        <p className="text-slate-500 font-bold">競合ネットワークを構築中...</p>
      </div>
    </div>
  );

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8 flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
            <Share2 className="w-8 h-8 text-orange-500" />
            競合関係・勢力図分析
          </h1>
          <p className="text-slate-500 font-bold mt-2">業種ごとの成約率と、主要な競合他社との相関関係を可視化します。</p>
        </div>
        <div className="flex gap-4">
           <div className="bg-white border border-slate-200 px-6 py-4 rounded-3xl shadow-sm text-center">
              <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Industries</div>
              <div className="text-xl font-black text-slate-800">{data?.summary?.industries}</div>
           </div>
           <div className="bg-white border border-slate-200 px-6 py-4 rounded-3xl shadow-sm text-center">
              <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Competitors</div>
              <div className="text-xl font-black text-orange-600">{data?.summary?.competitors}</div>
           </div>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-[2.5rem] shadow-xl overflow-hidden relative" ref={containerRef}>
         <div className="absolute top-6 left-6 z-10 flex flex-col gap-2">
            <div className="flex items-center gap-2 bg-white/80 backdrop-blur p-2 rounded-xl border border-slate-200 shadow-sm text-[10px] font-black text-slate-500 uppercase">
               <div className="w-3 h-3 rounded-full bg-emerald-500"></div> 業種 (高成約)
            </div>
            <div className="flex items-center gap-2 bg-white/80 backdrop-blur p-2 rounded-xl border border-slate-200 shadow-sm text-[10px] font-black text-slate-500 uppercase">
               <div className="w-3 h-3 rounded-full bg-rose-500"></div> 業種 (低成約)
            </div>
            <div className="flex items-center gap-2 bg-white/80 backdrop-blur p-2 rounded-xl border border-slate-200 shadow-sm text-[10px] font-black text-slate-500 uppercase">
               <div className="w-3 h-3 bg-slate-200 rotate-45"></div> 競合他社
            </div>
         </div>
         
         <div className="absolute top-6 right-6 z-10">
            <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 p-3 rounded-2xl shadow-xl text-[10px] font-black text-slate-400 uppercase">
               <MousePointer2 className="w-3 h-3 text-emerald-400" />
               Drag to move | Scroll to zoom
            </div>
         </div>

         <svg ref={svgRef} className="w-full h-[600px] bg-slate-50/30" />
      </div>

      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
         <div className="bg-orange-50 border border-orange-100 p-8 rounded-3xl">
            <h4 className="text-orange-900 font-black mb-2 flex items-center gap-2">
               <Zap className="w-5 h-5" />
               ワンポイントアドバイス
            </h4>
            <p className="text-orange-800 text-sm font-bold leading-relaxed">
               中央に位置する競合ノードは、複数の業種で弊社と競り合っている「主要なライバル」です。
               特に赤い業種（低成約率）と繋がっている競合の提案内容は、重点的に調査することをお勧めします。
            </p>
         </div>
         <div className="bg-white border border-slate-200 p-8 rounded-3xl flex items-center justify-between">
            <div>
               <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Total Analysis Count</div>
               <div className="text-3xl font-black text-slate-800">{data?.summary?.total_cases} <span className="text-sm text-slate-400">cases</span></div>
            </div>
            <div className="text-right">
               <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Overall Win Rate</div>
               <div className="text-3xl font-black text-emerald-600">{Math.round((data?.summary?.total_won / data?.summary?.total_cases) * 100)}%</div>
            </div>
         </div>
      </div>
    </div>
  );
}