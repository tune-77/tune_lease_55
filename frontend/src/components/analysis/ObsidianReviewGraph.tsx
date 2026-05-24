"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import { Share2 } from "lucide-react";

type ObsidianGraphNode = {
  id: string;
  label: string;
  type: string;
  color: string;
  radius: number;
  used?: boolean;
  path?: string;
  snippet?: string;
  term?: string;
  pinned?: boolean;
};

type ObsidianGraphEdge = {
  source: string;
  target: string;
  type: string;
  width?: number;
  color?: string;
};

type ObsidianGraph = {
  nodes: ObsidianGraphNode[];
  edges: ObsidianGraphEdge[];
  summary?: {
    total_hits?: number;
    used_hits?: number;
    linked_nodes?: number;
    generated_terms?: number;
  };
  legend?: Array<{ label: string; color: string }>;
};

type Props = {
  graph?: ObsidianGraph | null;
  title?: string;
};

function truncate(text: string, limit = 26) {
  if (!text) return "";
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

export default function ObsidianReviewGraph({ graph, title = "Obsidianグラフ" }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    if (!wrapRef.current) return;
    const el = wrapRef.current;
    const update = () => setWidth(el.clientWidth || 0);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const prepared = useMemo(() => {
    if (!graph?.nodes?.length) return null;
    return {
      nodes: graph.nodes.map((node) => ({ ...node })),
      edges: graph.edges.map((edge) => ({ ...edge })),
    };
  }, [graph]);

  useEffect(() => {
    if (!prepared || !svgRef.current || !width) return;

    const height = 420;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("width", width).attr("height", height);

    const g = svg.append("g");
    const defs = svg.append("defs");
    defs.selectAll("marker")
      .data(["main", "link"])
      .join("marker")
      .attr("id", (d) => `arrow-${d}`)
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 16)
      .attr("refY", 0)
      .attr("markerWidth", 5)
      .attr("markerHeight", 5)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", (d) => (d === "main" ? "#94a3b8" : "#cbd5e1"));

    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.35, 4])
        .on("zoom", (event) => g.attr("transform", event.transform)),
    );

    const simulation = d3.forceSimulation(prepared.nodes as any)
      .force("link", d3.forceLink(prepared.edges as any)
        .id((d: any) => d.id)
        .distance((d: any) => (d.type === "wikilink" ? 92 : d.type === "query" ? 118 : 82))
        .strength((d: any) => (d.type === "wikilink" ? 0.18 : 0.45)))
      .force("charge", d3.forceManyBody().strength((d: any) => (d.type === "focus" ? -700 : d.type === "linked" ? -80 : -260)))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide().radius((d: any) => (d.radius || 12) + 14));

    const link = g.append("g")
      .attr("stroke-linecap", "round")
      .selectAll("line")
      .data(prepared.edges)
      .join("line")
      .attr("stroke", (d: any) => d.color || "#cbd5e1")
      .attr("stroke-width", (d: any) => d.width || 1)
      .attr("stroke-dasharray", (d: any) => (d.type === "wikilink" ? "4,4" : null))
      .attr("opacity", (d: any) => (d.type === "wikilink" ? 0.5 : 0.85))
      .attr("marker-end", (d: any) => (d.type === "wikilink" ? "url(#arrow-link)" : "url(#arrow-main)"));

    const node = g.append("g")
      .selectAll("g")
      .data(prepared.nodes)
      .join("g")
      .attr("cursor", "grab")
      .call(
        d3.drag<SVGGElement, any>()
          .on("start", (event: any, d: any) => {
            if (!event.active) simulation.alphaTarget(0.25).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event: any, d: any) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event: any, d: any) => {
            if (!event.active) simulation.alphaTarget(0);
            if (!d.pinned) {
              d.fx = null;
              d.fy = null;
            }
          }) as any,
      );

    node.append("circle")
      .attr("r", (d: any) => d.radius || 12)
      .attr("fill", (d: any) => d.color || "#94a3b8")
      .attr("stroke", (d: any) => (d.used ? "#0f172a" : "#e2e8f0"))
      .attr("stroke-width", (d: any) => (d.used || d.type === "focus" ? 3 : 1.5))
      .attr("opacity", (d: any) => (d.type === "linked" ? 0.7 : 1))
      .attr("filter", (d: any) => (d.used || d.type === "focus" ? "drop-shadow(0 4px 8px rgba(15,23,42,0.16))" : null));

    node.append("text")
      .text((d: any) => truncate(d.label, d.type === "linked" ? 18 : 26))
      .attr("text-anchor", "middle")
      .attr("dy", (d: any) => (d.radius || 12) + 14)
      .attr("font-size", (d: any) => (d.type === "focus" ? 12 : d.type === "linked" ? 9 : 10))
      .attr("font-weight", 900)
      .attr("fill", "#1e293b");

    node.append("title").text((d: any) => {
      const parts = [d.label];
      if (d.path) parts.push(d.path);
      if (d.snippet) parts.push(d.snippet);
      if (d.term) parts.push(`term: ${d.term}`);
      return parts.filter(Boolean).join("\n");
    });

    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);

      node.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => {
      simulation.stop();
    };
  }, [prepared, width]);

  const summary = graph?.summary;
  const legend = graph?.legend || [];

  return (
    <div ref={wrapRef} className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4 mb-4">
        <div>
          <h2 className="text-sm font-black text-slate-700 flex items-center gap-2">
            <Share2 className="w-4 h-4 text-cyan-600" />
            {title}
          </h2>
          <p className="mt-1 text-xs font-medium text-slate-500">
            今回の審査で使ったノートを色分けし、関連リンクまで一緒に見られるようにしています。
          </p>
        </div>
        {summary && (
          <div className="grid grid-cols-2 gap-2 text-right">
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
              <div className="text-[10px] font-black text-slate-400 uppercase">Hits</div>
              <div className="text-sm font-black text-slate-800">{summary.total_hits ?? 0}</div>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
              <div className="text-[10px] font-black text-slate-400 uppercase">Used</div>
              <div className="text-sm font-black text-emerald-600">{summary.used_hits ?? 0}</div>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-2 mb-3">
        {legend.map((item) => (
          <div key={item.label} className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] font-black text-slate-600">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
            {item.label}
          </div>
        ))}
      </div>

      <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-50">
        <svg ref={svgRef} className="block w-full" />
      </div>
    </div>
  );
}
