"use client";
import React, { useCallback, useEffect, useState } from 'react';
import { apiClient } from '@/lib/api';
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
    const initialSpread = CONFIG?.initialSpread ?? 1;

    // Clear previous
    d3.select("#root").selectAll("svg").remove();

    const zoom = d3.zoom().scaleExtent([0.25, zoomMax]).on("zoom", e => g.attr("transform", e.transform));
    const svg = d3.select("#root").append("svg")
    .attr("width", W).attr("height", H)
    .call(zoom);

    const g = svg.append("g");

    GRAPH_DATA.nodes.forEach((node, index) => {
      if (typeof node.x === "number" && typeof node.y === "number") return;
      const angle = (index / Math.max(1, GRAPH_DATA.nodes.length)) * Math.PI * 2;
      const ring = Math.sqrt(index + 1) * 15 * initialSpread;
      node.x = W / 2 + Math.cos(angle) * ring;
      node.y = H / 2 + Math.sin(angle) * ring;
    });

    // シミュレーション
    const sim = d3.forceSimulation(GRAPH_DATA.nodes)
    .force("link", d3.forceLink(GRAPH_DATA.edges).id(d => d.id).distance(d => Math.max(35, linkBaseDistance - d.similarity * 60)).strength(0.8))
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
    const _esc = s => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    const tooltip = document.getElementById("tooltip");
    node
    .on("mouseover", (e, d) => {
        const rate = d.final_rate > 0 ? d.final_rate.toFixed(2) + "%" : "—";
        tooltip.innerHTML =
        '<b>' + _esc(d.industry_sub) + '</b><br>' +
        'スコア: ' + _esc(d.score.toFixed(0)) + '<br>' +
        '状態: ' + _esc(d.status) + '<br>' +
        '獲得金利: ' + _esc(rate) + '<br>' +
        '競合: ' + _esc(d.competitor_name || "なし") + '<br>' +
        (d.timestamp ? _esc(d.timestamp) : '');
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

type SimilarGraphSettings = {
  chargeStrength: number;
  linkBaseDistance: number;
  collisionPadding: number;
  nodeScale: number;
  zoomMax: number;
  initialSpread: number;
  fitOnLoad: boolean;
};

type SimilarGraphData = {
  nodes?: Array<{
    id?: string;
    radius?: number;
    x?: number;
    y?: number;
    industry_sub?: string;
    score?: number;
    status?: string;
    final_rate?: number;
    competitor_name?: string;
    timestamp?: string;
  }>;
  edges?: Array<{ source: number | string; target: number | string; similarity?: number }>;
  summary?: {
    total?: number;
    won?: number;
    lost?: number;
  };
};

const SIMILAR_SETTINGS_KEY = "similar-network-d3-settings";
const DEFAULT_SIMILAR_SETTINGS: SimilarGraphSettings = {
  chargeStrength: -180,
  linkBaseDistance: 102,
  collisionPadding: 10,
  nodeScale: 1,
  zoomMax: 3.5,
  initialSpread: 1,
  fitOnLoad: true,
};

const getRecommendedSettings = (data?: SimilarGraphData | null): SimilarGraphSettings => {
  const total = data?.summary?.total ?? data?.nodes?.length ?? 0;
  const edgeCount = data?.edges?.length ?? 0;
  const density = total > 0 ? edgeCount / total : 0;

  if (total <= 25) {
    return {
      chargeStrength: -115,
      linkBaseDistance: 88,
      collisionPadding: 8,
      nodeScale: 1.12,
      zoomMax: 4.5,
      initialSpread: 0.8,
      fitOnLoad: true,
    };
  }

  if (total <= 90 && density <= 3.2) {
    return DEFAULT_SIMILAR_SETTINGS;
  }

  if (total <= 280) {
    return {
      chargeStrength: density > 4 ? -260 : -220,
      linkBaseDistance: density > 4 ? 128 : 116,
      collisionPadding: 13,
      nodeScale: 0.94,
      zoomMax: 3.2,
      initialSpread: 1.16,
      fitOnLoad: true,
    };
  }

  return {
    chargeStrength: density > 4 ? -360 : -300,
    linkBaseDistance: density > 4 ? 150 : 136,
    collisionPadding: 16,
    nodeScale: 0.82,
    zoomMax: 2.8,
    initialSpread: 1.36,
    fitOnLoad: true,
  };
};

const normalizeGraphData = (data: SimilarGraphData | null | undefined): SimilarGraphData | null => {
  if (!data || !Array.isArray(data.nodes)) return data ?? null;
  const nodes = data.nodes
    .filter((node) => node && node.id !== undefined && node.id !== null)
    .map((node, index) => ({
      ...node,
      id: String(node.id || `case-${index + 1}`),
      score: typeof node.score === "number" ? node.score : 0,
      final_rate: typeof node.final_rate === "number" ? node.final_rate : 0,
      radius: typeof node.radius === "number" ? node.radius : 10,
    }));
  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = (Array.isArray(data.edges) ? data.edges : [])
    .map((edge) => {
      const source =
        typeof edge.source === "number" ? nodes[edge.source]?.id : String(edge.source ?? "");
      const target =
        typeof edge.target === "number" ? nodes[edge.target]?.id : String(edge.target ?? "");
      return {
        ...edge,
        source,
        target,
        similarity: typeof edge.similarity === "number" ? edge.similarity : 0,
      };
    })
    .filter((edge) => nodeIds.has(String(edge.source)) && nodeIds.has(String(edge.target)));
  return {
    ...data,
    nodes,
    edges,
    summary: {
      total: data.summary?.total ?? nodes.length,
      won: data.summary?.won ?? nodes.filter((node) => node.status === "成約").length,
      lost: data.summary?.lost ?? nodes.filter((node) => node.status === "失注").length,
    },
  };
};

export default function SimilarPage() {
  const [graphData, setGraphData] = useState<SimilarGraphData | null>(null);
  const [summary, setSummary] = useState<SimilarGraphData["summary"] | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [chargeStrength, setChargeStrength] = useState(DEFAULT_SIMILAR_SETTINGS.chargeStrength);
  const [linkBaseDistance, setLinkBaseDistance] = useState(DEFAULT_SIMILAR_SETTINGS.linkBaseDistance);
  const [collisionPadding, setCollisionPadding] = useState(DEFAULT_SIMILAR_SETTINGS.collisionPadding);
  const [nodeScale, setNodeScale] = useState(DEFAULT_SIMILAR_SETTINGS.nodeScale);
  const [zoomMax, setZoomMax] = useState(DEFAULT_SIMILAR_SETTINGS.zoomMax);
  const [initialSpread, setInitialSpread] = useState(DEFAULT_SIMILAR_SETTINGS.initialSpread);
  const [fitOnLoad, setFitOnLoad] = useState(DEFAULT_SIMILAR_SETTINGS.fitOnLoad);
  const [viewKey, setViewKey] = useState(0);
  const [hydrated, setHydrated] = useState(false);
  const [usingSavedSettings, setUsingSavedSettings] = useState(false);

  const applySettings = useCallback((settings: SimilarGraphSettings, saved = true) => {
    setChargeStrength(settings.chargeStrength);
    setLinkBaseDistance(settings.linkBaseDistance);
    setCollisionPadding(settings.collisionPadding);
    setNodeScale(settings.nodeScale);
    setZoomMax(settings.zoomMax);
    setInitialSpread(settings.initialSpread);
    setFitOnLoad(settings.fitOnLoad);
    setUsingSavedSettings(saved);
    setViewKey((v) => v + 1);
  }, []);

  useEffect(() => {
    triggerMebuki('guide', '案件類似ネットワーク画面ですね！\n過去の案件から似たパターンのものを可視化します！');

    let loadedFromStorage = false;
    try {
      const raw = window.localStorage.getItem(SIMILAR_SETTINGS_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as Partial<SimilarGraphSettings>;
        loadedFromStorage = true;
        setUsingSavedSettings(true);
        setChargeStrength(typeof parsed.chargeStrength === "number" ? parsed.chargeStrength : DEFAULT_SIMILAR_SETTINGS.chargeStrength);
        setLinkBaseDistance(typeof parsed.linkBaseDistance === "number" ? parsed.linkBaseDistance : DEFAULT_SIMILAR_SETTINGS.linkBaseDistance);
        setCollisionPadding(typeof parsed.collisionPadding === "number" ? parsed.collisionPadding : DEFAULT_SIMILAR_SETTINGS.collisionPadding);
        setNodeScale(typeof parsed.nodeScale === "number" ? parsed.nodeScale : DEFAULT_SIMILAR_SETTINGS.nodeScale);
        setZoomMax(typeof parsed.zoomMax === "number" ? parsed.zoomMax : DEFAULT_SIMILAR_SETTINGS.zoomMax);
        setInitialSpread(typeof parsed.initialSpread === "number" ? parsed.initialSpread : DEFAULT_SIMILAR_SETTINGS.initialSpread);
        setFitOnLoad(typeof parsed.fitOnLoad === "boolean" ? parsed.fitOnLoad : DEFAULT_SIMILAR_SETTINGS.fitOnLoad);
      }
    } catch {
      // ignore
    } finally {
      setHydrated(true);
    }
    
    const fetchData = async () => {
      try {
        const res = await apiClient.get<SimilarGraphData>(`/api/similar/data`);
        const normalized = normalizeGraphData(res.data);
        setGraphData(normalized);
        setSummary(normalized?.summary || null);
        setLoadError(null);
        if (!loadedFromStorage) {
          applySettings(getRecommendedSettings(normalized), false);
        }
      } catch (err) {
        console.error("Failed to load similar network data", err);
        setLoadError("案件類似ネットワークのデータ取得に失敗しました。API /api/similar/data と過去案件DBを確認してください。");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [applySettings]);

  useEffect(() => {
    if (!hydrated || !usingSavedSettings) return;
    try {
      const payload: SimilarGraphSettings = {
        chargeStrength,
        linkBaseDistance,
        collisionPadding,
        nodeScale,
        zoomMax,
        initialSpread,
        fitOnLoad,
      };
      window.localStorage.setItem(SIMILAR_SETTINGS_KEY, JSON.stringify(payload));
    } catch {
      // ignore
    }
  }, [hydrated, usingSavedSettings, chargeStrength, linkBaseDistance, collisionPadding, nodeScale, zoomMax, initialSpread, fitOnLoad]);

  const updateSetting = <K extends keyof SimilarGraphSettings>(key: K, value: SimilarGraphSettings[K]) => {
    setUsingSavedSettings(true);
    if (key === "chargeStrength") setChargeStrength(value as number);
    if (key === "linkBaseDistance") setLinkBaseDistance(value as number);
    if (key === "collisionPadding") setCollisionPadding(value as number);
    if (key === "nodeScale") setNodeScale(value as number);
    if (key === "zoomMax") setZoomMax(value as number);
    if (key === "initialSpread") setInitialSpread(value as number);
    if (key === "fitOnLoad") setFitOnLoad(value as boolean);
  };

  const resetSettings = () => {
    applySettings(getRecommendedSettings(graphData), false);
    try {
      window.localStorage.removeItem(SIMILAR_SETTINGS_KEY);
    } catch {
      // ignore
    }
  };

  const settingsLabel = usingSavedSettings ? "カスタム保存中" : "データ量に応じた推奨値";
  const settingsLabelClass = usingSavedSettings
    ? "bg-sky-50 text-sky-700 border-sky-200"
    : "bg-emerald-50 text-emerald-700 border-emerald-200";

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
        <h1 className="flex flex-wrap items-center gap-3 text-2xl sm:text-3xl font-black leading-tight text-slate-800">
          <Network className="w-8 h-8 shrink-0 text-teal-600" />
          <span className="inline-flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className="whitespace-nowrap">案件類似ネットワーク</span>
            <span className="whitespace-nowrap text-xl sm:text-2xl text-slate-500">(D3.js)</span>
          </span>
        </h1>
        <p className="text-slate-500 font-bold mt-2">過去案件をノードとし、類似度（業種・スコア・競合）に応じてエッジで繋ぎます。</p>
      </div>

      <div className="mb-6 bg-white border border-slate-200 rounded-2xl p-4 shadow-sm">
        <div className="flex flex-col gap-3 mb-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2 text-slate-700 font-bold">
            <SlidersHorizontal className="w-4 h-4 text-teal-600" />
            調整画面
          </div>
          <div className={`inline-flex w-fit items-center rounded-full border px-3 py-1 text-xs font-bold ${settingsLabelClass}`}>
            {settingsLabel}
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          <label className="space-y-2">
            <div className="flex justify-between text-sm text-slate-600">
              <span>反発の強さ</span>
              <span className="font-bold text-slate-800">{chargeStrength}</span>
            </div>
            <input type="range" min={-600} max={-40} step={10} value={chargeStrength} onChange={(e) => updateSetting("chargeStrength", Number(e.target.value))} className="w-full" />
          </label>
          <label className="space-y-2">
            <div className="flex justify-between text-sm text-slate-600">
              <span>リンク距離</span>
              <span className="font-bold text-slate-800">{linkBaseDistance}</span>
            </div>
            <input type="range" min={50} max={220} step={5} value={linkBaseDistance} onChange={(e) => updateSetting("linkBaseDistance", Number(e.target.value))} className="w-full" />
          </label>
          <label className="space-y-2">
            <div className="flex justify-between text-sm text-slate-600">
              <span>衝突余白</span>
              <span className="font-bold text-slate-800">{collisionPadding}</span>
            </div>
            <input type="range" min={0} max={40} step={1} value={collisionPadding} onChange={(e) => updateSetting("collisionPadding", Number(e.target.value))} className="w-full" />
          </label>
          <label className="space-y-2">
            <div className="flex justify-between text-sm text-slate-600">
              <span>ノードサイズ</span>
              <span className="font-bold text-slate-800">{nodeScale.toFixed(2)}x</span>
            </div>
            <input type="range" min={0.7} max={1.6} step={0.05} value={nodeScale} onChange={(e) => updateSetting("nodeScale", Number(e.target.value))} className="w-full" />
          </label>
          <label className="space-y-2">
            <div className="flex justify-between text-sm text-slate-600">
              <span>ズーム上限</span>
              <span className="font-bold text-slate-800">{zoomMax.toFixed(1)}x</span>
            </div>
            <input type="range" min={1.5} max={8} step={0.5} value={zoomMax} onChange={(e) => updateSetting("zoomMax", Number(e.target.value))} className="w-full" />
          </label>
          <label className="space-y-2">
            <div className="flex justify-between text-sm text-slate-600">
              <span>初期広がり</span>
              <span className="font-bold text-slate-800">{initialSpread.toFixed(2)}x</span>
            </div>
            <input type="range" min={0.5} max={2.2} step={0.05} value={initialSpread} onChange={(e) => updateSetting("initialSpread", Number(e.target.value))} className="w-full" />
          </label>
          <label className="flex items-center gap-3 pt-7 text-sm font-bold text-slate-700">
            <input type="checkbox" checked={fitOnLoad} onChange={(e) => updateSetting("fitOnLoad", e.target.checked)} className="w-4 h-4 rounded border-slate-300" />
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
          <button
            type="button"
            onClick={resetSettings}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-slate-200 bg-white text-slate-700 font-bold shadow-sm hover:bg-slate-50 transition-colors"
          >
            推奨値に戻す
          </button>
          <button
            type="button"
            onClick={() => applySettings(DEFAULT_SIMILAR_SETTINGS, true)}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-slate-200 bg-white text-slate-700 font-bold shadow-sm hover:bg-slate-50 transition-colors"
          >
            標準プリセット
          </button>
          <div className="text-xs text-slate-500">広がりすぎるときは反発を弱め、リンク距離を短くすると詰まります。</div>
        </div>
      </div>

      {loadError ? (
        <div className="bg-rose-50 border border-rose-200 p-6 rounded-2xl flex items-start gap-4">
          <Activity className="w-8 h-8 text-rose-500 shrink-0" />
          <div>
            <h3 className="font-bold text-rose-800 text-lg">案件データを取得できません</h3>
            <p className="text-rose-700 mt-1">{loadError}</p>
          </div>
        </div>
      ) : !graphData ? (
        <div className="bg-amber-50 border border-amber-200 p-6 rounded-2xl flex items-start gap-4">
          <Activity className="w-8 h-8 text-amber-500 shrink-0" />
          <div>
            <h3 className="font-bold text-amber-800 text-lg">案件データが読み込まれていません</h3>
            <p className="text-amber-700 mt-1">APIからネットワーク用データが返っていないため、表示できる案件情報がありません。</p>
          </div>
        </div>
      ) : summary && (summary.total ?? 0) < 2 ? (
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
                  initialSpread,
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
