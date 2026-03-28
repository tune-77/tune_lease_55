"""
リース審査 競合関係グラフビジュアライザ
D3.js force-directed graph を Streamlit に埋め込む

ノード:
  - 業種ノード（円、色=成約率、大きさ=案件数）
  - 競合ノード（ひし形、グレー）
  - 案件ノード（小円、色=成約/失注）

エッジ:
  - 業種 ↔ 競合: 競合が発生した関係（太さ=回数）
  - 業種 ↔ 案件: 所属関係
"""
import json
import streamlit as st
import streamlit.components.v1 as components

from data_cases import load_past_cases


# 業種カラーパレット（成約率に応じてグラデーション）
def _win_rate_color(win_rate: float) -> str:
    """成約率0〜1 → 色コード（赤→黄→緑）"""
    if win_rate >= 0.75:
        return "#22c55e"   # 緑
    elif win_rate >= 0.5:
        return "#f59e0b"   # 黄
    else:
        return "#ef4444"   # 赤


def build_graph_data() -> dict:
    """past_casesからグラフ用のノード・エッジデータを構築する"""
    cases = load_past_cases()

    industry_stats: dict[str, dict] = {}   # industry -> {total, won, cases}
    competitor_stats: dict[str, dict] = {} # competitor -> {total, won}
    industry_competitor_edges: dict[tuple, int] = {}  # (industry, competitor) -> count
    case_nodes: list[dict] = []

    for c in cases:
        industry = c.get("industry_major") or "不明"
        status = c.get("final_status", "")
        competitor = c.get("competitor_name", "") or ""
        has_competitor = c.get("competitor", "") == "競合あり"
        score = float(c.get("score") or (c.get("result") or {}).get("score") or 0)
        final_rate = float(c.get("final_rate") or 0)
        case_id = c.get("id", "")

        # 業種集計
        if industry not in industry_stats:
            industry_stats[industry] = {"total": 0, "won": 0, "cases": []}
        industry_stats[industry]["total"] += 1
        if status == "成約":
            industry_stats[industry]["won"] += 1
        industry_stats[industry]["cases"].append(case_id)

        # 案件ノード
        if status in ("成約", "失注"):
            case_nodes.append({
                "id": f"case_{case_id}",
                "type": "case",
                "status": status,
                "industry": industry,
                "score": score,
                "rate": final_rate,
                "competitor": competitor,
            })

        # 競合集計（競合名が記録されているもののみ）
        if has_competitor and competitor:
            if competitor not in competitor_stats:
                competitor_stats[competitor] = {"total": 0, "won": 0}
            competitor_stats[competitor]["total"] += 1
            if status == "成約":
                competitor_stats[competitor]["won"] += 1

            key = (industry, competitor)
            industry_competitor_edges[key] = industry_competitor_edges.get(key, 0) + 1

    # ── ノードリスト構築 ───────────────────────────────────────────────
    nodes = []
    node_ids = set()

    # 業種ノード
    for ind, stats in industry_stats.items():
        win_rate = stats["won"] / stats["total"] if stats["total"] > 0 else 0
        node_id = f"ind_{ind}"
        nodes.append({
            "id": node_id,
            "label": ind.split(" ", 1)[-1] if " " in ind else ind,  # "D 建設業" → "建設業"
            "type": "industry",
            "total": stats["total"],
            "won": stats["won"],
            "win_rate": round(win_rate, 3),
            "color": _win_rate_color(win_rate),
            "radius": max(20, min(50, stats["total"] * 6)),
        })
        node_ids.add(node_id)

    # 競合ノード
    for comp, stats in competitor_stats.items():
        node_id = f"comp_{comp}"
        nodes.append({
            "id": node_id,
            "label": comp,
            "type": "competitor",
            "total": stats["total"],
            "won": stats["won"],
            "win_rate": round(stats["won"] / stats["total"], 3) if stats["total"] > 0 else 0,
            "color": "#94a3b8",
            "radius": max(15, min(35, stats["total"] * 5)),
        })
        node_ids.add(node_id)

    # 案件ノード（上位20件に絞る）
    for cn in case_nodes[:20]:
        node_id = cn["id"]
        nodes.append({
            "id": node_id,
            "label": f"{'✓' if cn['status']=='成約' else '✗'} {cn['rate']:.1f}%",
            "type": "case",
            "status": cn["status"],
            "score": cn["score"],
            "rate": cn["rate"],
            "color": "#3b82f6" if cn["status"] == "成約" else "#f87171",
            "radius": 8,
        })
        node_ids.add(node_id)

    # ── エッジリスト構築 ───────────────────────────────────────────────
    edges = []

    # 業種 ↔ 競合エッジ
    for (ind, comp), count in industry_competitor_edges.items():
        src = f"ind_{ind}"
        tgt = f"comp_{comp}"
        if src in node_ids and tgt in node_ids:
            edges.append({
                "source": src,
                "target": tgt,
                "type": "competed",
                "count": count,
                "width": max(1.5, min(6, count * 1.5)),
                "label": f"{count}回",
            })

    # 業種 ↔ 案件エッジ（上位20件）
    for cn in case_nodes[:20]:
        src = f"ind_{cn['industry']}"
        tgt = cn["id"]
        if src in node_ids and tgt in node_ids:
            edges.append({
                "source": src,
                "target": tgt,
                "type": "belongs",
                "count": 1,
                "width": 0.8,
                "label": "",
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "total_cases": sum(s["total"] for s in industry_stats.values()),
            "total_won": sum(s["won"] for s in industry_stats.values()),
            "industries": len(industry_stats),
            "competitors": len(competitor_stats),
        }
    }


_D3_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body { margin: 0; background: #f8fafc; font-family: sans-serif; overflow: hidden; }
  #graph { width: 100%; height: 520px; }
  .tooltip {
    position: absolute; background: rgba(15,23,42,0.92); color: #f1f5f9;
    padding: 8px 12px; border-radius: 8px; font-size: 12px; pointer-events: none;
    opacity: 0; transition: opacity 0.15s; max-width: 200px; line-height: 1.6;
  }
  .legend { position: absolute; bottom: 12px; left: 12px; font-size: 11px; color: #475569; }
  .legend-item { display: flex; align-items: center; gap: 5px; margin: 3px 0; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
  .controls { position: absolute; top: 10px; right: 12px; font-size: 11px; color: #94a3b8; }
</style>
</head>
<body>
<svg id="graph"></svg>
<div class="tooltip" id="tooltip"></div>
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div>業種（高成約率 75%+）</div>
  <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>業種（中成約率 50-75%）</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>業種（低成約率 50%未満）</div>
  <div class="legend-item"><div class="legend-dot" style="background:#94a3b8; border-radius:2px;"></div>競合他社</div>
  <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>成約案件</div>
  <div class="legend-item"><div class="legend-dot" style="background:#f87171"></div>失注案件</div>
</div>
<div class="controls">スクロール：ズーム　ドラッグ：移動</div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const graphData = __GRAPH_DATA__;

const width = document.getElementById('graph').clientWidth || 800;
const height = 520;
const svg = d3.select('#graph')
  .attr('width', width)
  .attr('height', height);

const g = svg.append('g');

// ズーム
svg.call(d3.zoom()
  .scaleExtent([0.3, 4])
  .on('zoom', e => g.attr('transform', e.transform))
);

// 矢印マーカー
svg.append('defs').selectAll('marker')
  .data(['competed', 'belongs'])
  .join('marker')
  .attr('id', d => `arrow-${d}`)
  .attr('viewBox', '0 -4 8 8')
  .attr('refX', 18)
  .attr('refY', 0)
  .attr('markerWidth', 6)
  .attr('markerHeight', 6)
  .attr('orient', 'auto')
  .append('path')
  .attr('d', 'M0,-4L8,0L0,4')
  .attr('fill', d => d === 'competed' ? '#64748b' : '#cbd5e1');

// フォースシミュレーション
const simulation = d3.forceSimulation(graphData.nodes)
  .force('link', d3.forceLink(graphData.edges)
    .id(d => d.id)
    .distance(d => d.type === 'belongs' ? 80 : 160)
    .strength(d => d.type === 'belongs' ? 0.3 : 0.6)
  )
  .force('charge', d3.forceManyBody().strength(d => d.type === 'industry' ? -400 : -150))
  .force('center', d3.forceCenter(width / 2, height / 2))
  .force('collide', d3.forceCollide().radius(d => (d.radius || 10) + 12))
  .force('x', d3.forceX(width / 2).strength(0.03))
  .force('y', d3.forceY(height / 2).strength(0.03));

// エッジ
const link = g.append('g').selectAll('line')
  .data(graphData.edges)
  .join('line')
  .attr('stroke', d => d.type === 'competed' ? '#64748b' : '#e2e8f0')
  .attr('stroke-width', d => d.width)
  .attr('stroke-dasharray', d => d.type === 'belongs' ? '3,3' : null)
  .attr('opacity', d => d.type === 'competed' ? 0.7 : 0.4)
  .attr('marker-end', d => d.type === 'competed' ? 'url(#arrow-competed)' : null);

// エッジラベル
const linkLabel = g.append('g').selectAll('text')
  .data(graphData.edges.filter(d => d.label && d.type === 'competed'))
  .join('text')
  .attr('text-anchor', 'middle')
  .attr('font-size', 9)
  .attr('fill', '#94a3b8')
  .text(d => d.label);

// ノードグループ
const node = g.append('g').selectAll('g')
  .data(graphData.nodes)
  .join('g')
  .attr('cursor', 'pointer')
  .call(d3.drag()
    .on('start', (e, d) => {
      if (!e.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    })
    .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on('end', (e, d) => {
      if (!e.active) simulation.alphaTarget(0);
      d.fx = null; d.fy = null;
    })
  );

// ノード形状
node.each(function(d) {
  const el = d3.select(this);
  if (d.type === 'competitor') {
    // ひし形
    const r = d.radius || 15;
    el.append('polygon')
      .attr('points', `0,${-r} ${r},0 0,${r} ${-r},0`)
      .attr('fill', d.color)
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);
  } else if (d.type === 'case') {
    el.append('circle')
      .attr('r', d.radius)
      .attr('fill', d.color)
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .attr('opacity', 0.85);
  } else {
    // 業種: 円
    el.append('circle')
      .attr('r', d.radius)
      .attr('fill', d.color)
      .attr('stroke', '#fff')
      .attr('stroke-width', 3)
      .attr('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.15))');
  }
});

// ノードラベル
node.filter(d => d.type !== 'case').append('text')
  .attr('text-anchor', 'middle')
  .attr('dy', d => (d.radius || 15) + 13)
  .attr('font-size', d => d.type === 'industry' ? 11 : 10)
  .attr('font-weight', d => d.type === 'industry' ? 'bold' : 'normal')
  .attr('fill', '#1e293b')
  .text(d => d.label);

// 業種ノードに成約率表示
node.filter(d => d.type === 'industry').append('text')
  .attr('text-anchor', 'middle')
  .attr('dy', 4)
  .attr('font-size', 10)
  .attr('font-weight', 'bold')
  .attr('fill', '#fff')
  .text(d => `${Math.round(d.win_rate * 100)}%`);

// ツールチップ
const tooltip = d3.select('#tooltip');
node.on('mouseover', (e, d) => {
  let html = `<strong>${d.label}</strong><br>`;
  if (d.type === 'industry') {
    html += `案件数: ${d.total}件<br>成約: ${d.won}件<br>成約率: ${Math.round(d.win_rate*100)}%`;
  } else if (d.type === 'competitor') {
    html += `競合発生: ${d.total}回<br>弊社成約: ${d.won}回<br>競合に勝率: ${Math.round(d.win_rate*100)}%`;
  } else {
    html += `${d.status}<br>スコア: ${d.score.toFixed(1)}<br>金利: ${d.rate.toFixed(2)}%`;
  }
  tooltip.html(html)
    .style('left', (e.pageX + 12) + 'px')
    .style('top', (e.pageY - 28) + 'px')
    .style('opacity', 1);
})
.on('mousemove', e => {
  tooltip.style('left', (e.pageX + 12) + 'px').style('top', (e.pageY - 28) + 'px');
})
.on('mouseout', () => tooltip.style('opacity', 0));

// ホバーエフェクト
node.on('mouseover.highlight', (e, d) => {
  node.selectAll('circle, polygon').attr('opacity', n =>
    n.id === d.id || graphData.edges.some(l =>
      (l.source.id || l.source) === d.id || (l.target.id || l.target) === d.id
    ) ? 1 : 0.3
  );
  link.attr('opacity', l =>
    (l.source.id || l.source) === d.id || (l.target.id || l.target) === d.id ? 1 : 0.1
  );
})
.on('mouseout.highlight', () => {
  node.selectAll('circle, polygon').attr('opacity', d => d.type === 'case' ? 0.85 : 1);
  link.attr('opacity', d => d.type === 'competed' ? 0.7 : 0.4);
});

// シミュレーションtick
simulation.on('tick', () => {
  link
    .attr('x1', d => d.source.x)
    .attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x)
    .attr('y2', d => d.target.y);

  linkLabel
    .attr('x', d => ((d.source.x || 0) + (d.target.x || 0)) / 2)
    .attr('y', d => ((d.source.y || 0) + (d.target.y || 0)) / 2 - 5);

  node.attr('transform', d => `translate(${d.x},${d.y})`);
});
</script>
</body>
</html>
"""


def render_graph_view():
    """競合関係グラフを表示する"""
    st.subheader("🕸️ 競合関係グラフ")
    st.caption("業種・競合他社・案件の関係を力学グラフで可視化。ノードにホバーで詳細表示。")

    data = build_graph_data()
    summary = data["summary"]

    # サマリーメトリクス
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("総案件数", f"{summary['total_cases']}件")
    with c2:
        won = summary['total_won']
        total = summary['total_cases']
        st.metric("成約数", f"{won}件", f"{won/total*100:.0f}%" if total > 0 else "")
    with c3:
        st.metric("業種数", f"{summary['industries']}業種")
    with c4:
        st.metric("競合他社", f"{summary['competitors']}社")

    # グラフレンダリング
    html = _D3_HTML.replace("__GRAPH_DATA__", json.dumps(data, ensure_ascii=False))
    components.html(html, height=560, scrolling=False)

    st.caption(
        "📌 **操作方法**: ノードをドラッグ・スクロールでズーム・ホバーで詳細 | "
        "大円=業種（色は成約率）、ひし形=競合他社、小円=個別案件"
    )
