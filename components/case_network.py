"""
案件類似ネットワーク - MiroFish風 D3.js フォースグラフ

過去案件をノードとして、類似度に応じてエッジで繋ぐ。
現在審査中の案件（セッションステートから取得）を金色ノードで表示し、
近い過去案件クラスターを視覚的に特定する。

類似度 = 業種一致(0.4) + スコア近接(0~0.4) + 競合一致(0.2)
エッジ閾値: 0.5 以上のペアのみ描画。各ノード最大5本。
"""
import json
import streamlit as st
import streamlit.components.v1 as components
from data_cases import load_past_cases


def _score_of(case: dict) -> float:
    return float(
        case.get("score")
        or (case.get("result") or {}).get("score")
        or 0
    )


def _similarity(a: dict, b: dict) -> float:
    """2案件間の類似度（0〜1）を計算する"""
    sim = 0.0
    # 業種（大分類）一致
    if a.get("industry_major") and a.get("industry_major") == b.get("industry_major"):
        sim += 0.4
    # スコア近接（差が0なら+0.4、差が50以上なら+0）
    sa, sb = _score_of(a), _score_of(b)
    if sa > 0 and sb > 0:
        sim += max(0.0, 1.0 - abs(sa - sb) / 50.0) * 0.4
    # 競合他社名一致
    comp_a = (a.get("competitor_name") or "").strip()
    comp_b = (b.get("competitor_name") or "").strip()
    if comp_a and comp_a == comp_b:
        sim += 0.2
    return round(sim, 3)


def build_network_data(current_case: dict | None = None) -> dict:
    """
    ノード・エッジデータを構築する。
    current_case: 現在審査中の案件 dict（Noneなら表示しない）
    """
    cases = load_past_cases()

    # 登録済み案件のみ（未登録はグレーで追加）
    nodes = []
    node_index: dict[str, int] = {}

    for c in cases:
        status = c.get("final_status", "未登録")
        score = _score_of(c)
        if score <= 0:
            continue

        cid = c.get("id", "")
        label = (c.get("industry_sub") or c.get("industry_major") or "不明")[:8]
        timestamp = (c.get("timestamp") or "")[:10]

        if status == "成約":
            color = "#3b82f6"      # 青
            shape = "circle"
        elif status == "失注":
            color = "#ef4444"      # 赤
            shape = "circle"
        else:
            color = "#94a3b8"      # グレー（未登録）
            shape = "circle"

        nodes.append({
            "id": cid,
            "label": f"{label}\n{score:.0f}pt",
            "industry_major": c.get("industry_major") or "不明",
            "industry_sub": c.get("industry_sub") or "不明",
            "score": score,
            "status": status,
            "color": color,
            "shape": shape,
            "radius": 8 + score / 100 * 10,
            "competitor_name": c.get("competitor_name") or "",
            "final_rate": float(c.get("final_rate") or 0),
            "timestamp": timestamp,
            "is_current": False,
        })
        node_index[cid] = len(nodes) - 1

    # 現在の案件ノードを追加
    if current_case:
        cur_score = _score_of(current_case)
        if cur_score > 0:
            nodes.append({
                "id": "__current__",
                "label": f"現在の案件\n{cur_score:.0f}pt",
                "industry_major": current_case.get("industry_major") or current_case.get("selected_major") or "不明",
                "industry_sub": current_case.get("industry_sub") or current_case.get("selected_sub") or "不明",
                "score": cur_score,
                "status": "審査中",
                "color": "#f59e0b",    # 金色
                "shape": "star",
                "radius": 14,
                "competitor_name": current_case.get("competitor") or "",
                "final_rate": 0.0,
                "timestamp": "現在",
                "is_current": True,
            })
            node_index["__current__"] = len(nodes) - 1

    # エッジ生成（類似度 >= 0.5、各ノード最大5本）
    edges = []
    edge_count: dict[str, int] = {}

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            sim = _similarity(a, b)
            if sim < 0.5:
                continue
            ai, bi = a["id"], b["id"]
            if edge_count.get(ai, 0) >= 5 and edge_count.get(bi, 0) >= 5:
                continue
            edges.append({
                "source": i,
                "target": j,
                "similarity": sim,
                "width": round(sim * 3, 1),
                "opacity": round(0.2 + sim * 0.6, 2),
            })
            edge_count[ai] = edge_count.get(ai, 0) + 1
            edge_count[bi] = edge_count.get(bi, 0) + 1

    # 統計
    total = len(nodes)
    won = sum(1 for n in nodes if n["status"] == "成約")
    lost = sum(1 for n in nodes if n["status"] == "失注")

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {"total": total, "won": won, "lost": lost},
    }


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
    line-height: 1.6; max-width: 200px;
  }
  .legend { position: absolute; bottom: 12px; left: 12px; color: #94a3b8; font-size: 11px; }
  .legend-item { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .temp-badge {
    position: absolute; top: 12px; right: 12px;
    background: rgba(15,23,42,0.8); border: 1px solid #334155;
    border-radius: 6px; padding: 6px 10px; color: #94a3b8; font-size: 11px;
  }
</style>
</head>
<body>
<div class="tooltip" id="tooltip"></div>
<div class="temp-badge" id="stats">読み込み中...</div>
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>現在の案件</div>
  <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>成約</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>失注</div>
  <div class="legend-item"><div class="legend-dot" style="background:#94a3b8"></div>未登録</div>
  <div style="margin-top:6px; color:#64748b">線の太さ = 類似度</div>
</div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const GRAPH_DATA = __GRAPH_DATA__;
const W = window.innerWidth, H = __HEIGHT__;

const svg = d3.select("body").append("svg")
  .attr("width", W).attr("height", H)
  .call(d3.zoom().scaleExtent([0.3, 4]).on("zoom", e => g.attr("transform", e.transform)));

const g = svg.append("g");

// 統計バッジ
const s = GRAPH_DATA.summary;
document.getElementById("stats").textContent =
  `案件数: ${s.total}  成約: ${s.won}  失注: ${s.lost}`;

// シミュレーション
const sim = d3.forceSimulation(GRAPH_DATA.nodes)
  .force("link", d3.forceLink(GRAPH_DATA.edges).distance(d => 120 - d.similarity * 60).strength(0.6))
  .force("charge", d3.forceManyBody().strength(-180))
  .force("center", d3.forceCenter(W / 2, H / 2))
  .force("collide", d3.forceCollide(d => d.radius + 6));

// エッジ
const link = g.append("g").selectAll("line")
  .data(GRAPH_DATA.edges).enter().append("line")
  .attr("stroke", "#475569")
  .attr("stroke-width", d => d.width)
  .attr("stroke-opacity", d => d.opacity);

// ノード
const node = g.append("g").selectAll("g")
  .data(GRAPH_DATA.nodes).enter().append("g")
  .call(d3.drag()
    .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
  );

// 現在の案件は星形、それ以外は円
node.each(function(d) {
  const el = d3.select(this);
  if (d.is_current) {
    el.append("polygon")
      .attr("points", starPoints(0, 0, d.radius, d.radius * 0.45, 5))
      .attr("fill", d.color)
      .attr("stroke", "#fff")
      .attr("stroke-width", 2);
  } else {
    el.append("circle")
      .attr("r", d.radius)
      .attr("fill", d.color)
      .attr("stroke", d.is_current ? "#fff" : "#1e293b")
      .attr("stroke-width", 1.5)
      .attr("opacity", 0.9);
  }
});

// ラベル
node.append("text")
  .attr("text-anchor", "middle")
  .attr("dy", d => d.radius + 12)
  .attr("fill", "#cbd5e1")
  .attr("font-size", "9px")
  .text(d => d.industry_sub.slice(0, 6));

// ツールチップ
const tooltip = document.getElementById("tooltip");
node
  .on("mouseover", (e, d) => {
    const rate = d.final_rate > 0 ? d.final_rate.toFixed(2) + "%" : "—";
    tooltip.innerHTML =
      `<b>${d.industry_sub}</b><br>` +
      `スコア: ${d.score.toFixed(0)}<br>` +
      `状態: ${d.status}<br>` +
      `獲得金利: ${rate}<br>` +
      `競合: ${d.competitor_name || "なし"}<br>` +
      `${d.timestamp}`;
    tooltip.style.display = "block";
    // 接続ノードを強調
    link.attr("stroke-opacity", l =>
      l.source.id === d.id || l.target.id === d.id ? 0.9 : 0.05);
    node.selectAll("circle, polygon").attr("opacity", n =>
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
    node.selectAll("circle, polygon").attr("opacity", 0.9);
  });

// tick
sim.on("tick", () => {
  link
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  node.attr("transform", d => `translate(${d.x},${d.y})`);
});

// 星形ポイント生成
function starPoints(cx, cy, outerR, innerR, points) {
  let str = "";
  for (let i = 0; i < points * 2; i++) {
    const r = i % 2 === 0 ? outerR : innerR;
    const angle = (Math.PI / points) * i - Math.PI / 2;
    str += (cx + r * Math.cos(angle)) + "," + (cy + r * Math.sin(angle)) + " ";
  }
  return str.trim();
}
</script>
</body>
</html>
"""


def render_case_network():
    """案件類似ネットワークを描画する（サイドバーメニューから呼ぶ）"""
    st.title("🔗 案件類似ネットワーク")
    st.caption("過去案件の類似度をネットワークで可視化します。線が太いほど似ている案件です。ノードをホバーすると詳細が表示されます。")

    # 現在審査中の案件をセッションから取得
    current_case = None
    last_res = st.session_state.get("last_result")
    last_inputs = st.session_state.get("last_submitted_inputs")
    if last_res and last_inputs:
        current_case = {**last_res, **last_inputs}

    if current_case:
        st.info("💡 金色の星ノードが現在審査中の案件です。近くにある過去案件ほど類似しています。")

    graph_data = build_network_data(current_case)
    summary = graph_data["summary"]

    # サマリーメトリクス
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総案件数", f"{summary['total']}件")
    c2.metric("成約", f"{summary['won']}件", delta=None)
    c3.metric("失注", f"{summary['lost']}件", delta=None)
    c4.metric("エッジ数", f"{len(graph_data['edges'])}本")

    if summary["total"] < 2:
        st.warning("案件が2件以上登録されるとネットワークが表示されます。")
        return

    height = 560
    html = _D3_TEMPLATE \
        .replace("__GRAPH_DATA__", json.dumps(graph_data, ensure_ascii=False)) \
        .replace("__HEIGHT__", str(height))

    components.html(html, height=height + 20, scrolling=False)

    with st.expander("📋 類似ペア一覧", expanded=False):
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]
        if edges:
            rows = []
            for e in sorted(edges, key=lambda x: -x["similarity"])[:20]:
                na = nodes[e["source"]]
                nb = nodes[e["target"]]
                rows.append({
                    "案件A": f"{na['industry_sub']} ({na['score']:.0f}pt/{na['status']})",
                    "案件B": f"{nb['industry_sub']} ({nb['score']:.0f}pt/{nb['status']})",
                    "類似度": f"{e['similarity']:.2f}",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("類似度0.5以上のペアがありません。")
