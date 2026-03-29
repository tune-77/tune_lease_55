"""
スコアリング結果 DAG（有向非巡回グラフ）ビジュアライザ
=========================================================
3モデルブレンド → 借手/物件スコア → 総合スコア → 最終判定
という因果連鎖を D3.js 固定5列レイアウトで可視化する。

列構成:
  Col 0 : 回帰モデル（全体・指標別・業種別）
  Col 1 : 補正因子（ai_completed_factors）＋定性項目
  Col 2 : スコア成分（借手・物件・定性スコア）
  Col 3 : 総合スコア
  Col 4 : 最終判定

グラフ理論的根拠:
  - ノード = スコアリング変数（入力・中間・出力）
  - 有向エッジ = 因果的寄与（スコアへの加算/減算）
  - エッジ幅  = |effect_percent| または ブレンド重み
  - ノード次数 = 何個の因子が集約されるかを示すハブ度
"""

import json
import streamlit as st
import streamlit.components.v1 as components


# ── DAGデータ構築 ──────────────────────────────────────────────────────────────

def build_dag_data(res: dict) -> dict:
    nodes: list[dict] = []
    edges: list[dict] = []

    def add_node(id_, label, col, color="#64748b", size=30, tooltip=""):
        nodes.append({"id": id_, "label": label, "col": col,
                      "color": color, "size": size, "tooltip": tooltip})

    def add_edge(src, tgt, width=2.0, color="#94a3b8", label=""):
        edges.append({"source": src, "target": tgt,
                      "width": width, "color": color, "label": label})

    # ── Col 4: 最終判定 ────────────────────────────────────────────────────
    hantei = res.get("hantei", "—")
    hantei_color = {
        "承認": "#22c55e", "条件付承認": "#84cc16",
        "要確認": "#f59e0b", "否決": "#ef4444",
        "強制承認": "#06b6d4", "強制否決": "#dc2626",
    }.get(hantei, "#6b7280")
    total_score = res.get("score", 0) or 0

    add_node("judgment", f"最終判定\n{hantei}", col=4,
             color=hantei_color, size=42,
             tooltip=f"最終判定: {hantei}\n総合スコア: {total_score:.1f}")

    # ── Col 3: 総合スコア ──────────────────────────────────────────────────
    add_node("total_score", f"総合スコア\n{total_score:.1f}", col=3,
             color="#3b82f6", size=36,
             tooltip=f"定量スコア × 重み + 定性スコア × 重み = {total_score:.1f}")
    add_edge("total_score", "judgment", width=4.0, color="#3b82f6")

    # ── Col 2: スコア成分（借手・物件・定性） ────────────────────────────
    borrower_score = res.get("score_borrower", 0) or 0
    asset_score    = res.get("asset_score", 0) or 0

    # res に重みが入っていない場合は get_score_weights() のデフォルト(85/15)を使う
    try:
        from data_cases import get_score_weights
        _w_b, _w_a, _, _ = get_score_weights()
    except Exception:
        _w_b, _w_a = 0.85, 0.15
    asset_weight   = res.get("asset_weight") or _w_a
    obligor_weight = res.get("obligor_weight") or _w_b

    add_node("borrower_score", f"借手スコア\n{borrower_score:.1f}%", col=2,
             color="#0ea5e9", size=34,
             tooltip=f"契約成立確率（成約期待度）: {borrower_score:.1f}%\n重み: {obligor_weight:.0%}")
    add_edge("borrower_score", "total_score",
             width=max(1.5, obligor_weight * 6), color="#0ea5e9",
             label=f"×{obligor_weight:.0%}")

    if asset_score > 0:
        add_node("asset_score", f"物件スコア\n{asset_score:.1f}", col=2,
                 color="#f97316", size=32,
                 tooltip=f"担保価値・流動性スコア: {asset_score:.1f}\n重み: {asset_weight:.0%}")
        add_edge("asset_score", "total_score",
                 width=max(1.5, asset_weight * 6), color="#f97316",
                 label=f"×{asset_weight:.0%}")

    # 定性スコア（あれば）
    qcorr = res.get("qualitative_scoring_correction")
    has_qual = bool(qcorr and qcorr.get("weighted_score"))
    if has_qual:
        qual_score = qcorr.get("weighted_score", 0)
        add_node("qual_score", f"定性スコア\n{qual_score}", col=2,
                 color="#8b5cf6", size=32,
                 tooltip=f"定性評価の加重平均スコア: {qual_score}/100")
        add_edge("qual_score", "total_score", width=2.5, color="#8b5cf6",
                 label="定性寄与")

    # ── Col 1: 補正因子 ＋ 定性スコアリング項目 ──────────────────────────
    ai_factors = res.get("ai_completed_factors") or []
    for i, f in enumerate(ai_factors):
        fid    = f"factor_{i}"
        effect = f.get("effect_percent", 0)
        fname  = f.get("factor", f"因子{i+1}")
        detail = f.get("detail", "")

        fcolor = ("#22c55e" if effect >= 5 else "#86efac") if effect >= 0 \
                 else ("#ef4444" if effect <= -5 else "#fca5a5")
        short  = fname if len(fname) <= 10 else fname[:10] + "…"

        add_node(fid, f"{short}\n{effect:+.0f}%", col=1,
                 color=fcolor, size=24,
                 tooltip=f"{fname}\n効果: {effect:+.0f}%\n{detail}")
        add_edge(fid, "borrower_score",
                 width=max(1.0, abs(effect) / 5), color=fcolor,
                 label=f"{effect:+.0f}%")

    if not ai_factors:
        add_node("factor_placeholder", "補正因子\nなし", col=1,
                 color="#334155", size=22, tooltip="補正因子が記録されていません")

    # 定性スコアリング個別項目
    if has_qual:
        for item_id, item_data in (qcorr.get("items") or {}).items():
            val = item_data.get("value")
            if val is None:
                continue
            label      = item_data.get("label", item_id)
            weight     = item_data.get("weight", 0)
            level_lbl  = item_data.get("level_label") or f"{int(val / 4 * 100)}点"
            score_pct  = val / 4 * 100
            icolor     = "#a78bfa" if score_pct >= 60 else ("#c4b5fd" if score_pct >= 40 else "#7c3aed")
            short      = label if len(label) <= 9 else label[:9] + "…"

            nid = f"qitem_{item_id}"
            add_node(nid, f"{short}\n{level_lbl}", col=1,
                     color=icolor, size=22,
                     tooltip=f"{label}\n評価: {level_lbl}（{val}/4）\n重み: {weight}%")
            add_edge(nid, "qual_score",
                     width=max(1.0, weight / 10), color=icolor,
                     label=f"×{weight}%")

    # ── Col 0: 回帰モデル（全体・指標別・業種別） ─────────────────────────
    # ブレンド重みは res に保存されていないため get_model_blend_weights() から取得
    try:
        from data_cases import get_model_blend_weights
        w_main, w_bench, w_ind = get_model_blend_weights()
    except Exception:
        w_main, w_bench, w_ind = 0.5, 0.3, 0.2

    main_score  = res.get("score_borrower", 0) or 0   # 全体モデルの生スコア
    bench_score = res.get("bench_score", 0) or 0
    ind_score   = res.get("ind_score", 0) or 0
    ind_name    = res.get("ind_name", "業種別") or "業種別"
    # ind_name 例: "運送業_既存先" → "運送業" だけ表示
    ind_label   = ind_name.split("_")[0] if "_" in ind_name else ind_name

    add_node("model_main", f"全体モデル\n{main_score:.1f}%", col=0,
             color="#38bdf8", size=28,
             tooltip=f"全体（新規先/既存先）ロジスティック回帰\nスコア: {main_score:.1f}%\nブレンド重み: {w_main:.0%}")
    add_edge("model_main", "borrower_score",
             width=max(1.5, w_main * 6), color="#38bdf8",
             label=f"×{w_main:.0%}")

    if bench_score > 0:
        add_node("model_bench", f"指標別モデル\n{bench_score:.1f}%", col=0,
                 color="#67e8f9", size=26,
                 tooltip=f"指標ベンチマークモデル（全体_指標）\nスコア: {bench_score:.1f}%\nブレンド重み: {w_bench:.0%}")
        add_edge("model_bench", "borrower_score",
                 width=max(1.0, w_bench * 6), color="#67e8f9",
                 label=f"×{w_bench:.0%}")

    if ind_score > 0:
        add_node("model_ind", f"{ind_label}モデル\n{ind_score:.1f}%", col=0,
                 color="#a5f3fc", size=26,
                 tooltip=f"業種別モデル（{ind_name}）\nスコア: {ind_score:.1f}%\nブレンド重み: {w_ind:.0%}")
        add_edge("model_ind", "borrower_score",
                 width=max(1.0, w_ind * 6), color="#a5f3fc",
                 label=f"×{w_ind:.0%}")

    return {"nodes": nodes, "edges": edges}


# ── D3.js HTML テンプレート ────────────────────────────────────────────────────

_DAG_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {
    margin: 0;
    background: #0f172a;
    font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
    overflow: hidden;
  }
  .tooltip {
    position: absolute;
    background: rgba(15,23,42,0.95);
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    pointer-events: none;
    display: none;
    white-space: pre-line;
    line-height: 1.6;
    max-width: 220px;
    z-index: 999;
  }
  .legend {
    position: absolute;
    bottom: 10px;
    left: 12px;
    color: #64748b;
    font-size: 10px;
    line-height: 1.8;
  }
  .title-badge {
    position: absolute;
    top: 10px;
    left: 12px;
    color: #94a3b8;
    font-size: 11px;
    font-weight: bold;
  }
</style>
</head>
<body>
<div class="tooltip" id="tooltip"></div>
<div class="title-badge">スコアリング因果グラフ（DAG）</div>
<div class="legend">
  ● 水色: 回帰モデル　● 緑: プラス補正　● 赤: マイナス補正<br>
  ● 青: 借手スコア　● オレンジ: 物件スコア　● 紫: 定性<br>
  線の太さ = ブレンド重み / 貢献度
</div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const DATA = __DAG_DATA__;
const W = window.innerWidth || 800;
const H = __HEIGHT__;

// ── 5列の x 座標 ──────────────────────────────────────────────────────────────
const COL_X = { 0: W*0.09, 1: W*0.28, 2: W*0.52, 3: W*0.74, 4: W*0.92 };

// ── ノードを列ごとにグループ化して y 均等配置 ────────────────────────────────
const byCol = {};
DATA.nodes.forEach(n => { (byCol[n.col] = byCol[n.col] || []).push(n); });
Object.entries(byCol).forEach(([col, nodes]) => {
  const pad = 55, step = (H - pad * 2) / Math.max(nodes.length, 1);
  nodes.forEach((n, i) => { n.x = COL_X[col]; n.y = pad + step * i + step / 2; });
});

const nodeById = {};
DATA.nodes.forEach(n => { nodeById[n.id] = n; });

// ── SVG ───────────────────────────────────────────────────────────────────────
const svg = d3.select("body").append("svg").attr("width", W).attr("height", H);
const g = svg.append("g");
svg.call(d3.zoom().scaleExtent([0.3, 3]).on("zoom", e => g.attr("transform", e.transform)));

// 矢印マーカー
const defs = svg.append("defs");
const markerColors = {
  sky:"#38bdf8", pos:"#22c55e", neg:"#ef4444",
  blue:"#0ea5e9", orange:"#f97316", purple:"#8b5cf6", neutral:"#64748b"
};
Object.entries(markerColors).forEach(([key, col]) => {
  defs.append("marker")
    .attr("id", `arrow-${key}`)
    .attr("viewBox", "0 -4 8 8").attr("refX", 14).attr("refY", 0)
    .attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto")
    .append("path").attr("d", "M0,-4L8,0L0,4").attr("fill", col);
});

function markerKey(color) {
  if (["#38bdf8","#67e8f9","#a5f3fc"].includes(color)) return "sky";
  if (["#22c55e","#86efac"].includes(color)) return "pos";
  if (["#ef4444","#fca5a5"].includes(color)) return "neg";
  if (color === "#0ea5e9") return "blue";
  if (color === "#f97316") return "orange";
  if (["#8b5cf6","#a78bfa","#c4b5fd","#7c3aed"].includes(color)) return "purple";
  return "neutral";
}

// ── エッジ ────────────────────────────────────────────────────────────────────
const linkGen = d3.linkHorizontal().x(d => d.x).y(d => d.y);

const edgeSel = g.append("g").selectAll("path")
  .data(DATA.edges).join("path")
  .attr("d", d => {
    const s = nodeById[d.source], t = nodeById[d.target];
    return (s && t) ? linkGen({source: s, target: t}) : "";
  })
  .attr("fill", "none")
  .attr("stroke", d => d.color)
  .attr("stroke-width", d => d.width)
  .attr("stroke-opacity", 0.6)
  .attr("marker-end", d => `url(#arrow-${markerKey(d.color)})`);

// エッジラベル
g.append("g").selectAll("text")
  .data(DATA.edges.filter(d => d.label))
  .join("text")
  .attr("x", d => { const s=nodeById[d.source],t=nodeById[d.target]; return s&&t?(s.x+t.x)/2:0; })
  .attr("y", d => { const s=nodeById[d.source],t=nodeById[d.target]; return s&&t?(s.y+t.y)/2-5:0; })
  .attr("text-anchor", "middle").attr("font-size", 9)
  .attr("fill", d => d.color).attr("opacity", 0.75)
  .text(d => d.label);

// ── ノード ────────────────────────────────────────────────────────────────────
const tooltip = document.getElementById("tooltip");

const nodeGroup = g.append("g").selectAll("g")
  .data(DATA.nodes).join("g")
  .attr("transform", d => `translate(${d.x},${d.y})`)
  .attr("cursor", "pointer");

nodeGroup.append("circle")
  .attr("r", d => d.size / 2)
  .attr("fill", d => d.color).attr("fill-opacity", 0.85)
  .attr("stroke", "#fff").attr("stroke-width", 2)
  .attr("filter", "drop-shadow(0 2px 6px rgba(0,0,0,0.4))");

nodeGroup.each(function(d) {
  const lines = d.label.split("\\n");
  const fs = d.size > 30 ? 11 : 9;
  const lh = d.size > 30 ? 13 : 11;
  const t = d3.select(this).append("text")
    .attr("text-anchor", "middle").attr("fill", "#f8fafc")
    .attr("font-size", fs).attr("font-weight", d.col >= 3 ? "bold" : "normal");
  const dy0 = -((lines.length - 1) * lh) / 2;
  lines.forEach((line, i) => t.append("tspan").attr("x", 0).attr("dy", i===0 ? dy0 : lh).text(line));
});

// 列ヘッダー
[
  {col:0, text:"回帰モデル"},
  {col:1, text:"補正因子"},
  {col:2, text:"スコア成分"},
  {col:3, text:"総合スコア"},
  {col:4, text:"最終判定"},
].forEach(d => {
  g.append("text").attr("x", COL_X[d.col]).attr("y", 18)
   .attr("text-anchor", "middle").attr("font-size", 10)
   .attr("fill", "#475569").attr("font-weight", "bold").text(d.text);
  g.append("line")
   .attr("x1", COL_X[d.col]).attr("y1", 25)
   .attr("x2", COL_X[d.col]).attr("y2", H - 30)
   .attr("stroke", "#1e293b").attr("stroke-width", 1).attr("stroke-dasharray", "4,4");
});

// ツールチップ
nodeGroup
  .on("mouseover", (e, d) => {
    tooltip.textContent = d.tooltip || d.label;
    tooltip.style.display = "block";
  })
  .on("mousemove", e => {
    tooltip.style.left = (e.pageX + 14) + "px";
    tooltip.style.top  = (e.pageY - 30) + "px";
  })
  .on("mouseout", () => { tooltip.style.display = "none"; });

// ホバーハイライト
nodeGroup
  .on("mouseover.hl", (e, d) => {
    edgeSel.attr("stroke-opacity", l =>
      (l.source === d.id || l.target === d.id) ? 1.0 : 0.07);
    nodeGroup.select("circle").attr("fill-opacity", n => {
      const hit = DATA.edges.some(l => {
        const s = typeof l.source==="object" ? l.source.id : l.source;
        const t = typeof l.target==="object" ? l.target.id : l.target;
        return (s===d.id&&t===n.id)||(t===d.id&&s===n.id)||n.id===d.id;
      });
      return hit ? 1.0 : 0.25;
    });
  })
  .on("mouseout.hl", () => {
    edgeSel.attr("stroke-opacity", 0.6);
    nodeGroup.select("circle").attr("fill-opacity", 0.85);
  });
</script>
</body>
</html>
"""


# ── Streamlit 表示関数 ─────────────────────────────────────────────────────────

def render_score_dag(res: dict, height: int = 480):
    ai_factors = res.get("ai_completed_factors") or []
    qcorr = res.get("qualitative_scoring_correction")
    n_qual_items = len((qcorr or {}).get("items") or {})

    # 行数が多い列（Col 1）に合わせて高さを調整
    n_col1 = len(ai_factors) + n_qual_items
    auto_height = max(height, n_col1 * 52 + 130)
    auto_height = min(auto_height, 900)

    dag_data = build_dag_data(res)
    html = (_DAG_HTML
            .replace("__DAG_DATA__", json.dumps(dag_data, ensure_ascii=False))
            .replace("__HEIGHT__", str(auto_height)))
    components.html(html, height=auto_height + 10, scrolling=False)

    hantei = res.get("hantei", "—")
    score  = res.get("score", 0) or 0
    n_pos  = sum(1 for f in ai_factors if f.get("effect_percent", 0) > 0)
    n_neg  = sum(1 for f in ai_factors if f.get("effect_percent", 0) < 0)
    st.caption(
        f"**左→右へ因果が流れます。** "
        f"回帰モデル3種（全体/指標別/業種別）ブレンド → 補正因子 {len(ai_factors)}件"
        f"（+{n_pos} / -{n_neg}）＋定性項目 {n_qual_items}件 → "
        f"スコア成分 → 総合スコア {score:.1f} → {hantei}。"
        f" ノードにホバーで詳細、エッジの太さはブレンド重み/貢献度。"
    )
