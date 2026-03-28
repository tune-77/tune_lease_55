"""
D3.js アニメーションスコアゲージ
審査結果ページに埋め込む大型スピードメーター型ゲージ。
針が0からスコアまでアニメーションしながら跳ね上がる。
"""
import streamlit.components.v1 as components


_GAUGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body { margin:0; background:transparent; display:flex; justify-content:center; align-items:center; height:__HEIGHT__px; }
  .score-label { font-family: 'Segoe UI', sans-serif; }
</style>
</head>
<body>
<svg id="gauge" width="__WIDTH__" height="__HEIGHT__" viewBox="0 0 300 200"></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const SCORE = __SCORE__;
const HANTEI = "__HANTEI__";
const W = 300, H = 200;
const cx = W / 2, cy = 155;
const R = 110, rInner = 72;

// 角度: -225度(左端) ～ 45度(右端) = 270度スイープ
const startAngle = -225 * Math.PI / 180;
const endAngle   =   45 * Math.PI / 180;
const scoreToAngle = s => startAngle + (s / 100) * (endAngle - startAngle);

const svg = d3.select("#gauge");

// ── グラデーション弧（赤→黄→緑） ───────────────────────────────────────
const zones = [
  { from: 0,  to: 41,  color: "#ef4444" },
  { from: 41, to: 71,  color: "#f59e0b" },
  { from: 71, to: 100, color: "#22c55e" },
];

const arc = d3.arc().innerRadius(rInner).outerRadius(R).cornerRadius(2);

zones.forEach(z => {
  svg.append("path")
    .attr("d", arc({
      startAngle: scoreToAngle(z.from),
      endAngle:   scoreToAngle(z.to),
    }))
    .attr("fill", z.color)
    .attr("opacity", 0.85)
    .attr("transform", `translate(${cx},${cy})`);
});

// 内側の白い塗り（ドーナツ内部）
svg.append("circle")
  .attr("cx", cx).attr("cy", cy).attr("r", rInner - 2)
  .attr("fill", "#0f172a");

// 承認ライン マーカー（71点）
const lineAngle = scoreToAngle(71);
svg.append("line")
  .attr("x1", cx + (rInner - 4) * Math.cos(lineAngle))
  .attr("y1", cy + (rInner - 4) * Math.sin(lineAngle))
  .attr("x2", cx + (R + 8) * Math.cos(lineAngle))
  .attr("y2", cy + (R + 8) * Math.sin(lineAngle))
  .attr("stroke", "#fff")
  .attr("stroke-width", 2)
  .attr("stroke-dasharray", "4,2");

svg.append("text")
  .attr("x", cx + (R + 18) * Math.cos(lineAngle))
  .attr("y", cy + (R + 18) * Math.sin(lineAngle))
  .attr("fill", "#94a3b8")
  .attr("font-size", "9px")
  .attr("text-anchor", "middle")
  .text("71");

// ── 目盛り ──────────────────────────────────────────────────────────────
[0, 25, 50, 75, 100].forEach(v => {
  const a = scoreToAngle(v);
  svg.append("line")
    .attr("x1", cx + (R + 2) * Math.cos(a)).attr("y1", cy + (R + 2) * Math.sin(a))
    .attr("x2", cx + (R + 10) * Math.cos(a)).attr("y2", cy + (R + 10) * Math.sin(a))
    .attr("stroke", "#475569").attr("stroke-width", 1.5);
  svg.append("text")
    .attr("x", cx + (R + 20) * Math.cos(a)).attr("y", cy + (R + 20) * Math.sin(a))
    .attr("fill", "#64748b").attr("font-size", "9px").attr("text-anchor", "middle")
    .text(v);
});

// ── 針 ──────────────────────────────────────────────────────────────────
const needleLen = R - 8;
const needleGroup = svg.append("g").attr("transform", `translate(${cx},${cy})`);

const needle = needleGroup.append("line")
  .attr("x1", 0).attr("y1", 0)
  .attr("stroke", "#f8fafc").attr("stroke-width", 3)
  .attr("stroke-linecap", "round");

needleGroup.append("circle").attr("r", 8).attr("fill", "#1e293b").attr("stroke", "#f8fafc").attr("stroke-width", 2);

// ── スコア数字（カウントアップ） ────────────────────────────────────────
const scoreText = svg.append("text")
  .attr("x", cx).attr("y", cy - 18)
  .attr("text-anchor", "middle")
  .attr("fill", "#f8fafc")
  .attr("font-size", "36px")
  .attr("font-weight", "800")
  .attr("font-family", "Segoe UI, sans-serif")
  .text("0");

svg.append("text")
  .attr("x", cx).attr("y", cy - 2)
  .attr("text-anchor", "middle")
  .attr("fill", "#94a3b8")
  .attr("font-size", "11px")
  .text("審査スコア");

// 判定テキスト
const hanteiColor = SCORE >= 71 ? "#22c55e" : SCORE >= 41 ? "#f59e0b" : "#ef4444";
svg.append("text")
  .attr("x", cx).attr("y", cy + 22)
  .attr("text-anchor", "middle")
  .attr("fill", hanteiColor)
  .attr("font-size", "15px")
  .attr("font-weight", "700")
  .text(HANTEI);

// ── アニメーション ───────────────────────────────────────────────────────
const duration = 1800;
const ease = d3.easeCubicOut;
const start = Date.now();

function updateNeedle(angle) {
  needle
    .attr("x2", needleLen * Math.cos(angle))
    .attr("y2", needleLen * Math.sin(angle));
}

updateNeedle(scoreToAngle(0));

function animate() {
  const elapsed = Date.now() - start;
  const t = Math.min(elapsed / duration, 1);
  const current = ease(t) * SCORE;
  updateNeedle(scoreToAngle(current));
  scoreText.text(Math.round(current));
  if (t < 1) requestAnimationFrame(animate);
}

requestAnimationFrame(animate);
</script>
</body>
</html>
"""


def render_score_gauge(score: float, hantei: str = "", width: int = 300, height: int = 200):
    """D3アニメーションゲージをStreamlitに埋め込む"""
    html = (_GAUGE_TEMPLATE
            .replace("__SCORE__", str(round(float(score), 1)))
            .replace("__HANTEI__", hantei.replace('"', ''))
            .replace("__WIDTH__", str(width))
            .replace("__HEIGHT__", str(height)))
    components.html(html, height=height + 10, scrolling=False)
