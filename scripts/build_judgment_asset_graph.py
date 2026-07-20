#!/usr/bin/env python3
"""Build an offline HTML graph of judgment assets.

The graph is a local visualization sidecar. It reads canonical judgment assets,
field feedback, and the latest growth evaluation, then writes a self-contained
HTML report. It does not promote assets, change prompts, write to Obsidian,
touch scoring, or call external services.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_JSON = PROJECT_ROOT / "data" / "canonical_judgment_rules.json"
DEFAULT_FEEDBACK_JSONL = PROJECT_ROOT / "data" / "judgment_asset_usage_feedback.jsonl"
DEFAULT_GROWTH_EVALUATION_JSON = PROJECT_ROOT / "reports" / "shion_growth_evaluation_latest.json"
DEFAULT_OUTPUT_HTML = PROJECT_ROOT / "reports" / "judgment_asset_graph_latest.html"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "judgment_asset_graph_latest.json"

NODE_COLORS = {
    "rule": "#2563eb",
    "risk_axis": "#f59e0b",
    "domain": "#10b981",
    "evidence": "#8b5cf6",
    "case": "#ef4444",
}

EDGE_COLORS = {
    "risk_axis": "#d97706",
    "domain": "#059669",
    "evidence": "#7c3aed",
    "used": "#64748b",
    "helped": "#16a34a",
    "challenged": "#dc2626",
    "rejected": "#991b1b",
    "neutral": "#94a3b8",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _canonical_rules(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    rules = canonical.get("rules")
    if isinstance(rules, list):
        return [item for item in rules if isinstance(item, dict)]
    rules = canonical.get("canonical_rules")
    if isinstance(rules, list):
        return [item for item in rules if isinstance(item, dict)]
    return []


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    normalized = str(value).strip()
    return [normalized] if normalized else []


def _shorten(value: str, length: int = 90) -> str:
    normalized = " ".join(str(value or "").split())
    if len(normalized) <= length:
        return normalized
    return normalized[: length - 1] + "…"


def _feedback_rule_id(row: dict[str, Any]) -> str:
    for key in ("rule_id", "judgment_asset_id", "asset_id", "id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _feedback_outcome(row: dict[str, Any]) -> str:
    outcome = str(row.get("outcome") or row.get("status") or "").strip().lower()
    if outcome in {"used", "helped", "challenged", "rejected", "neutral"}:
        return outcome
    if row.get("helped") is True:
        return "helped"
    if row.get("challenged") is True or row.get("wrong") is True:
        return "challenged"
    if row.get("used") is True:
        return "used"
    return ""


def _case_id(row: dict[str, Any], fallback_index: int) -> str:
    value = str(row.get("case_id") or row.get("case") or "").strip()
    return value or f"feedback-{fallback_index + 1}"


def _rule_label(rule: dict[str, Any]) -> str:
    concept = str(rule.get("concept") or "").strip()
    if concept:
        return concept
    statement = str(rule.get("canonical_statement") or "").strip()
    return _shorten(statement, 42) if statement else str(rule.get("id") or "rule")


def build_graph_data(
    *,
    canonical: dict[str, Any],
    feedback_rows: list[dict[str, Any]] | None = None,
    growth_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feedback_rows = feedback_rows or []
    growth_evaluation = growth_evaluation or {}
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    edge_seen: set[tuple[str, str, str]] = set()
    feedback_by_rule: dict[str, Counter[str]] = defaultdict(Counter)
    feedback_cases_by_rule: dict[str, set[str]] = defaultdict(set)

    for index, row in enumerate(feedback_rows):
        rule_id = _feedback_rule_id(row)
        outcome = _feedback_outcome(row)
        if not rule_id or not outcome:
            continue
        feedback_by_rule[rule_id][outcome] += 1
        feedback_cases_by_rule[rule_id].add(_case_id(row, index))

    def add_node(node: dict[str, Any]) -> None:
        node_id = str(node["id"])
        if node_id in nodes:
            existing = nodes[node_id]
            existing["weight"] = max(float(existing.get("weight") or 1), float(node.get("weight") or 1))
            existing["degree"] = int(existing.get("degree") or 0) + int(node.get("degree") or 0)
            return
        node.setdefault("degree", 0)
        node.setdefault("color", NODE_COLORS.get(str(node.get("type")), "#475569"))
        nodes[node_id] = node

    def add_edge(source: str, target: str, edge_type: str, label: str = "", weight: float = 1.0) -> None:
        key = (source, target, edge_type)
        if key in edge_seen:
            return
        edge_seen.add(key)
        edges.append(
            {
                "source": source,
                "target": target,
                "type": edge_type,
                "label": label or edge_type,
                "weight": weight,
                "color": EDGE_COLORS.get(edge_type, "#64748b"),
            }
        )
        if source in nodes:
            nodes[source]["degree"] = int(nodes[source].get("degree") or 0) + 1
        if target in nodes:
            nodes[target]["degree"] = int(nodes[target].get("degree") or 0) + 1

    rules = [rule for rule in _canonical_rules(canonical) if str(rule.get("status") or "active") == "active"]
    for rule in rules:
        rule_id = str(rule.get("id") or "").strip()
        if not rule_id:
            continue
        feedback_counts = feedback_by_rule.get(rule_id, Counter())
        helped = int(feedback_counts.get("helped") or 0)
        challenged = int(feedback_counts.get("challenged") or 0) + int(feedback_counts.get("rejected") or 0)
        used = sum(int(v or 0) for v in feedback_counts.values())
        confidence = float(rule.get("confidence") or 0)
        evidence_count = int(rule.get("evidence_count") or 0)
        user_evidence_count = int(rule.get("user_evidence_count") or 0)
        add_node(
            {
                "id": f"rule:{rule_id}",
                "type": "rule",
                "label": _rule_label(rule),
                "title": str(rule.get("canonical_statement") or _rule_label(rule)),
                "concept": str(rule.get("concept") or ""),
                "statement": str(rule.get("canonical_statement") or ""),
                "confidence": confidence,
                "evidence_count": evidence_count,
                "user_evidence_count": user_evidence_count,
                "feedback_used": used,
                "feedback_helped": helped,
                "feedback_challenged": challenged,
                "weight": max(8, evidence_count + user_evidence_count * 2 + used * 4),
            }
        )

        for axis in _as_list(rule.get("risk_axis")):
            axis_id = f"risk:{axis}"
            add_node(
                {
                    "id": axis_id,
                    "type": "risk_axis",
                    "label": axis,
                    "title": f"Risk axis: {axis}",
                    "weight": 10,
                }
            )
            add_edge(f"rule:{rule_id}", axis_id, "risk_axis", "risk axis", 1.4)

        domains = _as_list(rule.get("domains")) or _as_list(rule.get("domain"))
        for domain in domains:
            domain_id = f"domain:{domain}"
            add_node(
                {
                    "id": domain_id,
                    "type": "domain",
                    "label": domain,
                    "title": f"Domain: {domain}",
                    "weight": 8,
                }
            )
            add_edge(f"rule:{rule_id}", domain_id, "domain", "domain", 0.9)

        for evidence_path in _as_list(rule.get("evidence_paths"))[:3]:
            evidence_id = f"evidence:{evidence_path}"
            add_node(
                {
                    "id": evidence_id,
                    "type": "evidence",
                    "label": Path(evidence_path).name or _shorten(evidence_path, 36),
                    "title": evidence_path,
                    "weight": 6,
                }
            )
            add_edge(f"rule:{rule_id}", evidence_id, "evidence", "evidence", 0.6)

    for index, row in enumerate(feedback_rows):
        rule_id = _feedback_rule_id(row)
        outcome = _feedback_outcome(row)
        if not rule_id or not outcome:
            continue
        rule_node_id = f"rule:{rule_id}"
        if rule_node_id not in nodes:
            continue
        case_id = _case_id(row, index)
        case_node_id = f"case:{case_id}"
        add_node(
            {
                "id": case_node_id,
                "type": "case",
                "label": case_id,
                "title": str(row.get("note") or case_id),
                "outcome": outcome,
                "weight": 12,
            }
        )
        add_edge(rule_node_id, case_node_id, outcome, outcome, 2.2)

    type_counts = Counter(str(node.get("type") or "") for node in nodes.values())
    outcome_counts = Counter(_feedback_outcome(row) for row in feedback_rows)
    outcome_counts.pop("", None)
    latest_judgment = growth_evaluation.get("judgment") if isinstance(growth_evaluation.get("judgment"), dict) else {}
    latest_period = growth_evaluation.get("period") if isinstance(growth_evaluation.get("period"), dict) else {}
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "mode": "local_visualization_only",
        "guardrail": "no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun_no_obsidian_write",
        "summary": {
            "nodes": len(nodes),
            "edges": len(edges),
            "rules": type_counts.get("rule", 0),
            "risk_axes": type_counts.get("risk_axis", 0),
            "domains": type_counts.get("domain", 0),
            "evidence": type_counts.get("evidence", 0),
            "cases": type_counts.get("case", 0),
            "feedback": dict(sorted(outcome_counts.items())),
            "growth_label": latest_judgment.get("label", ""),
            "growth_score": latest_judgment.get("score", ""),
            "growth_period": {
                "start_date": latest_period.get("start_date", ""),
                "end_date": latest_period.get("end_date", ""),
            },
        },
        "nodes": sorted(nodes.values(), key=lambda node: (str(node.get("type")), str(node.get("label")))),
        "edges": edges,
    }


def build_html(graph: dict[str, Any]) -> str:
    payload_json = json.dumps(graph, ensure_ascii=False)
    summary = graph.get("summary") if isinstance(graph.get("summary"), dict) else {}
    generated = html.escape(str(graph.get("generated_at") or ""))
    growth_label = html.escape(str(summary.get("growth_label") or "未判定"))
    growth_score = html.escape(str(summary.get("growth_score") or "-"))
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Judgment Asset Graph</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f8fafc;
  --panel: #ffffff;
  --ink: #0f172a;
  --muted: #64748b;
  --line: #cbd5e1;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--ink);
}}
.app {{
  display: grid;
  grid-template-columns: 320px 1fr;
  min-height: 100vh;
}}
aside {{
  background: var(--panel);
  border-right: 1px solid var(--line);
  padding: 18px;
  overflow: auto;
}}
main {{ position: relative; min-width: 0; }}
h1 {{ font-size: 20px; margin: 0 0 8px; letter-spacing: 0; }}
h2 {{ font-size: 13px; margin: 20px 0 8px; color: var(--muted); text-transform: uppercase; letter-spacing: 0; }}
.lead {{ margin: 0 0 16px; color: var(--muted); line-height: 1.5; }}
.verdict {{
  border: 1px solid var(--line);
  background: #f8fafc;
  border-radius: 8px;
  padding: 12px;
  margin: 12px 0;
}}
.verdict strong {{ display: block; font-size: 18px; margin-bottom: 4px; }}
.stats {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}}
.stat {{
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  background: #fff;
}}
.stat b {{ display: block; font-size: 20px; }}
.stat span {{ color: var(--muted); font-size: 12px; }}
.filters {{ display: grid; gap: 8px; }}
label {{ display: flex; align-items: center; gap: 8px; color: #334155; font-size: 14px; }}
input[type="search"] {{
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 11px;
  font-size: 14px;
}}
button {{
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  padding: 9px 10px;
  cursor: pointer;
}}
button:hover {{ background: #f1f5f9; }}
#graph {{ width: 100%; height: 100vh; display: block; }}
.tooltip {{
  position: absolute;
  pointer-events: none;
  max-width: 360px;
  background: rgba(15, 23, 42, 0.94);
  color: #fff;
  padding: 10px 12px;
  border-radius: 8px;
  font-size: 12px;
  line-height: 1.45;
  opacity: 0;
  transform: translate(12px, 12px);
}}
.legend {{ display: grid; gap: 7px; }}
.legend-row {{ display: flex; align-items: center; gap: 8px; font-size: 13px; color: #334155; }}
.dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.note {{ color: var(--muted); font-size: 12px; line-height: 1.5; margin-top: 16px; }}
@media (max-width: 860px) {{
  .app {{ grid-template-columns: 1fr; }}
  aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
  #graph {{ height: 72vh; }}
}}
</style>
</head>
<body>
<div class="app">
  <aside>
    <h1>Judgment Asset Graph</h1>
    <p class="lead">判断資産、リスク軸、根拠ログ、実案件フィードバックのつながりを見るローカル図。</p>
    <div class="verdict">
      <strong>{growth_label}</strong>
      <span>Growth score: {growth_score}</span>
    </div>
    <div class="stats">
      <div class="stat"><b id="stat-nodes">{summary.get("nodes", 0)}</b><span>nodes</span></div>
      <div class="stat"><b id="stat-edges">{summary.get("edges", 0)}</b><span>edges</span></div>
      <div class="stat"><b>{summary.get("rules", 0)}</b><span>rules</span></div>
      <div class="stat"><b>{summary.get("cases", 0)}</b><span>cases</span></div>
    </div>
    <h2>Search</h2>
    <input id="search" type="search" placeholder="concept, risk axis, evidence...">
    <h2>Show</h2>
    <div class="filters">
      <label><input type="checkbox" data-type="rule" checked> Judgment assets</label>
      <label><input type="checkbox" data-type="risk_axis" checked> Risk axes</label>
      <label><input type="checkbox" data-type="domain" checked> Domains</label>
      <label><input type="checkbox" data-type="evidence" checked> Evidence logs</label>
      <label><input type="checkbox" data-type="case" checked> Cases / feedback</label>
    </div>
    <h2>Legend</h2>
    <div class="legend">
      <div class="legend-row"><span class="dot" style="background:#2563eb"></span>判断資産</div>
      <div class="legend-row"><span class="dot" style="background:#f59e0b"></span>リスク軸</div>
      <div class="legend-row"><span class="dot" style="background:#10b981"></span>ドメイン</div>
      <div class="legend-row"><span class="dot" style="background:#8b5cf6"></span>根拠ログ</div>
      <div class="legend-row"><span class="dot" style="background:#ef4444"></span>案件フィードバック</div>
    </div>
    <h2>Actions</h2>
    <button id="reset">Reset layout</button>
    <p class="note">Generated: {generated}<br>Offline HTML. No CDN. No external calls.</p>
  </aside>
  <main>
    <svg id="graph" role="img" aria-label="Judgment asset network graph"></svg>
    <div id="tooltip" class="tooltip"></div>
  </main>
</div>
<script>
const graph = {payload_json};
const svg = document.getElementById("graph");
const tooltip = document.getElementById("tooltip");
const search = document.getElementById("search");
const filters = Array.from(document.querySelectorAll("input[data-type]"));
let width = 0;
let height = 0;
let nodes = [];
let edges = [];
let frame = null;
let tick = 0;

function activeTypes() {{
  return new Set(filters.filter(input => input.checked).map(input => input.dataset.type));
}}

function visibleData() {{
  const term = search.value.trim().toLowerCase();
  const types = activeTypes();
  const baseNodes = graph.nodes.filter(node => {{
    if (!types.has(node.type)) return false;
    if (!term) return true;
    return [node.label, node.title, node.concept, node.statement].join(" ").toLowerCase().includes(term);
  }});
  const nodeIds = new Set(baseNodes.map(node => node.id));
  const baseEdges = graph.edges.filter(edge => nodeIds.has(edge.source) && nodeIds.has(edge.target));
  return {{ nodes: baseNodes.map(node => ({{...node}})), edges: baseEdges.map(edge => ({{...edge}})) }};
}}

function initPositions() {{
  width = svg.clientWidth || 900;
  height = svg.clientHeight || 700;
  nodes.forEach((node, index) => {{
    const angle = (index / Math.max(1, nodes.length)) * Math.PI * 2;
    const radius = Math.min(width, height) * (node.type === "rule" ? 0.22 : 0.36);
    node.x = width / 2 + Math.cos(angle) * radius;
    node.y = height / 2 + Math.sin(angle) * radius;
    node.vx = 0;
    node.vy = 0;
  }});
}}

function radius(node) {{
  const weight = Number(node.weight || 1);
  const base = node.type === "rule" ? 9 : 6;
  return Math.max(5, Math.min(22, base + Math.sqrt(weight) * 1.5));
}}

function graphPadding() {{
  return Math.max(72, Math.min(132, Math.min(width, height) * 0.12));
}}

function simulate() {{
  const byId = new Map(nodes.map(node => [node.id, node]));
  const centerX = width / 2;
  const centerY = height / 2;
  for (let i = 0; i < nodes.length; i++) {{
    const a = nodes[i];
    a.vx += (centerX - a.x) * 0.0008;
    a.vy += (centerY - a.y) * 0.0008;
    for (let j = i + 1; j < nodes.length; j++) {{
      const b = nodes[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dist2 = Math.max(80, dx * dx + dy * dy);
      const force = 150 / dist2;
      a.vx += dx * force;
      a.vy += dy * force;
      b.vx -= dx * force;
      b.vy -= dy * force;
    }}
  }}
  edges.forEach(edge => {{
    const source = byId.get(edge.source);
    const target = byId.get(edge.target);
    if (!source || !target) return;
    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const dist = Math.max(1, Math.hypot(dx, dy));
    const targetDist = edge.type === "evidence" ? 155 : edge.type === "domain" ? 125 : 105;
    const force = (dist - targetDist) * 0.0025 * Number(edge.weight || 1);
    const fx = dx / dist * force;
    const fy = dy / dist * force;
    source.vx += fx;
    source.vy += fy;
    target.vx -= fx;
    target.vy -= fy;
  }});
  nodes.forEach(node => {{
    node.vx *= 0.86;
    node.vy *= 0.86;
    const pad = graphPadding();
    node.x = Math.max(pad, Math.min(width - pad, node.x + node.vx));
    node.y = Math.max(pad, Math.min(height - pad, node.y + node.vy));
  }});
}}

function render() {{
  const byId = new Map(nodes.map(node => [node.id, node]));
  svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
  const edgeMarkup = edges.map(edge => {{
    const source = byId.get(edge.source);
    const target = byId.get(edge.target);
    if (!source || !target) return "";
    const width = Math.max(1, Math.min(5, Number(edge.weight || 1)));
    return `<line x1="${{source.x.toFixed(1)}}" y1="${{source.y.toFixed(1)}}" x2="${{target.x.toFixed(1)}}" y2="${{target.y.toFixed(1)}}" stroke="${{edge.color}}" stroke-width="${{width}}" stroke-opacity="0.42" />`;
  }}).join("");
  const nodeMarkup = nodes.map(node => {{
    const r = radius(node);
    const label = String(node.label || "");
    const escaped = label.replace(/[&<>"']/g, char => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[char]));
    const fontSize = node.type === "rule" ? 11 : 10;
    const labelY = r + 13;
    const showLabel = node.type !== "evidence";
    const labelMarkup = showLabel
      ? `<text y="${{labelY}}" text-anchor="middle" font-size="${{fontSize}}" fill="#0f172a">${{escaped.slice(0, 20)}}</text>`
      : "";
    return `<g class="node" data-id="${{node.id}}" transform="translate(${{node.x.toFixed(1)}},${{node.y.toFixed(1)}})">
      <circle r="${{r}}" fill="${{node.color}}" stroke="#fff" stroke-width="2" />
      ${{labelMarkup}}
    </g>`;
  }}).join("");
  svg.innerHTML = `<rect width="${{width}}" height="${{height}}" fill="#f8fafc"></rect>${{edgeMarkup}}${{nodeMarkup}}`;
  svg.querySelectorAll(".node").forEach(element => {{
    const node = byId.get(element.dataset.id);
    element.addEventListener("mousemove", event => showTooltip(event, node));
    element.addEventListener("mouseleave", hideTooltip);
  }});
}}

function showTooltip(event, node) {{
  if (!node) return;
  const parts = [
    `<strong>${{node.label || node.id}}</strong>`,
    `type: ${{node.type}}`,
  ];
  if (node.statement) parts.push(node.statement);
  if (node.evidence_count !== undefined) parts.push(`evidence: ${{node.evidence_count}} / user: ${{node.user_evidence_count || 0}}`);
  if (node.feedback_used) parts.push(`feedback: used ${{node.feedback_used}}, helped ${{node.feedback_helped || 0}}, challenged ${{node.feedback_challenged || 0}}`);
  if (node.title && !node.statement) parts.push(node.title);
  tooltip.innerHTML = parts.join("<br>");
  tooltip.style.opacity = "1";
  tooltip.style.left = `${{event.clientX}}px`;
  tooltip.style.top = `${{event.clientY}}px`;
}}

function hideTooltip() {{
  tooltip.style.opacity = "0";
}}

function step() {{
  if (tick < 520) {{
    simulate();
    tick += 1;
    render();
    frame = requestAnimationFrame(step);
  }}
}}

function rebuild() {{
  if (frame) cancelAnimationFrame(frame);
  const data = visibleData();
  nodes = data.nodes;
  edges = data.edges;
  document.getElementById("stat-nodes").textContent = nodes.length;
  document.getElementById("stat-edges").textContent = edges.length;
  tick = 0;
  initPositions();
  step();
}}

window.addEventListener("resize", rebuild);
search.addEventListener("input", rebuild);
filters.forEach(input => input.addEventListener("change", rebuild));
document.getElementById("reset").addEventListener("click", rebuild);
rebuild();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--feedback-jsonl", type=Path, default=DEFAULT_FEEDBACK_JSONL)
    parser.add_argument("--growth-evaluation-json", type=Path, default=DEFAULT_GROWTH_EVALUATION_JSON)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph = build_graph_data(
        canonical=_read_json(args.canonical_json),
        feedback_rows=_read_jsonl(args.feedback_jsonl),
        growth_evaluation=_read_json(args.growth_evaluation_json),
    )
    _write_json(args.output_json, graph)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(build_html(graph), encoding="utf-8")
    print(
        "Judgment Asset Graph: "
        f"{graph['summary']['nodes']} nodes / {graph['summary']['edges']} edges -> {args.output_html}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
