#!/usr/bin/env python3
"""Check whether Obsidian graph complexity helps judgment.

This is a read-only sidecar. It scans wikilinks in the Obsidian Vault and joins
them with RAG feedback/search logs plus Shion memory usage. It does not edit
Obsidian, RAG ranking, prompts, scoring, or judgment assets.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAULT = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
DEFAULT_MEMORY_INDEX = PROJECT_ROOT / "data" / "shion_memory_index.json"
DEFAULT_MEMORY_USAGE = PROJECT_ROOT / "data" / "shion_memory_usage_log.jsonl"
DEFAULT_RAG_SEARCH = PROJECT_ROOT / "data" / "rag_search_log.jsonl"
DEFAULT_RAG_FEEDBACK = PROJECT_ROOT / "data" / "rag_feedback_log.jsonl"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "obsidian_graph_judgment_effect_latest.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "obsidian_graph_judgment_effect_latest.md"

GUARDRAIL = "read_only_no_obsidian_write_no_rag_rank_no_prompt_no_scoring_change"
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
EXCLUDED_PARTS = {".obsidian", ".trash", ".git", "node_modules"}
POSITIVE_RATINGS = {"good", "useful", "shion_like", "specific", "discomfort_hit"}
NEGATIVE_RATINGS = {"bad", "wrong", "needs_fix", "thin", "not_shion", "over_inferred"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path, *, limit: int = 50000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _note_key(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("[[") and text.endswith("]]"):
        text = text[2:-2]
    text = text.split("|", 1)[0].split("#", 1)[0].strip()
    if text.endswith(".md"):
        text = text[:-3]
    return Path(text).stem.strip()


def _title(body: str, path: Path) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def scan_graph(vault: Path) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    duplicate_stems: Counter[str] = Counter()
    raw_edges: list[tuple[str, str]] = []
    if not vault.exists():
        return {"nodes": {}, "edges": [], "vault_exists": False}

    for path in sorted(vault.rglob("*.md")):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        key = path.stem
        duplicate_stems[key] += 1
        nodes.setdefault(
            key,
            {
                "key": key,
                "title": _title(body, path),
                "path": _safe_rel(path, vault),
                "duplicate_stem_count": 0,
            },
        )
        for target in WIKILINK_RE.findall(body):
            target_key = _note_key(target)
            if target_key and target_key != key:
                raw_edges.append((key, target_key))

    edges = [(src, dst) for src, dst in raw_edges if src in nodes and dst in nodes]
    inbound: Counter[str] = Counter()
    outbound: Counter[str] = Counter()
    for src, dst in edges:
        outbound[src] += 1
        inbound[dst] += 1
    for key, node in nodes.items():
        node["in_degree"] = int(inbound[key])
        node["out_degree"] = int(outbound[key])
        node["degree"] = int(inbound[key] + outbound[key])
        node["duplicate_stem_count"] = int(duplicate_stems[key])
    return {"nodes": nodes, "edges": edges, "vault_exists": True}


def build_report(
    *,
    vault: Path,
    memory_index: dict[str, Any],
    memory_usage_rows: list[dict[str, Any]],
    rag_search_rows: list[dict[str, Any]],
    rag_feedback_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    graph = scan_graph(vault)
    nodes: dict[str, dict[str, Any]] = graph["nodes"]
    signals: dict[str, dict[str, float]] = defaultdict(lambda: {
        "rag_search_weight": 0.0,
        "rag_feedback_positive": 0.0,
        "rag_feedback_negative": 0.0,
        "memory_usage_count": 0.0,
    })

    for row in rag_search_rows:
        for result in row.get("results") or []:
            if not isinstance(result, dict):
                continue
            key = _note_key(result.get("obsidian_ref") or result.get("file_name"))
            if not key:
                continue
            try:
                rank = max(1, int(result.get("rank") or 10))
            except (TypeError, ValueError):
                rank = 10
            signals[key]["rag_search_weight"] += 1.0 / rank

    for row in rag_feedback_rows:
        key = _note_key(row.get("obsidian_ref") or row.get("file_name") or row.get("doc_id"))
        rating = str(row.get("rating") or "").strip().lower()
        if not key:
            continue
        if rating in POSITIVE_RATINGS:
            signals[key]["rag_feedback_positive"] += 1
        elif rating in NEGATIVE_RATINGS:
            signals[key]["rag_feedback_negative"] += 1

    index_by_id = {
        str(record.get("id") or ""): record
        for record in memory_index.get("records") or []
        if isinstance(record, dict) and record.get("id")
    }
    for row in memory_usage_rows:
        for ref in row.get("refs") or []:
            record = index_by_id.get(str(ref))
            if not record:
                continue
            key = _note_key(record.get("source_path"))
            if key:
                signals[key]["memory_usage_count"] += 1

    records: list[dict[str, Any]] = []
    for key, node in nodes.items():
        sig = signals.get(key, {})
        used_score = _used_score(sig)
        complexity = _complexity_score(node)
        judgment_effect_score = round(min(100.0, used_score * 8.0 + complexity * 0.8), 1)
        record = {
            **node,
            "signals": {k: round(float(v), 3) for k, v in sig.items()},
            "complexity_score": round(complexity, 3),
            "used_score": round(used_score, 3),
            "judgment_effect_score": judgment_effect_score,
        }
        record["bucket"] = _bucket(record)
        record["next_action"] = _next_action(record["bucket"])
        records.append(record)

    records.sort(key=lambda item: (float(item["judgment_effect_score"]), int(item["degree"])), reverse=True)
    buckets = Counter(str(item["bucket"]) for item in records)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "guardrail": GUARDRAIL,
        "vault": str(vault),
        "summary": {
            "vault_exists": bool(graph["vault_exists"]),
            "notes": len(nodes),
            "edges": len(graph["edges"]),
            "avg_degree": round((sum(int(n.get("degree") or 0) for n in nodes.values()) / len(nodes)) if nodes else 0.0, 2),
            "notes_with_usage_signal": sum(1 for item in records if float(item["used_score"]) > 0),
            "buckets": dict(sorted(buckets.items())),
        },
        "answer": _answer(records),
        "top_effective_hubs": [r for r in records if r["bucket"] == "effective_hub"][:10],
        "bridge_candidates": [r for r in records if r["bucket"] == "bridge_candidate"][:10],
        "complex_but_unproven": sorted(
            [r for r in records if r["bucket"] == "complex_but_unproven"],
            key=lambda item: int(item["degree"]),
            reverse=True,
        )[:15],
        "isolated_but_used": [r for r in records if r["bucket"] == "isolated_but_used"][:10],
        "noisy_nodes": [r for r in records if r["bucket"] == "noisy"][:10],
    }


def _used_score(signals: dict[str, float]) -> float:
    return (
        float(signals.get("rag_search_weight") or 0.0)
        + float(signals.get("rag_feedback_positive") or 0.0) * 4.0
        + float(signals.get("memory_usage_count") or 0.0) * 3.0
        - float(signals.get("rag_feedback_negative") or 0.0) * 5.0
    )


def _complexity_score(node: dict[str, Any]) -> float:
    degree = int(node.get("degree") or 0)
    return math.log1p(degree) * 10.0


def _bucket(record: dict[str, Any]) -> str:
    signals = record.get("signals") or {}
    positive = float(signals.get("rag_feedback_positive") or 0.0)
    negative = float(signals.get("rag_feedback_negative") or 0.0)
    degree = int(record.get("degree") or 0)
    used = float(record.get("used_score") or 0.0)
    in_degree = int(record.get("in_degree") or 0)
    out_degree = int(record.get("out_degree") or 0)
    if negative > positive and used > 0:
        return "noisy"
    if degree >= 6 and used >= 4:
        return "effective_hub"
    if in_degree >= 2 and out_degree >= 2 and used >= 3:
        return "bridge_candidate"
    if degree >= 6 and used < 3:
        return "complex_but_unproven"
    if degree <= 1 and used >= 3:
        return "isolated_but_used"
    if degree == 0 and used == 0:
        return "dormant_isolate"
    return "ordinary"


def _next_action(bucket: str) -> str:
    return {
        "effective_hub": "判断に効いている中核ノード。関連ノートは増やしすぎず、入口として整える。",
        "bridge_candidate": "複数テーマをつなぐ橋。判断資産や検索語インデックスから明示リンクすると効きやすい。",
        "complex_but_unproven": "リンクは多いが使用痕跡が弱い。リンク整理か、審査時に本当に参照されるか観測する。",
        "isolated_but_used": "孤立しているが使われている。関連ハブへ1-2本だけ接続すると価値が増える。",
        "noisy": "使われているが低評価が勝つ。内容・適用条件・リンク先を見直す。",
        "dormant_isolate": "今は判断に効いた痕跡なし。保存目的がなければ整理候補。",
        "ordinary": "観測継続。",
    }[bucket]


def _answer(records: list[dict[str, Any]]) -> str:
    effective = sum(1 for r in records if r["bucket"] in {"effective_hub", "bridge_candidate", "isolated_but_used"})
    unproven = sum(1 for r in records if r["bucket"] == "complex_but_unproven")
    if effective > unproven:
        return "複雑さは一部判断に効いている。特に使用シグナルのあるハブと橋渡しノードは残す価値が高い。"
    if effective:
        return "複雑さは部分的に効いているが、未検証の絡まりも多い。効いたノードだけを入口化するのがよい。"
    return "現時点では複雑さが判断に効いている証拠は弱い。まず使用シグナルのあるノートへリンクを寄せる。"


def markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Obsidian Graph Judgment Effect",
        "",
        f"- Guardrail: `{report.get('guardrail')}`",
        f"- Answer: {report.get('answer')}",
        f"- Notes: {summary.get('notes', 0)}",
        f"- Edges: {summary.get('edges', 0)}",
        f"- Avg degree: {summary.get('avg_degree', 0)}",
        f"- Notes with usage signal: {summary.get('notes_with_usage_signal', 0)}",
        f"- Buckets: {summary.get('buckets', {})}",
    ]
    sections = [
        ("Effective Hubs", "top_effective_hubs"),
        ("Bridge Candidates", "bridge_candidates"),
        ("Complex But Unproven", "complex_but_unproven"),
        ("Isolated But Used", "isolated_but_used"),
        ("Noisy Nodes", "noisy_nodes"),
    ]
    for title, key in sections:
        lines.extend(["", f"## {title}"])
        for item in report.get(key) or []:
            lines.append(
                f"- `{item.get('path')}` degree={item.get('degree')} used={item.get('used_score')} effect={item.get('judgment_effect_score')} :: {item.get('next_action')}"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--memory-index", type=Path, default=DEFAULT_MEMORY_INDEX)
    parser.add_argument("--memory-usage", type=Path, default=DEFAULT_MEMORY_USAGE)
    parser.add_argument("--rag-search", type=Path, default=DEFAULT_RAG_SEARCH)
    parser.add_argument("--rag-feedback", type=Path, default=DEFAULT_RAG_FEEDBACK)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(
        vault=args.vault,
        memory_index=_read_json(args.memory_index),
        memory_usage_rows=_read_jsonl(args.memory_usage),
        rag_search_rows=_read_jsonl(args.rag_search),
        rag_feedback_rows=_read_jsonl(args.rag_feedback),
    )
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(report), encoding="utf-8")
    print(f"wrote={args.output_json}")
    print(f"wrote={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
