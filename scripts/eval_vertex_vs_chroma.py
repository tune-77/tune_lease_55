#!/usr/bin/env python3
"""毎晩、ローカル RAG（ChromaDB）と Vertex AI Search の検索・回答品質を同じ評価セットで比べる。

評価セット: api/knowledge/rag_eval_set.json（30問）＋ okf_rag_eval_set.json（12問）。正解は expected_path_any。
- chroma: ChromaDB 上位5件（本番の主経路）
- rerank: ChromaDB 上位10件を Ranking API で並べ替えた上位5件（shadow。本番には効いていない）
- vertex_search: Vertex AI Search 上位5件
- vertex_answer: Answer API の参照に正解が入るか（回答の根拠の当たり率）と grounding score の平均

並べ替えの本番適用（data/vertex_credit_state.json の rerank.promoted）:
- 直近3晩すべてで rerank の hit@5 が chroma 以上、かつ3晩平均の MRR が chroma より +0.02 以上 → 適用
- 適用中でも、最新の晩に rerank の hit@5 か MRR（−0.02超）が chroma を下回れば → 外す
クエリは vertex_agent_search 側で mask_for_vertex を通る。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.vertex_credit_mode import load_state  # noqa: E402
from scripts.evaluate_obsidian_rag import _display_path, evaluate_cases  # noqa: E402
from scripts.vertex_credit_monitor import write_state  # noqa: E402

EVAL_SETS = [ROOT / "api" / "knowledge" / "rag_eval_set.json", ROOT / "api" / "knowledge" / "okf_rag_eval_set.json"]
DEFAULT_REPORT = ROOT / "reports" / "vertex_rag_eval_latest.json"
TOP_K = 5
PROMOTE_NIGHTS = 3
MRR_MARGIN = 0.02
SearchFn = Callable[[str, int], list[dict[str, Any]]]


def load_cases(paths: list[Path] = EVAL_SETS) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in paths:
        cases.extend(json.loads(path.read_text(encoding="utf-8")))
    return cases


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {k: result[k] for k in ("total", "hit_at_1_rate", "hit_at_k_rate", "mrr", "forbidden_cases")}


def _path_hits(paths: list[str]) -> list[dict[str, Any]]:
    return [{"file_path": p} for p in paths if p]


def run_eval(
    cases: list[dict[str, Any]],
    *,
    chroma_search: SearchFn,
    rerank: Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]],
    vertex_search: SearchFn,
    vertex_answer: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    chroma_cache: dict[str, list[dict[str, Any]]] = {}

    def chroma10(query: str) -> list[dict[str, Any]]:
        if query not in chroma_cache:
            chroma_cache[query] = chroma_search(query, TOP_K * 2)
        return chroma_cache[query]

    def reranked(query: str, k: int) -> list[dict[str, Any]]:
        hits = chroma10(query)
        try:
            return rerank(query, hits)[:k]
        except Exception:  # noqa: BLE001 - 並べ替え失敗は元の順位（＝chroma と同点）として数える
            return hits[:k]

    answer_hits = 0
    grounding: list[float] = []
    answer_cases: list[dict[str, Any]] = []
    for case in cases:
        result = vertex_answer(case["query"])
        refs = [str(r) for r in result.get("refs") or []]
        expected = list(case.get("expected_path_any") or [])
        hit = any(p and p in ref for ref in refs for p in expected)
        answer_hits += int(hit)
        if result.get("grounding_score") is not None:
            grounding.append(float(result["grounding_score"]))
        answer_cases.append({"id": case["id"], "hit": hit, "refs": refs[:5], "status": result.get("status")})

    total = len(cases)
    return {
        "chroma": _summary(evaluate_cases(cases, lambda q, k: chroma10(q)[:k], TOP_K)),
        "rerank_shadow": _summary(evaluate_cases(cases, reranked, TOP_K)),
        "vertex_search": _summary(evaluate_cases(cases, vertex_search, TOP_K)),
        "vertex_answer": {
            "total": total,
            "ref_hit_rate": answer_hits / total if total else 0.0,
            "grounding_score_mean": round(sum(grounding) / len(grounding), 3) if grounding else None,
            "cases": answer_cases,
        },
    }


def decide_rerank(history: list[dict[str, Any]], currently_promoted: bool) -> tuple[bool, str]:
    """history は古い順の各晩 {"chroma": {...}, "rerank_shadow": {...}}。"""
    if not history:
        return currently_promoted, "評価なし"
    latest = history[-1]
    c, r = latest["chroma"], latest["rerank_shadow"]
    if currently_promoted:
        if r["hit_at_k_rate"] < c["hit_at_k_rate"] or r["mrr"] < c["mrr"] - MRR_MARGIN:
            return False, f"最新の晩で並べ替えが劣後（hit@5 {r['hit_at_k_rate']:.0%} vs {c['hit_at_k_rate']:.0%}, MRR {r['mrr']:.3f} vs {c['mrr']:.3f}）"
        return True, "効果継続"
    recent = history[-PROMOTE_NIGHTS:]
    if len(recent) < PROMOTE_NIGHTS:
        return False, f"評価{len(recent)}/{PROMOTE_NIGHTS}晩（判定待ち）"
    if any(n["rerank_shadow"]["hit_at_k_rate"] < n["chroma"]["hit_at_k_rate"] for n in recent):
        return False, "hit@5 が chroma を下回った晩がある"
    gain = sum(n["rerank_shadow"]["mrr"] - n["chroma"]["mrr"] for n in recent) / len(recent)
    if gain < MRR_MARGIN:
        return False, f"MRR の改善 {gain:+.3f} が基準 +{MRR_MARGIN} 未満"
    return True, f"{PROMOTE_NIGHTS}晩連続で hit@5 同等以上・MRR {gain:+.3f}"


def _default_fns() -> dict[str, Any]:
    from api.knowledge.vector_store import get_store
    from api.vertex_agent_search import answer_vertex_agent, rerank_hits, search_vertex_agent

    store = get_store()

    def vertex_search(query: str, k: int) -> list[dict[str, Any]]:
        return _path_hits([str(r) for r in search_vertex_agent(query, page_size=k).get("refs") or []])[:k]

    return {
        "chroma_search": lambda q, k: store.search(q, top_k=k),
        "rerank": rerank_hits,
        "vertex_search": vertex_search,
        "vertex_answer": lambda q: answer_vertex_agent(q, page_size=5, include_grounding_supports=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--state", type=Path, default=None)
    args = parser.parse_args()

    result = run_eval(load_cases(), **_default_fns())
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    state = load_state(args.state)
    rerank_state = dict(state.get("rerank") or {})
    history = list(rerank_state.get("history") or [])
    history.append({"at": now, "chroma": result["chroma"], "rerank_shadow": result["rerank_shadow"]})
    promoted, reason = decide_rerank(history, bool(rerank_state.get("promoted")))
    if promoted != bool(rerank_state.get("promoted")):
        rerank_state["changed_at"] = now
    rerank_state.update({"promoted": promoted, "reason": reason, "history": history[-14:]})
    state["rerank"] = rerank_state
    state["eval_latest"] = {
        "at": now,
        **{k: v for k, v in result.items() if k != "vertex_answer"},
        "vertex_answer": {k: v for k, v in result["vertex_answer"].items() if k != "cases"},
    }
    write_state(state, args.state)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"generated_at": now, **result, "rerank_decision": {"promoted": promoted, "reason": reason}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(morning_line(state))
    return 0


def morning_line(state: dict[str, Any]) -> str:
    latest = state.get("eval_latest") or {}
    if not latest:
        return "- 🔎 Vertex 品質比較: まだ評価なし"
    rerank = state.get("rerank") or {}
    pct = lambda block: f"{(latest.get(block) or {}).get('hit_at_k_rate', 0):.0%}"  # noqa: E731
    answer = latest.get("vertex_answer") or {}
    grounding = answer.get("grounding_score_mean")
    return (
        f"- 🔎 Vertex 品質比較（{str(latest.get('at', ''))[:10]}・hit@5）: ChromaDB {pct('chroma')} / "
        f"Vertex検索 {pct('vertex_search')} / 並べ替え案 {pct('rerank_shadow')}"
        f"（本番適用: {'あり' if rerank.get('promoted') else 'なし'}・{rerank.get('reason', '')}）/ "
        f"Answer根拠ヒット {answer.get('ref_hit_rate', 0):.0%}"
        + (f"・grounding {grounding}" if grounding is not None else "")
    )


if __name__ == "__main__":
    raise SystemExit(main())
