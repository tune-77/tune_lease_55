#!/usr/bin/env python3
"""毎晩、ローカル RAG（ChromaDB）と Vertex AI Search の検索・回答品質を同じ評価セットで比べる。

評価セット: api/knowledge/rag_eval_set.json（30問）＋ okf_rag_eval_set.json（12問）。正解は expected_path_any。
- chroma: ChromaDB 上位5件（本番の主経路）
- rerank: ChromaDB 上位10件を Ranking API で並べ替えた上位5件（shadow。本番には効いていない）
- vertex_search: Vertex AI Search 上位5件
- vertex_answer: Answer API の参照に正解が入るか（回答の根拠の当たり率）と grounding score の平均
- jev_rerank: ChromaDB 上位10件を Jev の「この質問に答えるのに役立つか」の確率で並べ替えた上位5件（shadow）。
  採点は他と同じく正解資料の順位だけ（Jev の自己採点は使わない）。Jev 不通の問題は ChromaDB の順位のまま数える
- vertex_covered: Vertex のコーパスに正解資料がある問題だけでの比較（公平性のための別指標）

並べ替えの本番適用（Vertex: data/vertex_credit_state.json の rerank.promoted、
Jev: data/jev_rag_rerank_state.json の promoted。条件は同じで、それぞれ独立に判定する）:
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
DEFAULT_JEV_REPORT = ROOT / "reports" / "jev_rag_eval_latest.json"
TOP_K = 5
PROMOTE_NIGHTS = 3
MRR_MARGIN = 0.02
SearchFn = Callable[[str, int], list[dict[str, Any]]]


def one_entry_per_date(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the latest retry for each calendar date, preserving date order."""
    by_date: dict[str, dict[str, Any]] = {}
    undated = 0
    for entry in history:
        date_key = str(entry.get("at") or "")[:10]
        if not date_key:
            date_key = f"undated-{undated}"
            undated += 1
        by_date[date_key] = entry
    return list(by_date.values())


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
    jev_rerank: Callable[[str, list[dict[str, Any]]], tuple[list[dict[str, Any]], dict[str, Any]]] | None = None,
    jev_eligible: Callable[[str], bool] | None = None,
    covered_ids: set[str] | None = None,
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

    jev_stats = {"calls": 0, "input_tokens": 0, "failures": 0}

    def jev_reranked(query: str, k: int) -> list[dict[str, Any]]:
        hits = chroma10(query)
        if jev_rerank is None:
            return hits[:k]
        try:
            ordered, meta = jev_rerank(query, hits)
        except Exception:  # noqa: BLE001 - Jev 不通は ChromaDB の順位のまま
            jev_stats["failures"] += 1
            return hits[:k]
        jev_stats["calls"] += int(meta.get("calls") or 0)
        jev_stats["input_tokens"] += int(meta.get("input_tokens") or 0)
        return ordered[:k]

    answer_hits = 0
    answer_hit_ids: set[str] = set()
    grounding: list[float] = []
    answer_cases: list[dict[str, Any]] = []
    for case in cases:
        result = vertex_answer(case["query"])
        refs = [str(r) for r in result.get("refs") or []]
        expected = list(case.get("expected_path_any") or [])
        hit = any(p and p in ref for ref in refs for p in expected)
        answer_hits += int(hit)
        if hit:
            answer_hit_ids.add(case["id"])
        if result.get("grounding_score") is not None:
            grounding.append(float(result["grounding_score"]))
        answer_cases.append({"id": case["id"], "hit": hit, "refs": refs[:5], "status": result.get("status")})

    total = len(cases)
    covered = [c for c in cases if c["id"] in (covered_ids or set())]
    covered_block = {
        "total": len(covered),
        "chroma": _summary(evaluate_cases(covered, lambda q, k: chroma10(q)[:k], TOP_K)),
        "vertex_search": _summary(evaluate_cases(covered, vertex_search, TOP_K)),
        "vertex_answer_ref_hit_rate": (sum(1 for c in covered if c["id"] in answer_hit_ids) / len(covered)) if covered else 0.0,
    }
    jev_cases = [case for case in cases if jev_eligible is None or jev_eligible(case["query"])]
    jev_chroma = _summary(evaluate_cases(jev_cases, lambda q, k: chroma10(q)[:k], TOP_K)) if jev_rerank is not None else None
    jev_block = _summary(evaluate_cases(jev_cases, jev_reranked, TOP_K)) if jev_rerank is not None else None
    if jev_block is not None:
        jev_block.update({**jev_stats, "estimated_jpy": jev_estimate_jpy(jev_stats["input_tokens"])})
    return {
        "chroma": _summary(evaluate_cases(cases, lambda q, k: chroma10(q)[:k], TOP_K)),
        "rerank_shadow": _summary(evaluate_cases(cases, reranked, TOP_K)),
        "jev_rerank": jev_block,
        "jev_chroma": jev_chroma,
        "vertex_search": _summary(evaluate_cases(cases, vertex_search, TOP_K)),
        "vertex_covered": covered_block,
        "vertex_answer": {
            "total": total,
            "ref_hit_rate": answer_hits / total if total else 0.0,
            "grounding_score_mean": round(sum(grounding) / len(grounding), 3) if grounding else None,
            "cases": answer_cases,
        },
    }


def decide_rerank(history: list[dict[str, Any]], currently_promoted: bool, key: str = "rerank_shadow") -> tuple[bool, str]:
    """history は古い順の各晩 {"chroma": {...}, key: {...}}。key は rerank_shadow（Vertex）か jev_rerank。"""
    history = one_entry_per_date([n for n in history if n.get(key)])
    if not history:
        return currently_promoted, "評価なし"
    latest = history[-1]
    c, r = latest["chroma"], latest[key]
    if currently_promoted:
        if r["hit_at_k_rate"] < c["hit_at_k_rate"] or r["mrr"] < c["mrr"] - MRR_MARGIN:
            return False, f"最新の晩で並べ替えが劣後（hit@5 {r['hit_at_k_rate']:.0%} vs {c['hit_at_k_rate']:.0%}, MRR {r['mrr']:.3f} vs {c['mrr']:.3f}）"
        return True, "効果継続"
    recent = history[-PROMOTE_NIGHTS:]
    if len(recent) < PROMOTE_NIGHTS:
        return False, f"評価{len(recent)}/{PROMOTE_NIGHTS}晩（判定待ち）"
    if any(n[key]["hit_at_k_rate"] < n["chroma"]["hit_at_k_rate"] for n in recent):
        return False, "hit@5 が chroma を下回った晩がある"
    gain = sum(n[key]["mrr"] - n["chroma"]["mrr"] for n in recent) / len(recent)
    if gain < MRR_MARGIN:
        return False, f"MRR の改善 {gain:+.3f} が基準 +{MRR_MARGIN} 未満"
    return True, f"{PROMOTE_NIGHTS}晩連続で hit@5 同等以上・MRR {gain:+.3f}"


def jev_estimate_jpy(input_tokens: int) -> float:
    from api.jev_rag_rerank import estimate_jpy

    return estimate_jpy(input_tokens)


EXPORT_MANIFEST = ROOT / "data" / "agent_search" / "lease_knowledge_export" / "manifest.json"


def vertex_covered_ids(cases: list[dict[str, Any]], manifest: Path = EXPORT_MANIFEST) -> set[str]:
    """Vertex に同期したコーパスに正解資料がある問題の id。"""
    try:
        paths = [str(d.get("source_path") or "") for d in json.loads(manifest.read_text(encoding="utf-8")).get("documents") or []]
    except (OSError, json.JSONDecodeError):
        return set()
    return {c["id"] for c in cases if any(p and p in sp for p in c.get("expected_path_any") or [] for sp in paths)}


def _default_fns() -> dict[str, Any]:
    from api.chat_retrieval import _jev_evaluation_eligible
    from api.jev_rag_rerank import jev_rerank
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
        "jev_rerank": jev_rerank,
        "jev_eligible": _jev_evaluation_eligible,
    }


def _default_jev_only_fns() -> dict[str, Any]:
    """Dependencies for Jev shadow evaluation without importing/calling Vertex."""
    from api.chat_retrieval import _jev_evaluation_eligible
    from api.jev_rag_rerank import jev_rerank
    from api.knowledge.vector_store import get_store

    store = get_store()
    return {
        "chroma_search": lambda q, k: store.search(q, top_k=k),
        "rerank": lambda _q, hits: hits,
        "vertex_search": lambda _q, _k: [],
        "vertex_answer": lambda _q: {"status": "skipped", "refs": []},
        "jev_rerank": jev_rerank,
        "jev_eligible": _jev_evaluation_eligible,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--jev-report", type=Path, default=DEFAULT_JEV_REPORT)
    parser.add_argument(
        "--jev-only",
        action="store_true",
        help="Evaluate Chroma vs Jev only; do not import or call Vertex clients.",
    )
    parser.add_argument("--state", type=Path, default=None)
    args = parser.parse_args()

    cases = load_cases()
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    if args.jev_only:
        result = run_eval(cases, covered_ids=set(), **_default_jev_only_fns())
        promoted, reason = update_jev_state(result, now)
        args.jev_report.parent.mkdir(parents=True, exist_ok=True)
        args.jev_report.write_text(
            json.dumps(
                {
                    "generated_at": now,
                    "mode": "jev_only",
                    "chroma": result.get("jev_chroma") or result["chroma"],
                    "jev_rerank": result["jev_rerank"],
                    "jev_rerank_decision": {"promoted": promoted, "reason": reason},
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"Jev shadow: {'promoted' if promoted else 'not promoted'} / {reason} / "
            f"calls={(result.get('jev_rerank') or {}).get('calls', 0)}"
        )
        return 0

    result = run_eval(cases, covered_ids=vertex_covered_ids(cases), **_default_fns())
    state = load_state(args.state)
    rerank_state = dict(state.get("rerank") or {})
    history = list(rerank_state.get("history") or [])
    history.append({"at": now, "chroma": result["chroma"], "rerank_shadow": result["rerank_shadow"]})
    history = one_entry_per_date(history)
    promoted, reason = decide_rerank(history, bool(rerank_state.get("promoted")))
    if promoted != bool(rerank_state.get("promoted")):
        rerank_state["changed_at"] = now
    rerank_state.update({"promoted": promoted, "reason": reason, "history": history[-14:]})
    state["rerank"] = rerank_state
    jev_promoted, jev_reason = update_jev_state(result, now)
    state["eval_latest"] = {
        "at": now,
        **{k: v for k, v in result.items() if k != "vertex_answer"},
        "vertex_answer": {k: v for k, v in result["vertex_answer"].items() if k != "cases"},
    }
    write_state(state, args.state)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"generated_at": now, **result, "rerank_decision": {"promoted": promoted, "reason": reason}, "jev_rerank_decision": {"promoted": jev_promoted, "reason": jev_reason}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(morning_line(state))
    return 0


def update_jev_state(result: dict[str, Any], now: str, path: Path | None = None) -> tuple[bool, str]:
    """Jev 並べ替えの昇格判定を、Vertex とは別の状態ファイルに保存する。"""
    from api.jev_rag_rerank import STATE_PATH, load_state as load_jev_state

    path = path or STATE_PATH
    jev_state = load_jev_state(path)
    jev = result.get("jev_rerank")
    if not jev:
        return bool(jev_state.get("promoted")), "Jev 評価なし"
    if not jev.get("calls"):
        # 全問不通を「Chromaと同等」とみなさない。本番適用中なら待ち時間を増やし続けないよう即時解除する。
        was_promoted = bool(jev_state.get("promoted"))
        reason = f"Jev 全問不通のため本番適用を解除（{jev.get('failures', 0)}問）" if was_promoted else f"Jev 全問不通のため今晩は判定に数えない（{jev.get('failures', 0)}問）"
        jev_state.update({"promoted": False, "reason": reason, "last_failed_at": now, "last_failures": jev.get("failures", 0)})
        if was_promoted:
            jev_state["changed_at"] = now
        write_state(jev_state, path)
        return False, reason
    history = list(jev_state.get("history") or [])
    history.append({"at": now, "chroma": result.get("jev_chroma") or result["chroma"], "jev_rerank": result["jev_rerank"]})
    history = one_entry_per_date(history)
    promoted, reason = decide_rerank(history, bool(jev_state.get("promoted")), key="jev_rerank")
    if promoted != bool(jev_state.get("promoted")):
        jev_state["changed_at"] = now
    jev_state.update({"promoted": promoted, "reason": reason, "history": history[-14:]})
    write_state(jev_state, path)
    return promoted, reason


def morning_line(state: dict[str, Any]) -> str:
    latest = state.get("eval_latest") or {}
    if not latest:
        return "- 🔎 Vertex 品質比較: まだ評価なし"
    rerank = state.get("rerank") or {}
    pct = lambda block: f"{(latest.get(block) or {}).get('hit_at_k_rate', 0):.0%}"  # noqa: E731
    answer = latest.get("vertex_answer") or {}
    grounding = answer.get("grounding_score_mean")
    line = (
        f"- 🔎 Vertex 品質比較（{str(latest.get('at', ''))[:10]}・hit@5）: ChromaDB {pct('chroma')} / "
        f"Vertex検索 {pct('vertex_search')} / 並べ替え案 {pct('rerank_shadow')}"
        f"（本番適用: {'あり' if rerank.get('promoted') else 'なし'}・{rerank.get('reason', '')}）/ "
        f"Answer根拠ヒット {answer.get('ref_hit_rate', 0):.0%}"
        + (f"・grounding {grounding}" if grounding is not None else "")
    )
    covered = latest.get("vertex_covered") or {}
    if covered.get("total"):
        line += (
            f" ／ Vertexに正解がある{covered['total']}問: ChromaDB {covered['chroma']['hit_at_k_rate']:.0%}"
            f"・Vertex検索 {covered['vertex_search']['hit_at_k_rate']:.0%}・Answer {covered['vertex_answer_ref_hit_rate']:.0%}"
        )
    jev = latest.get("jev_rerank")
    if jev:
        from api.jev_rag_rerank import load_state as load_jev_state

        jev_state = load_jev_state()
        line += (
            f" ／ Jev並べ替え {jev['hit_at_k_rate']:.0%}（本番適用: {'あり' if jev_state.get('promoted') else 'なし'}・"
            f"{jev_state.get('reason', '')}）Jev {jev.get('calls', 0)}回・不通{jev.get('failures', 0)}・"
            f"推定¥{float(jev.get('estimated_jpy') or 0):.2f}（入力トークン課金のみ・クレジット対象外）"
        )
    return line


if __name__ == "__main__":
    raise SystemExit(main())
