"""Purpose-specific Vertex AI Search workflows for lease knowledge.

Vertex stays a supplementary index. These helpers turn a lease topic into a
bounded search/answer request for evidence support, judgment-asset candidate
finding, or corpus audit.
"""

from __future__ import annotations

from typing import Any, Callable


WORKFLOW_DEFINITIONS: dict[str, dict[str, str]] = {
    "evidence_support": {
        "label": "審査コメント根拠補強",
        "query_suffix": "審査 根拠 確認論点 条件 リスク 出典 稟議 コメント",
        "preamble": (
            "リース審査コメントの根拠補強として、Obsidian由来の整理済み知識だけを使う。"
            "断定せず、根拠、確認論点、追加確認に分けて短く整理する。"
        ),
        "output_hint": "審査コメントへ使う場合は、引用元を確認し、低根拠の主張は確認論点に留める。",
    },
    "judgment_candidates": {
        "label": "判断資産候補探索",
        "query_suffix": "判断資産 確認質問 条件付き承認 反証 リスク 兆候 再利用",
        "preamble": (
            "今回テーマに近い判断資産候補を探す。過去知識をそのまま結論にせず、"
            "今回案件へ転用できる確認質問、承認条件、反証材料として整理する。"
        ),
        "output_hint": "候補は自動採用せず、人間レビュー後に判断資産へ昇格する。",
    },
    "knowledge_audit": {
        "label": "知識棚卸し",
        "query_suffix": "canonical_topic 重複 不足 更新トリガー 審査論点 体系化",
        "preamble": (
            "Vertexに投入済みの整理済み知識を棚卸しする。重複、偏り、不足、"
            "古くなりそうな論点を点検し、次に整えるべき知識テーマを出す。"
        ),
        "output_hint": "監査結果は知識整理の候補であり、審査判断やスコアリングへ直結しない。",
    },
}


def normalize_workflow_mode(mode: str | None) -> str:
    value = (mode or "evidence_support").strip()
    if value not in WORKFLOW_DEFINITIONS:
        return "evidence_support"
    return value


def build_workflow_query(topic: str, mode: str | None) -> str:
    clean_topic = " ".join(str(topic or "").split())[:300]
    workflow = WORKFLOW_DEFINITIONS[normalize_workflow_mode(mode)]
    return f"{clean_topic} {workflow['query_suffix']}".strip()


def _compact(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _refs_from_result(result: dict[str, Any], limit: int = 8) -> list[str]:
    refs: list[str] = []
    for ref in result.get("refs") or []:
        text = _compact(ref, 240)
        if text and text not in refs:
            refs.append(text)
        if len(refs) >= limit:
            return refs
    for item in result.get("results") or result.get("search_results") or []:
        if not isinstance(item, dict):
            continue
        text = _compact(item.get("source_path") or item.get("uri") or item.get("title"), 240)
        if text and text not in refs:
            refs.append(text)
        if len(refs) >= limit:
            return refs
    return refs


def _high_signal_hits(search_result: dict[str, Any], limit: int = 5) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for item in search_result.get("results") or []:
        if not isinstance(item, dict):
            continue
        title = _compact(item.get("title") or item.get("source_path") or item.get("doc_id"), 160)
        source = _compact(item.get("source_path") or item.get("link") or item.get("doc_id"), 220)
        excerpt = _compact(item.get("excerpt"), 360)
        if title or source or excerpt:
            hits.append({"title": title, "source": source, "excerpt": excerpt})
        if len(hits) >= limit:
            break
    return hits


def _workflow_actions(mode: str, search_result: dict[str, Any], answer_result: dict[str, Any]) -> list[str]:
    refs = _refs_from_result(search_result, 5)
    used = bool(search_result.get("used") or answer_result.get("used"))
    if not used:
        return [
            "Vertex側で十分な知識が拾えていない。Obsidianの原本かLocal RAGで確認する。",
            "このテーマをVertex対象にするなら、ResearchまたはJudgment Assetsへ整理済みノートを追加する。",
        ]
    if mode == "judgment_candidates":
        return [
            "上位ヒットから確認質問・承認条件・反証材料だけを候補化する。",
            "候補は自動昇格せず、案件で効いたかを人間レビューしてから判断資産にする。",
            f"優先確認元: {refs[0] if refs else 'Vertex refs'}",
        ]
    if mode == "knowledge_audit":
        return [
            "同じ論点のヒットが多い場合はcanonical topicを統合する。",
            "重要テーマなのにヒットが薄い場合はResearchノートを作る。",
            "古くなりやすい制度・補助金・業界動向は更新日と確認先を追記する。",
        ]
    return [
        "審査コメントに使う前に、上位refの原文を確認する。",
        "低根拠の内容は断定せず、追加確認事項として残す。",
        f"根拠候補: {refs[0] if refs else 'Vertex refs'}",
    ]


def run_vertex_knowledge_workflow(
    topic: str,
    *,
    mode: str | None = None,
    page_size: int = 5,
    search_fn: Callable[..., dict[str, Any]] | None = None,
    answer_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    clean_topic = _compact(topic, 300)
    if not clean_topic:
        raise ValueError("topic is required")
    workflow_mode = normalize_workflow_mode(mode)
    workflow = WORKFLOW_DEFINITIONS[workflow_mode]
    query = build_workflow_query(clean_topic, workflow_mode)

    if search_fn is None or answer_fn is None:
        from api.vertex_agent_search import answer_vertex_agent, search_vertex_agent

        search_fn = search_fn or search_vertex_agent
        answer_fn = answer_fn or answer_vertex_agent

    search_result = search_fn(query, page_size=page_size, apply_controls=True)
    answer_result = answer_fn(
        query,
        page_size=page_size,
        preamble=workflow["preamble"],
        include_related_questions=workflow_mode != "knowledge_audit",
        include_grounding_supports=True,
        max_rephrase_steps=3,
    )

    refs = _refs_from_result(search_result) + [
        ref for ref in _refs_from_result(answer_result) if ref not in _refs_from_result(search_result)
    ]
    return {
        "mode": workflow_mode,
        "label": workflow["label"],
        "topic": clean_topic,
        "query": query,
        "guardrail": "Vertexは補助索引。Local RAG/Obsidian原本と人間レビューを主判断にする。",
        "output_hint": workflow["output_hint"],
        "search": search_result,
        "answer": answer_result,
        "high_signal_hits": _high_signal_hits(search_result),
        "refs": refs[:8],
        "next_actions": _workflow_actions(workflow_mode, search_result, answer_result),
    }
