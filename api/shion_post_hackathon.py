"""Post-hackathon Shion control-plane helpers.

These helpers turn the backlog in planning/post_hackathon_shion_backlog.md into
observable, low-risk runtime surfaces. They do not alter live chat prompts,
scoring, RAG indexes, or canonical judgment assets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from api.knowledge.shion_hyde import build_combined_search_query, build_shion_hyde_query

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _exists(rel_path: str) -> bool:
    return (_REPO_ROOT / rel_path).exists()


def build_post_hackathon_status() -> dict[str, Any]:
    """Return implementation status for the post-hackathon backlog."""
    sections = [
        {
            "id": "relationship_control",
            "title": "関係性を制御対象にする",
            "status": "implemented",
            "evidence": [
                "api/routers/relationship.py",
                "api/shion_relationship.py",
            ],
            "runtime_surface": "/api/relationship/state",
            "next_guardrail": "回答制御へ直結せず、フィードバック観測と状態表示に留める",
        },
        {
            "id": "practical_ai_finish",
            "title": "奥行きのある実務AIとして締める",
            "status": "implemented",
            "evidence": [
                "frontend/src/app/shion-memory-system/page.tsx",
                "frontend/src/lib/shionReview.ts",
            ],
            "runtime_surface": "/shion-memory-system",
            "next_guardrail": "人格・内部設計は表へ出しすぎず、業務判断の短い確認点へ変換する",
        },
        {
            "id": "judgment_asset_ops",
            "title": "判断資産の実戦運用",
            "status": "implemented",
            "evidence": [
                "frontend/src/app/judgment-review/page.tsx",
                "api/routers/feedback_loop.py",
            ],
            "runtime_surface": "/judgment-review",
            "next_guardrail": "採用・保留・却下と、事後の効果測定を分離して追う",
        },
        {
            "id": "shion_hyde_rag",
            "title": "Shion-HyDE RAG / 検索改善",
            "status": "debug_ready",
            "evidence": [
                "api/knowledge/shion_hyde.py",
                "tests/test_shion_hyde.py",
            ],
            "runtime_surface": "/api/debug/shion-hyde-rag",
            "next_guardrail": "本回答には注入せず、既存検索との差分比較だけを返す",
        },
        {
            "id": "memory_health",
            "title": "記憶の健康診断",
            "status": "implemented",
            "evidence": [
                "scripts/check_shion_memory_health.py",
                "frontend/src/app/shion-memory-system/page.tsx",
            ],
            "runtime_surface": "/api/shion/memory-health",
            "next_guardrail": "記憶削除・昇格は行わず、急減・混在・レビュー待ちだけを見る",
        },
        {
            "id": "response_impact",
            "title": "Response Impact Predictor の検証",
            "status": "observability_only",
            "evidence": [
                "api/routers/feedback_loop.py",
                "api/shion_experience_loop.py",
            ],
            "runtime_surface": "/api/shion/action-ledger/summary",
            "next_guardrail": "回答本文の自動書き換えには使わず、反応分類とLedger観測に留める",
        },
        {
            "id": "ai_reviewer_strategy",
            "title": "対AI審査戦略の振り返り",
            "status": "implemented",
            "evidence": [
                "docs/hackathon_submission_copy.md",
                "reports/hackathon_final_review_checklist.md",
            ],
            "runtime_surface": "/system-overview",
            "next_guardrail": "人間向け説明は実務AI、AI向け痕跡は内部構造で読ませる",
        },
        {
            "id": "counter_hypothesis_training",
            "title": "違和感抽出 / 人間訓練モード",
            "status": "training_ready",
            "evidence": [
                "api/shion_post_hackathon.py",
            ],
            "runtime_surface": "/api/shion/counter-hypothesis-card",
            "next_guardrail": "通常回答には混ぜず、反証用の次善仮説カードとしてだけ使う",
        },
        {
            "id": "autonomous_devops_transition",
            "title": "完全自律DevOpsへ向けた段階移行",
            "status": "implemented",
            "evidence": [
                "api/shion_action_ledger.py",
                "tests/test_shion_action_ledger.py",
                "api/routers/improvement.py",
            ],
            "runtime_surface": "/api/shion/action-ledger/summary",
            "next_guardrail": "低リスク変更でも、実行権限と監査ログを分離する",
        },
    ]
    for section in sections:
        section["evidence_available"] = [
            path for path in section["evidence"] if _exists(path)
        ]
        section["complete"] = len(section["evidence_available"]) == len(section["evidence"])
    return {
        "source": "planning/post_hackathon_shion_backlog.md",
        "mode": "post_hackathon_control_plane",
        "total": len(sections),
        "complete_count": sum(1 for section in sections if section["complete"]),
        "sections": sections,
    }


def build_memory_health_payload(
    *,
    index_path: Path | None = None,
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Expose memory health check as a read-only API payload."""
    from scripts.check_shion_memory_health import (
        DEFAULT_INDEX,
        DEFAULT_STATE,
        check,
        load_index_summary,
        load_state,
    )

    index = index_path or DEFAULT_INDEX
    state = state_path or DEFAULT_STATE
    summary = load_index_summary(index)
    previous = load_state(state)
    healthy, message = check(summary, previous)
    return {
        "healthy": healthy,
        "message": message,
        "index_path": str(index),
        "state_path": str(state),
        "summary": summary or {},
        "previous": previous,
        "guardrail": "read_only_no_memory_delete_no_promotion",
    }


def _compact_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(hit.get("path") or hit.get("title") or ""),
        "title": str(hit.get("title") or Path(str(hit.get("path") or "")).stem),
        "score": hit.get("score") or hit.get("semantic_score"),
        "snippet": str(hit.get("snippet") or hit.get("content") or "")[:280],
        "source_type": hit.get("source_type") or hit.get("source") or "",
    }


def build_shion_hyde_debug_payload(
    *,
    message: str,
    industry: str = "",
    asset_name: str = "",
    score_band: str = "",
    known_risk_tags: list[str] | None = None,
    limit: int = 5,
    search_fn: Callable[[str, int], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Compare existing retrieval with Shion-HyDE retrieval in debug mode."""
    query = build_shion_hyde_query(
        message,
        industry=industry,
        asset_name=asset_name,
        score_band=score_band,
        known_risk_tags=known_risk_tags or [],
    )
    capped_limit = max(1, min(int(limit or 5), 10))
    if search_fn is None:
        from mobile_app.obsidian_bridge import search_notes

        def search_fn(q: str, n: int) -> list[dict[str, Any]]:  # type: ignore[no-redef]
            return list(search_notes(q, limit=n, max_chars=360))

    baseline_hits = search_fn(message, capped_limit) if message.strip() else []
    combined_query = build_combined_search_query(query)
    hyde_hits = search_fn(combined_query, capped_limit) if query.should_search else []
    baseline_paths = {str(hit.get("path") or "") for hit in baseline_hits if hit.get("path")}
    hyde_paths = {str(hit.get("path") or "") for hit in hyde_hits if hit.get("path")}
    new_paths = sorted(hyde_paths - baseline_paths)
    confidence = 0.0
    if query.should_search:
        confidence = min(0.95, 0.35 + 0.08 * len(hyde_hits) + 0.06 * len(query.intent_tags))
    return {
        "mode": "debug_only_no_prompt_injection",
        "hyde": query.to_dict(),
        "combined_query": combined_query,
        "baseline_hits": [_compact_hit(hit) for hit in baseline_hits[:capped_limit]],
        "hyde_hits": [_compact_hit(hit) for hit in hyde_hits[:capped_limit]],
        "new_paths": new_paths[:capped_limit],
        "confidence": round(confidence, 3),
        "fallback_reason": "" if query.should_search else query.reason,
        "guardrail": "does_not_modify_live_chat_rag_or_index",
    }


def build_counter_hypothesis_card(
    *,
    case_summary: str,
    industry: str = "",
    asset_name: str = "",
    score_band: str = "",
    human_discomfort: str = "",
    risk_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Build a training-only second-best hypothesis card.

    The card is deliberately framed as a refutable hypothesis, not as a final
    answer or a purposely wrong answer.
    """
    risk_tags = [str(tag).strip() for tag in (risk_tags or []) if str(tag).strip()]
    theme = "返済原資・物件換金性・商流のどれかに確認不足がある"
    if any("競合" in tag or "失注" in tag for tag in risk_tags):
        theme = "競合条件と案件化経緯の見落としがある"
    elif any("補助" in tag for tag in risk_tags):
        theme = "補助金未採択時の代替返済原資が弱い"
    elif "新規" in case_summary or "初取引" in case_summary:
        theme = "新規先として外部信用と代表者経験の確認が不足している"
    elif "中古" in case_summary or "残価" in case_summary or asset_name:
        theme = "物件の換金性を押し材料にしすぎている可能性がある"

    card = (
        f"{industry or '対象業種'}の{asset_name or '設備'}案件は、表面スコアだけなら進められる。"
        f"ただし次善仮説としては「{theme}」を置く。"
        "この仮説が外れているなら、何を見れば外れたと言えるかを人間が反証する。"
    )
    if score_band:
        card += f" スコア帯は{score_band}として扱う。"
    if human_discomfort:
        card += f" 人間の違和感メモ: {human_discomfort[:120]}"
    questions = [
        "この仮説はどこが浅いか。",
        "今回の案件で本当に見るべき条件は何か。",
        "稟議に一文だけ残すなら、どの条件を残すか。",
    ]
    return {
        "mode": "counter_hypothesis_training_only",
        "friction_level": 0.68,
        "card": card,
        "questions": questions,
        "judgment_asset_template": {
            "condition": "",
            "review_point": "",
            "ringi_sentence": "",
            "do_not_use_when": "",
        },
        "metrics_to_watch": [
            "human_refuted",
            "specific_correction_added",
            "new_review_point_added",
            "converted_to_judgment_asset_candidate",
            "user_fatigue",
        ],
        "guardrail": "never_mix_into_normal_answer",
    }
