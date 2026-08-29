"""Agentic skill tools for Shion's ADK agent.

These functions expose project-local skill workflows as ADK-callable tools.
They are intentionally deterministic and side-effect free: no Obsidian writes,
no RAG promotion, no score changes, and no external API calls.
"""

from __future__ import annotations

import re
from typing import Any

from api.agentic_skill_usage import check_agentic_skill_flow, load_agentic_skill_review_inbox


_RISK_ORIGIN_KEYWORDS = {
    "credit_repayment": ("返済", "資金繰", "赤字", "債務", "借入", "利益", "売上", "DSCR", "cash", "payment"),
    "asset_value": ("物件", "資産", "残価", "中古", "耐用", "陳腐化", "設備", "asset", "resale"),
    "sales_competition": ("競合", "相見積", "成約", "営業", "価格", "他社", "competition", "quote"),
    "documentation_compliance": ("契約", "書類", "法令", "補助金", "許認可", "compliance", "regulation"),
    "input_model_consistency": ("入力", "スコア", "矛盾", "モデル", "Q_risk", "異常", "score"),
}


def _clean_text(text: str | None, *, maximum: int = 4000) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value[:maximum]


def _sentences(text: str, *, limit: int = 5) -> list[str]:
    parts = [p.strip(" ・-　") for p in re.split(r"(?<=[。.!?？])\s+|[。\n]", text) if p.strip()]
    return parts[:limit]


def _infer_risk_origins(text: str) -> list[str]:
    lowered = text.lower()
    origins: list[str] = []
    for origin, keywords in _RISK_ORIGIN_KEYWORDS.items():
        if any(str(keyword).lower() in lowered for keyword in keywords):
            origins.append(origin)
    return origins or ["needs_human_classification"]


def _confidence_from_source(source_type: str, evidence: str = "") -> str:
    source = str(source_type or "").lower()
    if source in {"user_judgment", "success_signal", "failure_signal"}:
        return "medium"
    if source == "case_observation" and evidence:
        return "medium"
    return "low"


def structure_judgment_asset_candidate(
    note: str,
    source_type: str = "user_judgment",
    title: str = "",
) -> dict[str, Any]:
    """Structure a note into a reusable lease judgment asset candidate.

    Args:
        note: Case note, chat excerpt, user correction, or observation to structure.
        source_type: case_observation, user_judgment, failure_signal,
            success_signal, or research_signal.
        title: Optional short title for the candidate.

    Returns:
        A review-gated judgment asset candidate. This does not promote memory,
        edit Obsidian, or alter scoring.
    """
    cleaned = _clean_text(note)
    facts = _sentences(cleaned, limit=4)
    origins = _infer_risk_origins(cleaned)
    source = source_type if source_type in {
        "case_observation",
        "user_judgment",
        "failure_signal",
        "success_signal",
        "research_signal",
    } else "user_judgment"
    asset_type = "risk_signal" if "risk" in source or "failure" in source else "judgment_rule"
    if source == "research_signal":
        asset_type = "confirmation_question"

    return {
        "mode": "agentic_skill_judgment_asset_structurer",
        "status": "candidate",
        "candidate": {
            "title": title.strip() or (facts[0][:40] if facts else "判断資産候補"),
            "asset_type": asset_type,
            "source_type": source,
            "core_judgment": facts[0] if facts else cleaned,
            "when_to_apply": f"同種のリスク起点がある案件: {', '.join(origins)}",
            "do_not_apply_when": "案件事実・人間レビュー・結果検証で前提が崩れる場合",
            "case_signals": origins,
            "recommended_action": "次回審査で確認質問または条件文に変換して使う",
            "evidence_needed": "案件資料、営業確認、返済原資、物件利用実態、結果登録",
            "expected_outcome_to_verify": "懸念が成約/失注/延滞/条件変更に影響したか",
            "source_note": cleaned[:600],
            "confidence": _confidence_from_source(source, cleaned),
            "review_reason": "human_review_required_before_promotion",
        },
        "dropped_guidance": [
            "generic advice without a screening action",
            "personal evaluation material",
            "unverified source-only claims",
        ],
        "guardrail": "candidate_only_no_memory_promotion_no_scoring_change_no_obsidian_write",
    }


def validate_lease_source_summary(
    source_title: str,
    source_url_or_path: str = "",
    publisher: str = "",
    published_date: str = "",
    evidence_summary: str = "",
    lease_relevance: str = "",
) -> dict[str, Any]:
    """Validate whether a source is safe to use in lease screening context.

    Args:
        source_title: Source title or topic.
        source_url_or_path: URL or local note path, if available.
        publisher: Publisher, author, or institution.
        published_date: Publication or event date.
        evidence_summary: Short description of the evidence presented.
        lease_relevance: How the source affects repayment, asset value, demand,
            funding cost, regulation, or competition.

    Returns:
        Source validation summary with use level and cautions. Current facts
        should still be browsed/verified when freshness matters.
    """
    publisher_l = publisher.lower()
    evidence_l = evidence_summary.lower()
    primary_markers = ("go.jp", "boj.or.jp", "fsa.go.jp", "meti.go.jp", "edinet", "官報", "決算", "統計")
    weak_markers = ("匿名", "まとめ", "sns", "広告", "pr", "スポンサー", "未確認")
    if any(marker in publisher_l or marker in evidence_l for marker in weak_markers):
        source_type = "weak"
    elif any(marker in publisher_l or marker in source_url_or_path.lower() for marker in primary_markers):
        source_type = "primary"
    elif publisher and evidence_summary:
        source_type = "specialist_secondary"
    else:
        source_type = "general_secondary"

    has_date = bool(str(published_date or "").strip())
    has_evidence = bool(str(evidence_summary or "").strip())
    relevant = bool(str(lease_relevance or "").strip())
    if source_type == "primary" and has_date and has_evidence and relevant:
        use_level = "usable_as_fact"
    elif source_type != "weak" and relevant:
        use_level = "usable_as_signal"
    elif source_type == "weak":
        use_level = "do_not_use"
    else:
        use_level = "background_only"

    cautions: list[str] = []
    if not has_date:
        cautions.append("publication_or_event_date_missing")
    if not has_evidence:
        cautions.append("evidence_summary_missing")
    if not relevant:
        cautions.append("lease_relevance_unclear")
    if source_type == "weak":
        cautions.append("weak_or_unverified_source")

    return {
        "mode": "agentic_skill_lease_source_validator",
        "source": {
            "title": source_title,
            "url_or_path": source_url_or_path,
            "publisher": publisher,
            "published_date": published_date,
            "type": source_type,
        },
        "assessment": {
            "freshness": "date_present" if has_date else "date_missing",
            "reliability": source_type,
            "lease_relevance": lease_relevance or "unclear",
            "use_level": use_level,
            "cautions": cautions,
        },
        "screening_use": {
            "comment_style": "事実として断定" if use_level == "usable_as_fact" else "確認点・注意信号として扱う",
            "additional_check": "一次情報または複数ソースで裏取り" if use_level != "usable_as_fact" else "案件事実との接続を確認",
        },
        "guardrail": "no_external_fetch_no_auto_memory_promotion_no_case_decision_by_source_alone",
    }


def convert_research_to_screening_insights(
    research_summary: str,
    industry: str = "",
    asset: str = "",
    source_strength: str = "plausible",
    max_checks: int = 3,
) -> dict[str, Any]:
    """Convert research material into lease screening checks and comment draft.

    Args:
        research_summary: Research/news/note content to convert.
        industry: Affected industry, if known.
        asset: Affected leased asset, if known.
        source_strength: confirmed, plausible, or weak.
        max_checks: Maximum checks to return, clamped to 1..3.

    Returns:
        Research-to-screening synthesis. It does not write notes or promote
        judgment assets.
    """
    cleaned = _clean_text(research_summary)
    try:
        requested_checks = int(max_checks or 3)
    except (TypeError, ValueError):
        requested_checks = 3
    limit = max(1, min(3, requested_checks))
    origins = _infer_risk_origins(cleaned)
    sentences = _sentences(cleaned, limit=limit)
    checks: list[dict[str, str]] = []
    for idx in range(limit):
        basis = sentences[idx] if idx < len(sentences) else cleaned[:80]
        origin = origins[idx] if idx < len(origins) else origins[0]
        checks.append({
            "question": f"{industry or '対象先'}で、{basis[:60]} が返済原資・成約条件に影響していないか確認する",
            "why": f"risk_origin={origin}",
            "evidence_to_request": "営業ヒアリング、直近受注、資金繰り資料、物件利用計画、相見積状況",
            "if_yes": "条件設定・追加資料・コメントで明示",
            "if_no": "マクロ情報として扱い、個別判断へ過度に反映しない",
        })

    confidence = source_strength if source_strength in {"confirmed", "plausible", "weak"} else "plausible"
    return {
        "mode": "agentic_skill_research_to_screening_insights",
        "research_summary": {
            "industry": industry,
            "asset": asset,
            "source_strength": confidence,
            "text": cleaned[:700],
        },
        "screening_implications": origins,
        "top_checks": checks,
        "comment_draft": (
            f"{industry or '対象業種'}に関する外部情報は、"
            f"{', '.join(origins)} の確認材料として扱う。"
            "個別案件では返済原資・物件利用・競合状況で裏取りする。"
        ),
        "judgment_asset_candidate_status": "candidate_only_research_signal",
        "guardrail": "max_three_checks_no_auto_approval_no_score_change_no_memory_promotion",
    }


def build_screening_decision_flow(
    criteria: str,
    case_context: str = "",
    output_format: str = "mermaid",
) -> dict[str, Any]:
    """Build a compact lease screening decision flow from criteria.

    Args:
        criteria: Decision criteria, concern, or proposed screening logic.
        case_context: Optional case facts to anchor the flow.
        output_format: mermaid or text.

    Returns:
        A human-reviewable decision flow. It is not an automated approval rule.
    """
    text = _clean_text(" ".join([criteria or "", case_context or ""]))
    origins = _infer_risk_origins(text)
    primary = origins[0]
    nodes = [
        {"id": "A", "type": "start", "label": "案件入力"},
        {"id": "B", "type": "decision", "label": f"主要リスク起点は {primary} か"},
        {"id": "C", "type": "question", "label": "返済原資・物件利用・競合状況を追加確認"},
        {"id": "D", "type": "action", "label": "条件付き承認または保留として人間レビュー"},
        {"id": "E", "type": "action", "label": "通常コメントで根拠を明示"},
    ]
    connections = [
        {"from": "A", "to": "B", "label": "スコア/案件事実を確認"},
        {"from": "B", "to": "C", "label": "不確実または高影響"},
        {"from": "C", "to": "D", "label": "資料不足/懸念残り"},
        {"from": "C", "to": "E", "label": "裏取り完了"},
    ]
    mermaid = "\n".join([
        "flowchart TD",
        "  A[案件入力] --> B{主要リスク起点を特定}",
        "  B -->|不確実/高影響| C[追加確認]",
        "  C -->|懸念残り| D[条件付き承認/保留: 人間レビュー]",
        "  C -->|裏取り完了| E[通常コメントで根拠明示]",
    ])
    return {
        "mode": "agentic_skill_screening_decision_flow_builder",
        "risk_origins": origins,
        "nodes": nodes,
        "connections": connections,
        "mermaid": mermaid if output_format != "text" else "",
        "notes": [
            "approval_or_denial_is_not_automated",
            "missing_facts_should_be_question_nodes",
            "human_review_required_for_high_impact_credit_decisions",
        ],
        "guardrail": "decision_support_only_no_automatic_approval_or_denial",
    }


def write_scqa_report(
    material: str,
    audience: str = "internal",
    purpose: str = "report",
    max_length: str = "short",
) -> dict[str, Any]:
    """Draft an SCQA report for lease screening or Shion project communication.

    Args:
        material: Content to rewrite.
        audience: internal, sales, judge_demo, or screening.
        purpose: report, slack, readme, presentation, or screening_comment.
        max_length: short or standard.

    Returns:
        SCQA blocks and a compact draft. The caller should adapt wording to the
        final surface.
    """
    cleaned = _clean_text(material)
    facts = _sentences(cleaned, limit=4)
    situation = facts[0] if facts else "現状の情報を整理する必要がある。"
    complication = facts[1] if len(facts) > 1 else "そのままでは論点・確認点・次の行動が混ざる。"
    question = "何を確認し、どの判断または説明に使うべきか。"
    if purpose == "screening_comment":
        question = "この案件で判定前に何を確認し、どの条件を残すべきか。"
    answer = facts[2] if len(facts) > 2 else "SCQAで論点を分け、結論と次の行動を短く示す。"
    compact = max_length == "short" or purpose == "slack"
    draft = (
        f"S: {situation}\nC: {complication}\nQ: {question}\nA: {answer}"
        if compact
        else {
            "Situation": situation,
            "Complication": complication,
            "Question": question,
            "Answer": answer,
        }
    )
    return {
        "mode": "agentic_skill_scqa_report_writer",
        "audience": audience,
        "purpose": purpose,
        "scqa": {
            "situation": situation,
            "complication": complication,
            "question": question,
            "answer": answer,
        },
        "draft": draft,
        "style_notes": [
            "keep_sentences_short",
            "do_not_overstate_evidence",
            "make_next_action_explicit",
        ],
        "guardrail": "writing_support_only_no_source_strength_inflation",
    }


def inspect_agentic_skill_flow(limit: int = 5) -> dict[str, Any]:
    """Inspect Shion's agentic skill usage flow and open review inbox.

    Args:
        limit: Maximum open inbox items to return, clamped to 1..10.

    Returns:
        Read-only flow status and compact open review candidates. This does not
        review, promote, write Obsidian, change RAG, or alter scoring.
    """
    try:
        max_items = max(1, min(10, int(limit or 5)))
    except (TypeError, ValueError):
        max_items = 5
    flow = check_agentic_skill_flow()
    inbox = load_agentic_skill_review_inbox(limit=max_items, status="candidate")
    compact_items = []
    for item in inbox:
        context = item.get("case_context") if isinstance(item.get("case_context"), dict) else {}
        compact_items.append({
            "id": item.get("id"),
            "tool_name": item.get("tool_name"),
            "candidate_type": item.get("candidate_type"),
            "claim": item.get("claim"),
            "created_at": item.get("created_at"),
            "case_context": {
                "company_name": context.get("company_name", ""),
                "industry_cat": context.get("industry_cat", ""),
                "asset_name": context.get("asset_name", ""),
                "score": context.get("score", None),
            },
        })
    return {
        "mode": "agentic_skill_flow_inspection",
        "status": flow.get("status"),
        "summary": flow.get("summary", {}),
        "checks": flow.get("checks", []),
        "open_review_items": compact_items,
        "next_action": (
            "人間がimprovement-logの紫苑ADKレビュー箱で採用・修正・保留・却下を選ぶ"
            if compact_items else
            "未レビュー候補はない。通常運用を続ける"
        ),
        "guardrail": "read_only_no_review_decision_no_promotion_no_scoring_no_rag_no_obsidian_write",
    }


def propose_agentic_skill_next_actions(limit: int = 3) -> dict[str, Any]:
    """Propose next human actions from Shion's visible agentic skill flow.

    Args:
        limit: Maximum proposals to return, clamped to 1..3.

    Returns:
        Read-only proposals for the human reviewer. This does not modify
        prompts, pipelines, review decisions, memory, RAG, or scoring.
    """
    try:
        max_items = max(1, min(3, int(limit or 3)))
    except (TypeError, ValueError):
        max_items = 3
    flow = check_agentic_skill_flow()
    summary = flow.get("summary", {}) if isinstance(flow.get("summary"), dict) else {}
    inbox = load_agentic_skill_review_inbox(limit=20, status="candidate")
    proposals: list[dict[str, Any]] = []

    if flow.get("status") == "warn":
        warn_checks = [
            check for check in flow.get("checks", [])
            if isinstance(check, dict) and check.get("status") == "warn"
        ]
        proposals.append({
            "priority": "high",
            "type": "repair_observation_flow",
            "title": "agentic skill の観測導線を確認する",
            "reason": "使用ログ、レビュー箱、レビュー決定のどこかにリンク欠損があります。",
            "target": warn_checks[:3],
            "human_action": "improvement-log の紫苑ADKレビュー箱で警告内容を確認する",
        })

    repeated = [item for item in inbox if item.get("prior_review_context")]
    if repeated:
        sample = repeated[0]
        proposals.append({
            "priority": "high",
            "type": "flag_repeated_pattern",
            "title": "過去にrejected/heldされた論点と似た候補がある",
            "reason": (
                "同じcandidate_typeで主張が重なる過去決定があります。"
                "同じ理由で繰り返し却下しないよう、まず過去の却下理由と比べてください。"
            ),
            "target": {
                "id": sample.get("id"),
                "candidate_type": sample.get("candidate_type"),
                "claim": sample.get("claim"),
                "prior_review_context": sample.get("prior_review_context", [])[:2],
            },
            "human_action": "過去の却下/保留理由と比べ、今回は根拠が違うか確認してから採用・修正・保留・却下を選ぶ",
        })

    if inbox:
        proposals.append({
            "priority": "medium" if len(inbox) <= 3 else "high",
            "type": "review_open_candidates",
            "title": f"未レビュー候補を{min(len(inbox), 3)}件だけ見る",
            "reason": f"未レビュー候補が {len(inbox)} 件あります。溜めすぎると判断資産候補がノイズ化します。",
            "target": [
                {
                    "id": item.get("id"),
                    "tool_name": item.get("tool_name"),
                    "candidate_type": item.get("candidate_type"),
                    "claim": item.get("claim"),
                    "prior_review_context": item.get("prior_review_context", []),
                }
                for item in inbox[:3]
            ],
            "human_action": "採用・修正・保留・却下のどれかを選ぶ",
        })

    by_type: dict[str, int] = {}
    for item in inbox:
        candidate_type = str(item.get("candidate_type") or "unknown")
        by_type[candidate_type] = by_type.get(candidate_type, 0) + 1
    if by_type:
        top_type, top_count = max(by_type.items(), key=lambda kv: kv[1])
        if top_count >= 3:
            proposals.append({
                "priority": "medium",
                "type": "tighten_or_theme_review",
                "title": f"{top_type} が多いので発火条件を見る",
                "reason": f"同じ種類の候補が {top_count} 件あります。役に立つ集中ならテーマ化、薄いなら発火条件を絞ります。",
                "target": {"candidate_type": top_type, "count": top_count},
                "human_action": "候補の質を見て、使えるテーマかノイズか判断する",
            })

    if not proposals:
        proposals.append({
            "priority": "low",
            "type": "continue_observation",
            "title": "通常運用を続ける",
            "reason": "agentic skill の流れに警告や未レビュー候補の滞留はありません。",
            "target": summary,
            "human_action": "次に紫苑が候補を作った時だけ確認する",
        })

    return {
        "mode": "agentic_skill_next_action_proposals",
        "status": flow.get("status"),
        "summary": summary,
        "proposals": proposals[:max_items],
        "guardrail": "proposal_only_no_pipeline_no_prompt_change_no_review_decision_no_memory_promotion_no_scoring_no_rag",
    }


AGENTIC_SKILL_TOOLS = [
    structure_judgment_asset_candidate,
    validate_lease_source_summary,
    convert_research_to_screening_insights,
    build_screening_decision_flow,
    write_scqa_report,
    inspect_agentic_skill_flow,
    propose_agentic_skill_next_actions,
]
