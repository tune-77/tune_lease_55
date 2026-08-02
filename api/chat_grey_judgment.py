"""Grey-judgment memory prompt helpers."""

from __future__ import annotations

from typing import Any, Callable

from api.chat_continuity_prompts import relationship_signal_route


GREY_JUDGMENT_QUERY_TERMS = (
    "グレー", "迷う", "違和感", "そうは言っても", "それでも", "条件付き",
    "通すなら", "否決寄り", "承認寄り", "人間的", "数字だけ", "稟議",
    "審査", "与信", "判断", "温度感", "例外", "境界", "定性", "現場感",
    "軍師", "落としどころ",
)

QUALITATIVE_FIELD_LABELS = {
    "qual_corr_company_history": "業歴",
    "qual_corr_customer_stability": "顧客安定性",
    "qual_corr_repayment_history": "返済履歴",
    "qual_corr_business_future": "事業将来性",
    "qual_corr_equipment_purpose": "設備目的",
    "qual_corr_main_bank": "メイン行",
}


def case_qualitative_summary(case: dict[str, Any]) -> str:
    inputs = case.get("inputs") if isinstance(case.get("inputs"), dict) else {}
    parts: list[str] = []
    passion = str(inputs.get("passion_text") or case.get("passion_text") or "").strip()
    if passion:
        parts.append(f"現場メモ={passion[:180]}")
    for key, label in QUALITATIVE_FIELD_LABELS.items():
        value = str(inputs.get(key) or case.get(key) or "").strip()
        if value and value != "未選択":
            parts.append(f"{label}={value[:80]}")
    intuition = inputs.get("intuition", case.get("intuition"))
    if intuition not in (None, "", 0):
        parts.append(f"直感スコア={intuition}")
    return " / ".join(parts)


def load_gunshi_judgment_memory(
    *,
    training_candidates_loader: Callable[..., list[dict[str, Any]]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    try:
        rows = training_candidates_loader(approved_only=False)
    except Exception:
        return []

    preferred_sources = {"gunshi_chat", "debate", "lease_news_debate", "register_trigger"}
    picked: list[dict[str, Any]] = []
    for row in reversed(rows):
        source = str(row.get("source") or "")
        reason = str(row.get("reason") or "").strip()
        if not reason:
            continue
        if source not in preferred_sources and not any(term in reason for term in GREY_JUDGMENT_QUERY_TERMS):
            continue
        picked.append({
            "source": source or "judgment_feedback",
            "case_id": row.get("case_id") or "",
            "score": row.get("score"),
            "model_decision": row.get("model_decision") or "",
            "human_decision": row.get("human_decision") or "",
            "reason": reason[:240],
            "review_status": row.get("review_status") or "",
            "evidence_snapshot": row.get("evidence_snapshot") or {},
        })
        if len(picked) >= limit:
            break
    return picked


def build_grey_judgment_prompt_block(
    message: str,
    *,
    cases_loader: Callable[[], list[dict[str, Any]]],
    gunshi_memory_loader: Callable[..., list[dict[str, Any]]],
    limit: int = 5,
) -> tuple[str, dict[str, Any]]:
    text = str(message or "")
    route = relationship_signal_route(text)
    should_use = route == "lease_judgment" or any(term in text for term in GREY_JUDGMENT_QUERY_TERMS)
    if not should_use:
        return "", {"used": False, "reason": "not_lease_judgment", "refs": []}

    try:
        cases = cases_loader()
    except Exception as grey_load_error:
        return "", {"used": False, "reason": f"load_error: {grey_load_error}", "refs": []}

    query_terms = [term for term in GREY_JUDGMENT_QUERY_TERMS if term in text]
    scored: list[tuple[int, dict[str, Any]]] = []
    for case in reversed(cases):
        grey = case.get("grey_judgment") if isinstance(case.get("grey_judgment"), dict) else {}
        fields = {
            "human_discomfort": str(grey.get("human_discomfort") or case.get("human_discomfort") or "").strip(),
            "but_still_reason": str(grey.get("but_still_reason") or case.get("but_still_reason") or "").strip(),
            "approval_condition_memo": str(grey.get("approval_condition_memo") or case.get("approval_condition_memo") or "").strip(),
            "non_negotiable_condition": str(grey.get("non_negotiable_condition") or case.get("non_negotiable_condition") or "").strip(),
            "retrospective_note": str(grey.get("retrospective_note") or case.get("retrospective_note") or "").strip(),
        }
        qualitative_summary = case_qualitative_summary(case)
        if not any(fields.values()) and not qualitative_summary:
            continue
        haystack = " ".join([
            str(case.get("company_name") or ""),
            str(case.get("industry_major") or ""),
            str(case.get("industry_sub") or ""),
            str(case.get("final_status") or ""),
            str(case.get("lost_reason") or ""),
            str(case.get("final_note") or ""),
            " ".join(str(v) for v in fields.values()),
            qualitative_summary,
        ])
        score = 1 + sum(1 for term in query_terms if term and term in haystack)
        if case.get("final_status") == "成約" and fields["but_still_reason"]:
            score += 1
        if fields["approval_condition_memo"] or fields["non_negotiable_condition"]:
            score += 1
        if qualitative_summary:
            score += 1
        scored.append((score, {**fields, "qualitative_summary": qualitative_summary, "case": case}))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [item[1] for item in scored[:limit]]
    refs: list[dict[str, Any]] = []
    lines: list[str] = []
    for item in selected:
        case = item["case"]
        result = case.get("result") if isinstance(case.get("result"), dict) else {}
        score = case.get("score") or case.get("score_base") or result.get("score")
        decision = case.get("hantei") or result.get("hantei") or ""
        ref = {
            "case_id": case.get("id") or "",
            "company_name": case.get("company_name") or "",
            "status": case.get("final_status") or "",
            "score": score,
            "decision": decision,
            "human_discomfort": item["human_discomfort"][:180],
            "but_still_reason": item["but_still_reason"][:180],
            "approval_condition_memo": item["approval_condition_memo"][:180],
            "non_negotiable_condition": item["non_negotiable_condition"][:180],
            "retrospective_note": item["retrospective_note"][:180],
            "qualitative_summary": item["qualitative_summary"][:240],
            "source": "past_cases",
        }
        refs.append(ref)
        parts = [
            f"- source=past_cases / 案件ID={ref['case_id'] or '-'}",
            f"結果={ref['status'] or '-'}",
            f"AI={score if score not in (None, '') else '-'}点/{decision or '-'}",
        ]
        if item["human_discomfort"]:
            parts.append(f"違和感={item['human_discomfort'][:120]}")
        if item["but_still_reason"]:
            parts.append(f"それでも={item['but_still_reason'][:120]}")
        if item["approval_condition_memo"]:
            parts.append(f"条件={item['approval_condition_memo'][:120]}")
        if item["non_negotiable_condition"]:
            parts.append(f"譲れない線={item['non_negotiable_condition'][:120]}")
        if item["retrospective_note"]:
            parts.append(f"振り返り={item['retrospective_note'][:120]}")
        if item["qualitative_summary"]:
            parts.append(f"定性={item['qualitative_summary'][:160]}")
        lines.append(" / ".join(parts))

    for item in gunshi_memory_loader(limit=limit):
        refs.append({
            "source": item["source"],
            "case_id": item["case_id"],
            "status": item["review_status"],
            "score": item["score"],
            "decision": f"{item['model_decision']}→{item['human_decision']}",
            "reason": item["reason"],
        })
        score_text = f"{float(item['score']):.1f}点" if item.get("score") is not None else "-"
        lines.append(
            f"- source={item['source']} / 案件ID={item['case_id'] or '-'} / "
            f"AI={score_text}/{item['model_decision'] or '-'}→担当者={item['human_decision'] or '-'} / "
            f"理由={item['reason']}"
        )

    payload = {
        "used": bool(lines),
        "reason": "matched_grey_judgment_cases" if lines else "no_registered_grey_judgment",
        "query_terms": query_terms,
        "refs": refs,
    }
    if not lines:
        return "", payload

    block = """

【グレー判断の過去記憶】
これは通常のスコアや一般論より優先して見る、人間が迷ったリース判断の経験です。
数字だけで採否を決めず、軍師AIで記録された判断変更・定性項目・現場メモ・違和感・それでも通した理由・通すなら条件・譲れない線を稟議判断へ変換してください。
過去登録:
""".rstrip() + "\n" + "\n".join(lines)
    return block, payload
