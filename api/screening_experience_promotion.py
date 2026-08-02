"""Promote closed case results into screening experience cases."""

from __future__ import annotations

from typing import Any, Callable


def experience_case_exists(
    source_case_id: str,
    source: str,
    *,
    ensure_table: Callable[..., None],
    connection_factory: Callable[[], Any],
    placeholder_factory: Callable[[], str],
) -> bool:
    source_case_id = str(source_case_id or "").strip()
    source = str(source or "").strip()
    if not source_case_id or not source:
        return False
    ensure_table(seed_demo=True)
    ph = placeholder_factory()
    with connection_factory() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id FROM screening_experience_cases
             WHERE source_case_id = {ph}
               AND source = {ph}
             LIMIT 1
            """,
            (source_case_id, source),
        )
        return cur.fetchone() is not None


def promote_case_result_to_screening_experience(
    *,
    case_id: str,
    case_data: dict[str, Any],
    status: str,
    request_factory: Callable[..., Any],
    save_screening_experience_case: Callable[[Any], dict[str, Any]],
    experience_exists: Callable[[str, str], bool],
    patches: dict[str, Any] | None = None,
    source: str = "case_result_auto",
) -> dict[str, Any]:
    patches = patches or {}
    status = str(status or patches.get("final_status") or case_data.get("final_status") or "").strip()
    if status not in {"成約", "失注", "検収", "検収完了"}:
        return {"status": "skipped", "reason": "not_closed_result"}
    normalized_status = "成約" if status in {"成約", "検収", "検収完了"} else "失注"
    if experience_exists(case_id, source):
        return {"status": "skipped", "reason": "already_promoted"}

    inputs = case_data.get("inputs") if isinstance(case_data.get("inputs"), dict) else case_data
    result = case_data.get("result") if isinstance(case_data.get("result"), dict) else {}
    company = str(inputs.get("company_name") or case_data.get("company_name") or "名称未設定").strip()
    industry_major = str(result.get("industry_major") or inputs.get("industry_major") or case_data.get("industry_major") or "").strip()
    industry_sub = str(result.get("industry_sub") or inputs.get("industry_sub") or case_data.get("industry_sub") or "").strip()
    sales_dept = str(inputs.get("sales_dept") or case_data.get("sales_dept") or "").strip()
    asset = str(inputs.get("asset_name") or inputs.get("asset_type") or "").strip()
    customer_type = str(inputs.get("customer_type") or case_data.get("customer_type") or "").strip()
    main_bank = str(inputs.get("main_bank") or case_data.get("main_bank") or "").strip()
    competitor = str(inputs.get("competitor") or case_data.get("competitor") or "").strip()
    deal_source = str(inputs.get("deal_source") or case_data.get("deal_source") or "").strip()
    score_raw = result.get("score", result.get("score_base", case_data.get("score", case_data.get("score_base"))))
    try:
        score = float(score_raw) if score_raw not in (None, "") else None
    except Exception:
        score = None
    decision = str(result.get("hantei") or case_data.get("hantei") or "").strip()
    final_rate = patches.get("final_rate", case_data.get("final_rate"))
    competitor_rate = patches.get("competitor_rate", case_data.get("competitor_rate"))
    lost_reason = str(patches.get("lost_reason") or case_data.get("lost_reason") or case_data.get("loss_reason") or "").strip()
    final_note = str(patches.get("final_note") or case_data.get("final_note") or "").strip()
    grey = patches.get("grey_judgment") if isinstance(patches.get("grey_judgment"), dict) else {}
    if not grey and isinstance(case_data.get("grey_judgment"), dict):
        grey = case_data.get("grey_judgment") or {}

    similarity_bits = [
        industry_sub or industry_major,
        customer_type,
        asset,
        main_bank,
        competitor,
        deal_source,
    ]
    similarity = " / ".join(str(bit) for bit in similarity_bits if str(bit or "").strip()) or "登録結果から生成した経験ケース"

    if normalized_status == "成約":
        outcome = "成約"
        if final_rate not in (None, "", 0):
            outcome += f"・最終金利 {final_rate}%"
        action_parts = ["成約登録。"]
        if grey.get("but_still_reason"):
            action_parts.append(f"それでも通した理由: {grey.get('but_still_reason')}")
        if grey.get("approval_condition_memo"):
            action_parts.append(f"承認条件: {grey.get('approval_condition_memo')}")
        if competitor_rate not in (None, "", 0):
            action_parts.append(f"競合金利 {competitor_rate}% を踏まえて条件調整。")
        action_taken = " ".join(action_parts).strip()
        lesson = "成約に至った判断材料を、次回の同種案件で承認理由・条件設定・価格判断へ再利用する。"
    else:
        outcome = "失注"
        if lost_reason:
            outcome += f"・理由: {lost_reason}"
        action_parts = ["失注登録。"]
        if lost_reason:
            action_parts.append(f"失注理由を記録: {lost_reason}")
        if competitor_rate not in (None, "", 0):
            action_parts.append(f"競合金利 {competitor_rate}% との差を記録。")
        if final_note:
            action_parts.append(f"補足: {final_note}")
        action_taken = " ".join(action_parts).strip()
        lesson = "失注要因を、次回の初期ヒアリング・競合確認・条件提示の改善材料として再利用する。"

    if grey.get("human_discomfort"):
        lesson += f" 違和感: {grey.get('human_discomfort')}"
    difference = "最終結果から自動昇格。次回類似案件では、今回の結果要因が同じか、顧客事情・競合・銀行支援・物件保全が違うかを確認する。"

    req = request_factory(
        source_case_id=str(case_id or "")[:160],
        company_name=company,
        period=str(patches.get("final_result_date") or case_data.get("final_result_date") or "")[:80] or "結果登録時",
        industry_major=industry_major,
        industry_sub=industry_sub,
        sales_dept=sales_dept,
        score=score,
        decision=decision or normalized_status,
        outcome=outcome,
        similarity=similarity,
        action_taken=action_taken,
        lesson=lesson,
        difference=difference,
        source=source,
        form_snapshot=inputs if isinstance(inputs, dict) else {},
        result_snapshot=result if isinstance(result, dict) else {},
    )
    entry = save_screening_experience_case(req)
    return {"status": "promoted", "case": entry}
