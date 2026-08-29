"""紫苑の追加確認モード。

審査スコアを変更せず、案件の不足・曖昧・高影響事項を最大3問に絞り、
人間の回答による判断更新と後日検証対象を保存する。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from api.db_connection import get_connection, placeholder
from constants import APPROVAL_LINE, CONDITIONAL_LINE, REVIEW_LINE


ANSWER_LABELS = {
    "confirmed": "確認できた",
    "partial": "一部確認",
    "concern": "未確認・懸念あり",
}

IMPACT_LABELS = {
    "decision_changed": "判断変更・条件追加",
    "risk_prevented": "事故・見落とし防止",
    "outcome_matched": "懸念が結果に表れた",
    "evidence_strengthened": "判断根拠の補強",
    "not_helpful": "役立たなかった",
}

_ADVERSE_OUTCOMES = {"失注", "延滞", "事故", "デフォルト", "貸倒", "解約"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _baseline_decision(result: dict[str, Any]) -> str:
    stated = _text(result.get("hantei") or result.get("decision"))
    if any(term in stated for term in ("否決", "否認")):
        return stated
    score = _number(result.get("score") if result.get("score") is not None else result.get("score_base"))
    if score is None:
        return stated or "現判断を維持"
    suffix = f"（{stated} / {score:.1f}点）" if stated else f"（{score:.1f}点）"
    if score < REVIEW_LINE:
        return f"即否決圏{suffix}"
    if score < CONDITIONAL_LINE:
        return f"否決・要再設計圏{suffix}"
    if score < APPROVAL_LINE:
        return f"条件付き・境界圏{suffix}"
    return f"承認圏{suffix}"


def _candidate(
    question_id: str,
    priority: int,
    category: str,
    question: str,
    reason: str,
    hypothesis: str,
    negative_condition: str,
    source_asset_id: str = "",
) -> dict[str, Any]:
    return {
        "id": question_id,
        "priority": priority,
        "category": category,
        "question": question,
        "reason": reason,
        "hypothesis": hypothesis,
        "negative_condition": negative_condition,
        "source_asset_id": source_asset_id,
        "answer_options": [
            {"value": value, "label": label}
            for value, label in ANSWER_LABELS.items()
        ],
    }


def build_followup_questions(
    form: dict[str, Any],
    result: dict[str, Any],
    judgment_assets: list[dict[str, Any]] | None = None,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """案件文脈から確認質問を最大3問選ぶ。生成は決定論的で説明可能。"""
    candidates: list[dict[str, Any]] = []
    purpose = _text(form.get("asset_purpose"))
    asset_name = _text(form.get("asset_name"))
    evidence = _text(form.get("asset_evidence_level"))
    competitor = _text(form.get("competitor"))
    customer_type = _text(form.get("customer_type"))
    score = _number(result.get("score") if result.get("score") is not None else result.get("score_base"))
    q_risk = _number(result.get("quantum_risk")) or 0.0
    anomaly = _number(result.get("umap_anomaly_score")) or 0.0
    acquisition_cost = _number(form.get("acquisition_cost"))
    sales = _number(form.get("nenshu"))

    if not purpose:
        candidates.append(_candidate(
            "asset-purpose", 100, "導入目的",
            "この設備を今導入する理由と、売上・原価・生産性のどこに効くか確認できましたか？",
            "導入目的が未入力で、資金使途と返済原資のつながりを説明できません。",
            "導入効果が具体化すれば、物件の必要性と返済原資を条件文にできます。",
            "導入効果を定量・具体化できない場合は、必要性と回収可能性を追加確認する。",
        ))
    elif not asset_name or evidence in {"", "未確認", "なし"}:
        candidates.append(_candidate(
            "asset-evidence", 96, "物件確認",
            f"対象物件{f'「{asset_name}」' if asset_name else ''}の見積・型式・設置場所・中古相場を確認できましたか？",
            "物件確認資料が不足しており、価格妥当性と処分可能性が未確定です。",
            "物件資料が揃えば、価格妥当性と保全条件を説明できます。",
            "資料不足が残る場合は、実行前の資料取得を承認条件にする。",
        ))

    if q_risk >= 35 or anomaly >= 2.5:
        candidates.append(_candidate(
            "numeric-consistency", 98, "数値整合性",
            "財務数値・申込内容・営業説明の食い違いについて、原資料まで戻って確認できましたか？",
            "異常・不整合シグナルがあり、スコア以前に入力事実の確認が必要です。",
            "原資料で整合すれば異常シグナルを説明可能な差分として扱えます。",
            "不整合が残る場合はスコアを根拠に進めず、原資料確認を継続する。",
        ))

    if score is None or 35 <= score < 70 or "条件" in _baseline_decision(result):
        candidates.append(_candidate(
            "repayment-source", 92, "返済原資",
            "月額リース料を何のキャッシュフローで支払い、下振れ時に何で補完するか確認できましたか？",
            "境界案件では点数より、返済原資と下振れ耐性の具体性が判断を分けます。",
            "主返済原資と補完原資が確認できれば、条件付き進行の理由になります。",
            "返済原資が曖昧な場合は、資金繰り資料または支援条件を追加確認する。",
        ))

    if acquisition_cost and sales and sales > 0 and acquisition_cost / sales >= 0.1:
        candidates.append(_candidate(
            "deal-scale", 94, "案件規模",
            "今回の取得価格を、年商・既存設備投資・過去最大案件と比べて無理のない規模と確認できましたか？",
            "案件規模が売上に対して大きく、通常取引の延長として扱えない可能性があります。",
            "過去投資と導入効果で規模の妥当性を説明できれば、規模リスクを条件へ変換できます。",
            "規模の妥当性を説明できない場合は、減額・協調・追加保証を検討する。",
        ))

    if competitor == "競合あり" or _number(form.get("num_competitors")) not in (None, 0):
        candidates.append(_candidate(
            "competition", 82, "競合・成約",
            "競合との差は金利だけですか。対象範囲・保守・期間・顧客が重視する条件まで揃えて比較できましたか？",
            "信用リスクと競合による成約リスクを分けないと、条件設定を誤ります。",
            "勝敗条件が分かれば、信用条件を崩さず営業条件を調整できます。",
            "競合差が不明な場合は、信用判断と価格交渉を分けて確認する。",
        ))

    if "新規" in customer_type:
        candidates.append(_candidate(
            "new-customer-history", 86, "新規取引",
            "新規先として、取引経緯・実質的な紹介元・既存借入の返済履歴を裏付けられましたか？",
            "新規先は社内取引履歴がなく、営業接点だけでは継続性を確認できません。",
            "第三者情報と返済履歴が確認できれば、新規先の情報不足を補えます。",
            "裏付けが弱い場合は、限度・保証・初回条件を保守的にする。",
        ))

    for index, asset in enumerate(judgment_assets or []):
        if _text(asset.get("candidate_type")) != "confirmation_question":
            continue
        claim = _text(asset.get("edited_claim") or asset.get("effective_claim") or asset.get("claim"))
        if len(claim) < 8:
            continue
        candidates.append(_candidate(
            f"asset-{_text(asset.get('id')) or index}", 78 - index, "判断資産",
            claim if claim.endswith(("？", "?")) else f"{claim}を確認できましたか？",
            "過去の調査・案件対応から得た確認パターンです。",
            "今回も有効なら、案件固有に応用した確認パターンとして検証できます。",
            "今回に当てはまらなければ無理に使わず、見送り理由を残す。",
            _text(asset.get("id")),
        ))

    if not candidates:
        candidates.append(_candidate(
            "decision-counterpoint", 50, "反証確認",
            "現在の判断を覆すとしたら、どの事実が一番あり得るか確認できましたか？",
            "入力上の明確な欠落が少ないため、結論を強めるより反証を確認します。",
            "反証候補を確認すれば、現在判断の適用条件を明確にできます。",
            "反証が残る場合は、現判断を確定せず追加資料を待つ。",
        ))

    unique: dict[str, dict[str, Any]] = {}
    for item in sorted(candidates, key=lambda row: (-int(row["priority"]), row["id"])):
        unique.setdefault(item["category"], item)
    return list(unique.values())[: max(1, min(int(limit or 3), 3))]


def build_updated_view(
    baseline_decision: str,
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {_text(answer.get("question_id")): answer for answer in answers}
    normalized: list[dict[str, Any]] = []
    for question in questions:
        answer = by_id.get(_text(question.get("id")), {})
        status = _text(answer.get("status"))
        if status not in ANSWER_LABELS:
            status = "partial"
        normalized.append({
            "question_id": question["id"],
            "status": status,
            "label": ANSWER_LABELS[status],
            "note": _text(answer.get("note"))[:2000],
        })

    concerns = [item for item in normalized if item["status"] == "concern"]
    partials = [item for item in normalized if item["status"] == "partial"]
    confirmed = [item for item in normalized if item["status"] == "confirmed"]
    baseline_negative = any(term in baseline_decision for term in ("否決", "否認"))
    if concerns:
        updated = "追加確認を継続"
        change = "懸念が残ったため、現判断を確定せず停止線を維持します。"
    elif partials:
        updated = "条件付きで進行可" if not baseline_negative else "条件再設計候補"
        change = "一部確認に留まる項目を、実行前の承認条件へ変換しました。"
    elif baseline_negative:
        updated = "条件再設計候補"
        change = "確認事項は解消しましたが、元の否決理由は自動で覆さず条件再設計へ回します。"
    else:
        updated = "現判断を維持・根拠補強"
        change = f"全{len(normalized)}件の確認が取れたため、点数を変えずに現判断の説明根拠を補強しました。"

    question_by_id = {_text(item.get("id")): item for item in questions}
    approval_conditions: list[str] = []
    for answer in concerns + partials:
        question = question_by_id.get(answer["question_id"], {})
        condition = _text(question.get("negative_condition"))
        if condition:
            approval_conditions.append(condition)
    if not approval_conditions:
        approval_conditions.append("確認済み内容を稟議へ明記し、実行条件との一致を最終確認する。")

    verification_targets = []
    for answer in normalized:
        question = question_by_id.get(answer["question_id"], {})
        verification_targets.append({
            "question_id": answer["question_id"],
            "category": _text(question.get("category")),
            "hypothesis": _text(question.get("hypothesis")),
            "answer_status": answer["status"],
            "verify_on_result": "成約・失注と条件履行時に、この確認が判断または成約条件を変えたか照合する。",
            "source_asset_id": _text(question.get("source_asset_id")),
        })

    category_summary = "、".join(_text(question_by_id[item["question_id"]].get("category")) for item in normalized)
    ringi_comment = (
        f"紫苑追加確認（{category_summary}）を実施。"
        f"確認済{len(confirmed)}件・一部確認{len(partials)}件・懸念{len(concerns)}件。"
        f"判断更新は「{updated}」。スコア自体は変更していない。"
    )
    return {
        "baseline_decision": baseline_decision,
        "updated_decision": updated,
        "change_reason": change,
        "answer_summary": normalized,
        "approval_conditions": approval_conditions[:3],
        "ringi_comment": ringi_comment,
        "verification_targets": verification_targets,
        "score_changed": False,
    }


def create_followup_session(
    *,
    case_id: str,
    review_id: int | None,
    form: dict[str, Any],
    result: dict[str, Any],
    judgment_assets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from api.db_connection import ensure_schema

    ensure_schema()
    normalized_case_id = _text(case_id)[:120]
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT outcome_status
              FROM shion_followup_sessions
             WHERE case_id = {ph} AND outcome_status <> ''
             ORDER BY created_at DESC
             LIMIT 1
            """,
            (normalized_case_id,),
        )
        if cur.fetchone():
            raise ValueError("case outcome already recorded; followup questions cannot be created")
    followup_id = f"SFU-{uuid.uuid4().hex[:12]}"
    questions = build_followup_questions(form, result, judgment_assets)
    baseline = _baseline_decision(result)
    values = (
        followup_id,
        normalized_case_id,
        review_id,
        baseline[:120],
        "",
        json.dumps(questions, ensure_ascii=False),
        "[]",
        "{}",
        "questions_ready",
        "",
        "",
    )
    with get_connection() as conn:
        conn.cursor().execute(
            f"""
            INSERT INTO shion_followup_sessions
                (followup_id, case_id, review_id, baseline_decision, updated_decision,
                 questions_json, answers_json, summary_json, status, outcome_status, outcome_note)
            VALUES ({', '.join([ph] * len(values))})
            """,
            values,
        )
    return {
        "followup_id": followup_id,
        "case_id": values[1],
        "review_id": review_id,
        "baseline_decision": baseline,
        "questions": questions,
        "answers": [],
        "summary": {},
        "status": "questions_ready",
    }


def answer_followup_session(followup_id: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
    from api.db_connection import ensure_schema

    ensure_schema()
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT case_id, review_id, baseline_decision, questions_json, status FROM shion_followup_sessions WHERE followup_id = {ph}",
            (_text(followup_id),),
        )
        row = cur.fetchone()
        if not row:
            raise KeyError("followup session not found")
        payload = dict(row)
        if _text(payload.get("status")).startswith("outcome_linked"):
            raise ValueError("outcome-linked followup cannot be changed")
        questions = json.loads(payload.get("questions_json") or "[]")
        expected_ids = {_text(question.get("id")) for question in questions}
        answer_ids = [_text(answer.get("question_id")) for answer in answers]
        if len(answer_ids) != len(set(answer_ids)):
            raise ValueError("duplicate question_id in answers")
        if set(answer_ids) != expected_ids:
            raise ValueError("answers must cover every followup question exactly once")
        summary = build_updated_view(_text(payload.get("baseline_decision")), questions, answers)
        normalized_answers = summary["answer_summary"]
        cur.execute(
            f"""
            UPDATE shion_followup_sessions
               SET updated_decision = {ph}, answers_json = {ph}, summary_json = {ph},
                   status = {ph}, updated_at = CURRENT_TIMESTAMP
             WHERE followup_id = {ph}
            """,
            (
                summary["updated_decision"],
                json.dumps(normalized_answers, ensure_ascii=False),
                json.dumps(summary, ensure_ascii=False),
                "answered",
                _text(followup_id),
            ),
        )
    return {
        "followup_id": _text(followup_id),
        "case_id": _text(payload.get("case_id")),
        "review_id": payload.get("review_id"),
        "baseline_decision": _text(payload.get("baseline_decision")),
        "questions": questions,
        "answers": normalized_answers,
        "summary": summary,
        "status": "answered",
    }


def list_followup_sessions(case_id: str, *, limit: int = 5) -> list[dict[str, Any]]:
    from api.db_connection import ensure_schema

    ensure_schema()
    ph = placeholder()
    capped = max(1, min(int(limit or 5), 20))
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT followup_id, case_id, review_id, baseline_decision, updated_decision,
                   questions_json, answers_json, summary_json, status,
                   outcome_status, outcome_note, created_at, updated_at
              FROM shion_followup_sessions
             WHERE case_id = {ph}
             ORDER BY created_at DESC
             LIMIT {capped}
            """,
            (_text(case_id),),
        )
        rows = cur.fetchall()
    feedback_by_followup: dict[str, list[dict[str, Any]]] = {}
    followup_ids = [_text(dict(row).get("followup_id")) for row in rows]
    if followup_ids:
        marks = ", ".join([ph] * len(followup_ids))
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT followup_id, question_id, impact_label, note, created_at, updated_at
                  FROM shion_followup_impact_feedback
                 WHERE followup_id IN ({marks})
                 ORDER BY created_at ASC
                """,
                tuple(followup_ids),
            )
            for feedback_row in cur.fetchall():
                feedback = dict(feedback_row)
                feedback_by_followup.setdefault(_text(feedback.get("followup_id")), []).append({
                    "question_id": _text(feedback.get("question_id")),
                    "impact_label": _text(feedback.get("impact_label")),
                    "impact_label_text": IMPACT_LABELS.get(_text(feedback.get("impact_label")), ""),
                    "note": _text(feedback.get("note")),
                    "created_at": str(feedback.get("created_at") or ""),
                    "updated_at": str(feedback.get("updated_at") or ""),
                })

    result = []
    for row in rows:
        item = dict(row)
        result.append({
            "followup_id": item["followup_id"],
            "case_id": item.get("case_id") or "",
            "review_id": item.get("review_id"),
            "baseline_decision": item.get("baseline_decision") or "",
            "updated_decision": item.get("updated_decision") or "",
            "questions": json.loads(item.get("questions_json") or "[]"),
            "answers": json.loads(item.get("answers_json") or "[]"),
            "summary": json.loads(item.get("summary_json") or "{}"),
            "status": item.get("status") or "",
            "outcome_status": item.get("outcome_status") or "",
            "outcome_note": item.get("outcome_note") or "",
            "impact_feedback": feedback_by_followup.get(_text(item.get("followup_id")), []),
            "created_at": str(item.get("created_at") or ""),
            "updated_at": str(item.get("updated_at") or ""),
        })
    return result


def save_followup_impact_feedback(
    followup_id: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """結果登録後の人間評価を質問単位で保存する。自動学習には接続しない。"""
    from api.db_connection import ensure_schema

    ensure_schema()
    normalized_followup_id = _text(followup_id)
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT questions_json, answers_json, status FROM shion_followup_sessions WHERE followup_id = {ph}",
            (normalized_followup_id,),
        )
        row = cur.fetchone()
        if not row:
            raise KeyError("followup session not found")
        session = dict(row)
        if not _text(session.get("status")).startswith("outcome_linked"):
            raise ValueError("impact feedback requires a recorded outcome")
        answered_ids = {
            _text(answer.get("question_id"))
            for answer in json.loads(session.get("answers_json") or "[]")
        }
        seen: set[str] = set()
        normalized_entries: list[dict[str, str]] = []
        for entry in entries:
            question_id = _text(entry.get("question_id"))
            impact_label = _text(entry.get("impact_label"))
            if not question_id or question_id in seen:
                raise ValueError("impact feedback question_id must be unique")
            if question_id not in answered_ids:
                raise ValueError("impact feedback is limited to answered questions")
            if impact_label not in IMPACT_LABELS:
                raise ValueError("invalid impact feedback label")
            seen.add(question_id)
            normalized_entries.append({
                "question_id": question_id,
                "impact_label": impact_label,
                "note": _text(entry.get("note"))[:2000],
            })

        for entry in normalized_entries:
            cur.execute(
                f"DELETE FROM shion_followup_impact_feedback WHERE followup_id = {ph} AND question_id = {ph}",
                (normalized_followup_id, entry["question_id"]),
            )
            cur.execute(
                f"""
                INSERT INTO shion_followup_impact_feedback
                    (followup_id, question_id, impact_label, note)
                VALUES ({ph}, {ph}, {ph}, {ph})
                """,
                (
                    normalized_followup_id,
                    entry["question_id"],
                    entry["impact_label"],
                    entry["note"],
                ),
            )
    return {
        "followup_id": normalized_followup_id,
        "saved_count": len(normalized_entries),
        "impact_feedback": [
            {**entry, "impact_label_text": IMPACT_LABELS[entry["impact_label"]]}
            for entry in normalized_entries
        ],
        "guardrail": "human_feedback_only_no_auto_promotion_no_scoring_change",
    }


def analyze_followup_question_impact(*, limit: int = 20) -> dict[str, Any]:
    """蓄積済みセッションから、質問別の判断・結果シグナルを説明可能に集計する。"""
    from api.db_connection import ensure_schema

    ensure_schema()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT followup_id, questions_json, answers_json, status, outcome_status
              FROM shion_followup_sessions
             ORDER BY created_at DESC, followup_id DESC
             LIMIT 5000
            """
        )
        sessions = [dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT feedback.followup_id, feedback.question_id, feedback.impact_label
              FROM shion_followup_impact_feedback AS feedback
              JOIN (
                    SELECT followup_id
                      FROM shion_followup_sessions
                     ORDER BY created_at DESC, followup_id DESC
                     LIMIT 5000
                   ) AS selected
                ON selected.followup_id = feedback.followup_id
            """
        )
        feedback_rows = [dict(row) for row in cur.fetchall()]

    feedback_map = {
        (_text(row.get("followup_id")), _text(row.get("question_id"))): _text(row.get("impact_label"))
        for row in feedback_rows
    }
    grouped: dict[str, dict[str, Any]] = {}
    for session in sessions:
        followup_id = _text(session.get("followup_id"))
        outcome = _text(session.get("outcome_status"))
        answers = {
            _text(answer.get("question_id")): answer
            for answer in json.loads(session.get("answers_json") or "[]")
        }
        for question in json.loads(session.get("questions_json") or "[]"):
            question_id = _text(question.get("id"))
            key = question_id or _text(question.get("question"))
            item = grouped.setdefault(key, {
                "question_id": question_id,
                "category": _text(question.get("category")),
                "question": _text(question.get("question")),
                "source_asset_id": _text(question.get("source_asset_id")),
                "asked_count": 0,
                "answered_count": 0,
                "condition_signal_count": 0,
                "warning_match_count": 0,
                "decision_changed_count": 0,
                "risk_prevented_count": 0,
                "outcome_matched_count": 0,
                "evidence_strengthened_count": 0,
                "not_helpful_count": 0,
            })
            item["asked_count"] += 1
            answer = answers.get(question_id)
            if answer:
                item["answered_count"] += 1
                answer_status = _text(answer.get("status"))
                if answer_status in {"partial", "concern"}:
                    item["condition_signal_count"] += 1
            impact_label = feedback_map.get((followup_id, question_id))
            if impact_label in IMPACT_LABELS:
                item[f"{impact_label}_count"] += 1
            if (
                answer
                and _text(answer.get("status")) in {"partial", "concern"}
                and outcome in _ADVERSE_OUTCOMES
                and impact_label == "outcome_matched"
            ):
                item["warning_match_count"] += 1

    rows = []
    for item in grouped.values():
        labeled_count = sum(
            int(item[f"{label}_count"])
            for label in IMPACT_LABELS
        )
        useful_count = labeled_count - int(item["not_helpful_count"])
        direct_impact_count = int(item["decision_changed_count"]) + int(item["risk_prevented_count"])
        item.update({
            "labeled_count": labeled_count,
            "useful_count": useful_count,
            "direct_impact_count": direct_impact_count,
            "usefulness_rate": round(useful_count / labeled_count, 3) if labeled_count else None,
            "evidence_level": "比較可能" if labeled_count >= 5 else ("暫定" if labeled_count else "蓄積中"),
        })
        rows.append(item)
    rows.sort(key=lambda item: (
        -int(item["labeled_count"] >= 5),
        -int(item["direct_impact_count"]),
        -int(item["useful_count"]),
        -int(item["asked_count"]),
        _text(item.get("question_id")),
    ))
    capped = max(1, min(int(limit or 20), 100))
    return {
        "session_count": len(sessions),
        "answered_session_count": sum(1 for session in sessions if json.loads(session.get("answers_json") or "[]")),
        "outcome_linked_session_count": sum(1 for session in sessions if _text(session.get("status")).startswith("outcome_linked")),
        "feedback_count": len(feedback_rows),
        "question_count": len(rows),
        "questions": rows[:capped],
        "minimum_comparable_feedback": 5,
        "guardrail": "descriptive_human_reviewed_analytics_no_auto_promotion_no_scoring_change",
    }


def record_followup_outcome(case_id: str, outcome_status: str, outcome_note: str = "") -> dict[str, Any]:
    """結果登録時に全セッションをロックし、回答済みか未回答かを保存する。"""
    from api.db_connection import ensure_schema

    ensure_schema()
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            UPDATE shion_followup_sessions
               SET outcome_status = {ph}, outcome_note = {ph}, status = {ph},
                   updated_at = CURRENT_TIMESTAMP
             WHERE case_id = {ph} AND status IN ('questions_ready', 'outcome_linked_unanswered')
            """,
            (
                _text(outcome_status)[:80],
                _text(outcome_note)[:2000],
                "outcome_linked_unanswered",
                _text(case_id),
            ),
        )
        unanswered_count = max(0, int(cur.rowcount or 0))
        cur.execute(
            f"""
            UPDATE shion_followup_sessions
               SET outcome_status = {ph}, outcome_note = {ph}, status = {ph},
                   updated_at = CURRENT_TIMESTAMP
             WHERE case_id = {ph} AND status IN ('answered', 'outcome_linked')
            """,
            (
                _text(outcome_status)[:80],
                _text(outcome_note)[:2000],
                "outcome_linked",
                _text(case_id),
            ),
        )
        answered_count = max(0, int(cur.rowcount or 0))
        updated = answered_count + unanswered_count
    impact_sessions = []
    if answered_count:
        sessions = list_followup_sessions(_text(case_id), limit=1)
        if (
            sessions
            and _text(sessions[0].get("status")).startswith("outcome_linked")
            and bool(sessions[0].get("answers"))
        ):
            impact_sessions.append(sessions[0])
    return {
        "status": "linked" if updated else "no_answered_followup",
        "case_id": _text(case_id),
        "outcome_status": _text(outcome_status),
        "linked_count": updated,
        "answered_count": answered_count,
        "unanswered_count": unanswered_count,
        "impact_sessions": impact_sessions,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "guardrail": "outcome_link_only_no_auto_promotion_no_scoring_change",
    }
