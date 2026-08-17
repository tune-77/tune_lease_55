"""Prediction-error loop for Shion judgment assets.

This layer records what Shion predicted, what happened later, which belief was
implicated, and what action should be reviewed next. It intentionally does not
change scoring, RAG ranking, prompts, or canonical judgment assets.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from api.loop_engineering_common import DATA_DIR, append_jsonl, load_jsonl
from scoring_core import APPROVAL_LINE, CONDITIONAL_LINE

EVENTS_PATH = DATA_DIR / "prediction_error_events.jsonl"
CANDIDATES_PATH = DATA_DIR / "prediction_error_update_candidates.jsonl"
REVIEWS_PATH = DATA_DIR / "prediction_error_candidate_reviews.jsonl"

VALID_FEEDBACK = {"worked", "helped", "marginal", "missed", "revised", "neutral", "challenged", "rejected"}
VALID_REVIEW_DECISIONS = {"adopted", "revised", "held", "rejected"}

_RISK_LEVELS = {
    "low": 1,
    "低": 1,
    "低リスク": 1,
    "medium": 2,
    "middle": 2,
    "中": 2,
    "中リスク": 2,
    "high": 3,
    "高": 3,
    "高リスク": 3,
}

_NEGATIVE_ACTION_TERMS = ("否決", "見送り", "不可", "reject", "decline")
_CONDITIONAL_ACTION_TERMS = ("条件", "追加確認", "要確認", "conditional")
_APPROVAL_ACTION_TERMS = ("承認", "approve")
_CONTRACTED_STATUSES = {"成約", "検収", "検収完了"}
_LOST_STATUSES = {"失注"}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _risk_rank(value: Any) -> int | None:
    text = _text(value).lower()
    if text in _RISK_LEVELS:
        return _RISK_LEVELS[text]
    try:
        numeric = float(text)
    except ValueError:
        return None
    if numeric >= 0.67:
        return 3
    if numeric >= 0.34:
        return 2
    return 1


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    folded = text.lower()
    return any(term.lower() in folded for term in terms)


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    """Keep the durable fields needed to measure a future prediction error."""
    used_assets = prediction.get("used_judgment_assets") or prediction.get("judgment_asset_ids") or []
    if isinstance(used_assets, str):
        used_assets = [used_assets]
    if not isinstance(used_assets, list):
        used_assets = []
    confidence = prediction.get("confidence")
    try:
        confidence = round(float(confidence), 3) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    return {
        "risk_level": _text(prediction.get("risk_level") or prediction.get("credit_risk")),
        "main_concern": _text(prediction.get("main_concern") or prediction.get("concern")),
        "recommended_action": _text(prediction.get("recommended_action") or prediction.get("action")),
        "confidence": confidence,
        "used_judgment_assets": [str(item).strip() for item in used_assets if str(item).strip()],
        # 予測がいつ・どこで固定されたか。Phase2 の prediction_coverage はこれを数える。
        "prediction_source": _text(prediction.get("prediction_source")) or "unspecified",
        "predicted_at": _text(prediction.get("predicted_at")),
        "snapshot_id": _text(prediction.get("snapshot_id")),
    }


def normalize_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    feedback = _text(outcome.get("user_feedback") or outcome.get("feedback") or outcome.get("result")).lower()
    return {
        "contracted": bool(outcome.get("contracted", False)),
        "delinquency": bool(outcome.get("delinquency", False) or outcome.get("defaulted", False)),
        "condition_changed": bool(outcome.get("condition_changed", False)),
        "user_feedback": feedback if feedback in VALID_FEEDBACK else _text(outcome.get("user_feedback") or outcome.get("feedback")),
        "feedback_note": _text(outcome.get("feedback_note") or outcome.get("note") or outcome.get("reason")),
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prediction_risk_from_score(score: float | None) -> str:
    if score is None:
        return ""
    if score >= APPROVAL_LINE:
        return "low"
    if score >= CONDITIONAL_LINE:
        return "medium"
    return "high"


def prediction_confidence_from_score(score: float | None) -> float | None:
    """スコアが条件付き承認ラインからどれだけ離れているかを confidence の代理にする。

    審査時点（`api/prediction_snapshot.py`）と結果登録時のフォールバックで同じ式を
    使うため、ここを唯一の定義にする。式そのものは代理指標にすぎないので、
    根拠は `confidence_basis` 側に別途残す。
    """
    if score is None:
        return None
    return min(0.95, max(0.35, abs(score - float(CONDITIONAL_LINE)) / 60.0 + 0.35))


def _action_from_decision(decision: str, score: float | None) -> str:
    text = _text(decision)
    if text:
        return text
    if score is None:
        return ""
    if score >= APPROVAL_LINE:
        return "承認"
    if score >= CONDITIONAL_LINE:
        return "条件付き承認"
    return "否決"


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    text = _text(item.get("title") or item.get("label") or item.get("text") or item.get("message"))
                else:
                    text = _text(item)
                if text:
                    return text
            continue
        text = _text(value)
        if text:
            return text
    return ""


def _collect_note_parts(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, dict):
            for key in (
                "human_discomfort",
                "but_still_reason",
                "approval_condition_memo",
                "non_negotiable_condition",
                "retrospective_note",
            ):
                text = _text(value.get(key))
                if text:
                    parts.append(text)
            continue
        if isinstance(value, list):
            parts.extend(_text(item) for item in value if _text(item))
            continue
        text = _text(value)
        if text:
            parts.append(text)
    return " / ".join(dict.fromkeys(parts))


def _snapshot_prediction(case_id: str) -> dict[str, Any] | None:
    """審査時点で固定した予測があればそれを使う（無ければ結果登録時の逆算へ落ちる）。

    遅延importにしているのは、`api/prediction_snapshot.py` 側がこのモジュールの
    ヘルパーを使っており、モジュール先頭importにすると循環参照になるため。
    """
    if not _text(case_id):
        return None
    try:
        from api.prediction_snapshot import load_prediction_snapshot

        snapshot = load_prediction_snapshot(case_id)
    except Exception:
        return None
    if not isinstance(snapshot, dict):
        return None
    prediction = snapshot.get("prediction")
    if not isinstance(prediction, dict):
        return None
    if not any(prediction.get(key) for key in ("risk_level", "main_concern", "recommended_action")):
        return None
    enriched = dict(prediction)
    enriched["prediction_source"] = "snapshot"
    enriched["predicted_at"] = snapshot.get("captured_at")
    enriched["snapshot_id"] = snapshot.get("id")
    return enriched


def build_case_result_prediction_error_payload(
    *,
    case_data: dict[str, Any],
    status: str,
    patches: dict[str, Any] | None = None,
    case_id: str = "",
) -> dict[str, Any]:
    """Build prediction/outcome payloads from a saved case and result patch.

    予測は審査時点のスナップショットを優先する。スナップショットが無い過去案件は
    従来どおり `case_data` からの逆算にフォールバックする（削除すると過去案件の
    予測誤差が測れなくなるため残す）。
    """
    patches = patches or {}
    result = case_data.get("result") if isinstance(case_data.get("result"), dict) else {}
    inputs = case_data.get("inputs") if isinstance(case_data.get("inputs"), dict) else {}
    grey = patches.get("grey_judgment") if isinstance(patches.get("grey_judgment"), dict) else {}
    if not grey and isinstance(case_data.get("grey_judgment"), dict):
        grey = case_data["grey_judgment"]

    score = _float_or_none(
        case_data.get("score")
        or case_data.get("score_base")
        or result.get("score")
        or result.get("score_base")
        or result.get("hantei_score")
    )
    hantei = _first_text(case_data.get("hantei"), result.get("hantei"), result.get("decision"))
    main_concern = _first_text(
        case_data.get("main_concern"),
        result.get("main_concern"),
        result.get("risk_summary"),
        result.get("diagnostic_recommendations"),
        result.get("conditional_approval_actions"),
        inputs.get("industry_sub"),
    )
    assets = (
        case_data.get("used_judgment_assets")
        or case_data.get("judgment_asset_ids")
        or result.get("used_judgment_assets")
        or result.get("judgment_asset_ids")
        or []
    )

    condition_changed = bool(
        patches.get("loan_conditions")
        or grey.get("approval_condition_memo")
        or grey.get("non_negotiable_condition")
        or patches.get("approval_condition_memo")
        or patches.get("non_negotiable_condition")
    )
    note = _collect_note_parts(
        patches.get("lost_reason"),
        patches.get("final_note"),
        patches.get("note"),
        patches.get("loan_conditions"),
        grey,
    )
    if patches.get("competitor_rate") or patches.get("competitor_name"):
        note = _collect_note_parts(note, "競合条件あり")

    final_status = _text(status)
    user_feedback = "revised" if note or condition_changed else "neutral"
    prediction = _snapshot_prediction(case_id or _text(case_data.get("id"))) or {
        "risk_level": _prediction_risk_from_score(score),
        "main_concern": main_concern,
        "recommended_action": _action_from_decision(hantei, score),
        "confidence": prediction_confidence_from_score(score),
        "used_judgment_assets": assets,
        "prediction_source": "reconstructed_at_result",
    }
    return {
        "prediction": prediction,
        "outcome": {
            "contracted": final_status in _CONTRACTED_STATUSES,
            "delinquency": final_status in {"延滞", "事故", "デフォルト"},
            "condition_changed": condition_changed,
            "user_feedback": user_feedback,
            "feedback_note": note,
        },
    }


def record_case_result_prediction_error(
    *,
    case_id: str,
    case_data: dict[str, Any],
    status: str,
    patches: dict[str, Any] | None = None,
    source: str = "case_result_registration",
    events_path: Path = EVENTS_PATH,
    candidates_path: Path = CANDIDATES_PATH,
) -> dict[str, Any]:
    final_status = _text(status)
    if final_status not in _CONTRACTED_STATUSES and final_status not in _LOST_STATUSES:
        return {"status": "skipped", "reason": "status_not_observable_for_prediction_error"}
    payload = build_case_result_prediction_error_payload(
        case_data=case_data,
        status=final_status,
        patches=patches,
        case_id=case_id,
    )
    prediction = payload["prediction"]
    if not any(prediction.get(key) for key in ("risk_level", "main_concern", "recommended_action")):
        return {"status": "skipped", "reason": "prediction_fields_unavailable"}
    return record_prediction_error_event(
        case_id=case_id,
        prediction=prediction,
        outcome=payload["outcome"],
        source=source,
        events_path=events_path,
        candidates_path=candidates_path,
    )


def classify_prediction_error(prediction: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    """Classify the observable gap between prior prediction and later outcome."""
    pred = normalize_prediction(prediction)
    obs = normalize_outcome(outcome)
    note = obs["feedback_note"]
    action = pred["recommended_action"]
    risk = _risk_rank(pred["risk_level"])
    feedback = obs["user_feedback"].lower()

    if _contains_any(note, ("競合", "価格", "成約", "営業", "銀行")):
        error_type = "sales_context_miss"
        description = "信用リスクではなく、営業・競合・成約文脈の見立てがずれた"
    elif obs["delinquency"] and (risk is None or risk <= 2 or _contains_any(action, _APPROVAL_ACTION_TERMS)):
        error_type = "risk_underestimated"
        description = "安全寄りに見たが、延滞・事故などの悪化が出た"
    elif obs["contracted"] and not obs["delinquency"] and (
        risk == 3 or _contains_any(action, _NEGATIVE_ACTION_TERMS)
    ):
        error_type = "risk_overestimated"
        description = "危険寄りに見たが、成約後の悪化は確認されなかった"
    elif obs["condition_changed"] or _contains_any(note, ("条件", "担保", "保証", "前受", "期間", "金利")):
        error_type = "condition_setting_miss"
        description = "判断の方向よりも、条件設定の強さや具体性がずれた"
    elif _contains_any(note, ("確認", "聞く", "質問", "抜け", "不足", "ヒアリング")):
        error_type = "confirmation_gap"
        description = "聞くべき確認点または優先順位が不足していた"
    elif _contains_any(note, ("説明", "理由", "使い", "稟議", "文面")) or feedback in {"marginal", "revised"}:
        error_type = "explanation_gap"
        description = "判断自体よりも、人間が使える説明への変換が弱かった"
    else:
        error_type = "general_prediction_error"
        description = "予測・結果・人間評価の間に分類未確定のずれがある"

    severity = "high" if error_type == "risk_underestimated" else "medium"
    if feedback in {"worked", "helped", "neutral"} and error_type == "general_prediction_error":
        severity = "low"

    return {
        "type": error_type,
        "description": description,
        "severity": severity,
    }


def infer_target_belief(prediction: dict[str, Any], error: dict[str, Any]) -> str:
    pred = normalize_prediction(prediction)
    if pred["used_judgment_assets"]:
        return pred["used_judgment_assets"][0]
    if pred["main_concern"]:
        return f"concern:{pred['main_concern']}"
    return f"unmapped:{error['type']}"


def _existing_repeat_count(
    *,
    target_belief: str,
    error_type: str,
    candidates_path: Path = CANDIDATES_PATH,
) -> int:
    count = 0
    for row in load_jsonl(candidates_path, newest_first=False):
        if row.get("target_belief") == target_belief and row.get("error_type") == error_type:
            count += 1
    return count


def build_update_candidate(
    *,
    event: dict[str, Any],
    review_threshold: int = 3,
    candidates_path: Path = CANDIDATES_PATH,
) -> dict[str, Any]:
    prediction = event["prediction"]
    outcome = event["outcome"]
    error = event["prediction_error"]
    target_belief = infer_target_belief(prediction, error)
    repeat_count = _existing_repeat_count(
        target_belief=target_belief,
        error_type=error["type"],
        candidates_path=candidates_path,
    ) + 1

    if error["type"] == "risk_underestimated":
        suggested_update = "安全寄り判断の前提を見直し、延滞・事故につながった未確認要因を次回の優先確認に入れる。"
        next_action_kind = "add_confirmation_question"
    elif error["type"] == "risk_overestimated":
        suggested_update = "危険視した根拠を分解し、一時要因・補強材料・解消条件がある場合は懸念を条件付きに弱める。"
        next_action_kind = "revise_risk_weight_candidate"
    elif error["type"] == "condition_setting_miss":
        suggested_update = "条件付き承認の条件文を、担保・保証・入金予定・期間・金利など確認可能な形へ具体化する。"
        next_action_kind = "revise_condition_template"
    elif error["type"] == "sales_context_miss":
        suggested_update = "信用リスクと営業・競合・成約リスクを分け、価格・競合・銀行支援を別ラベルで確認する。"
        next_action_kind = "split_risk_origin"
    elif error["type"] == "confirmation_gap":
        suggested_update = "次回案件で最初に聞く確認質問を1つ追加し、確認済みなら懸念を弱める分岐を作る。"
        next_action_kind = "add_confirmation_question"
    else:
        suggested_update = "判断理由を人間が使える稟議・確認文へ変換できるよう、説明粒度を修正候補にする。"
        next_action_kind = "revise_explanation_template"

    return {
        "candidate_id": f"PEC-{event['event_id']}",
        "source_event_id": event["event_id"],
        "case_id": event["case_id"],
        "target_belief": target_belief,
        "error_type": error["type"],
        "severity": error["severity"],
        "repeat_count": repeat_count,
        "status": "ready_for_review" if repeat_count >= max(1, review_threshold) else "candidate",
        "suggested_update": suggested_update,
        "next_action": {
            "kind": next_action_kind,
            "requires_human_review": True,
            "no_automatic_scoring_or_rag_change": True,
        },
        "evidence": {
            "main_concern": prediction.get("main_concern", ""),
            "recommended_action": prediction.get("recommended_action", ""),
            "user_feedback": outcome.get("user_feedback", ""),
            "feedback_note": outcome.get("feedback_note", ""),
        },
        "created_at": event["recorded_at"],
    }


def record_prediction_error_event(
    *,
    case_id: str,
    prediction: dict[str, Any],
    outcome: dict[str, Any],
    source: str = "manual",
    events_path: Path = EVENTS_PATH,
    candidates_path: Path = CANDIDATES_PATH,
    review_threshold: int = 3,
) -> dict[str, Any]:
    clean_case_id = _text(case_id)
    if not clean_case_id:
        raise ValueError("case_id is required")
    pred = normalize_prediction(prediction)
    obs = normalize_outcome(outcome)
    error = classify_prediction_error(pred, obs)
    recorded_at = _now_iso()
    event_id = f"pe-{recorded_at.replace(':', '').replace('+', 'Z')}-{clean_case_id}"
    event = {
        "event_id": event_id,
        "case_id": clean_case_id,
        "source": _text(source) or "manual",
        "prediction": pred,
        "outcome": obs,
        "prediction_error": error,
        "recorded_at": recorded_at,
    }
    candidate = build_update_candidate(
        event=event,
        review_threshold=review_threshold,
        candidates_path=candidates_path,
    )
    append_jsonl(events_path, event)
    append_jsonl(candidates_path, candidate)
    return {"status": "recorded", "event": event, "candidate": candidate}


def load_update_candidates(
    *,
    limit: int = 50,
    status: str | None = None,
    candidates_path: Path = CANDIDATES_PATH,
    reviews_path: Path = REVIEWS_PATH,
) -> list[dict[str, Any]]:
    rows = _apply_latest_reviews(load_jsonl(candidates_path, limit=None, newest_first=True), reviews_path=reviews_path)
    if status:
        rows = [row for row in rows if row.get("status") == status]
    return rows[: max(1, min(int(limit or 50), 200))]


def _latest_reviews(reviews_path: Path = REVIEWS_PATH) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(reviews_path, newest_first=False):
        candidate_id = _text(row.get("candidate_id"))
        if candidate_id:
            latest[candidate_id] = row
    return latest


def _apply_latest_reviews(rows: list[dict[str, Any]], *, reviews_path: Path = REVIEWS_PATH) -> list[dict[str, Any]]:
    latest = _latest_reviews(reviews_path)
    result: list[dict[str, Any]] = []
    for row in rows:
        candidate = dict(row)
        review = latest.get(str(candidate.get("candidate_id") or ""))
        if review:
            candidate["review"] = review
            candidate["review_decision"] = review.get("decision")
            candidate["review_note"] = review.get("note", "")
            candidate["reviewed_at"] = review.get("reviewed_at", "")
            candidate["status"] = str(review.get("decision") or candidate.get("status"))
        result.append(candidate)
    return result


def record_update_candidate_review(
    *,
    candidate_id: str,
    decision: str,
    note: str = "",
    edited_update: str = "",
    reviewer: str = "user",
    reviews_path: Path = REVIEWS_PATH,
    candidates_path: Path = CANDIDATES_PATH,
) -> dict[str, Any]:
    clean_candidate_id = _text(candidate_id)
    clean_decision = _text(decision)
    if not clean_candidate_id:
        raise ValueError("candidate_id is required")
    if clean_decision not in VALID_REVIEW_DECISIONS:
        raise ValueError("decision must be adopted, revised, held, or rejected")

    candidates = load_jsonl(candidates_path, newest_first=False)
    source = next((item for item in candidates if item.get("candidate_id") == clean_candidate_id), None)
    if not source:
        raise ValueError("candidate not found")

    review = {
        "candidate_id": clean_candidate_id,
        "source_event_id": source.get("source_event_id", ""),
        "case_id": source.get("case_id", ""),
        "target_belief": source.get("target_belief", ""),
        "error_type": source.get("error_type", ""),
        "decision": clean_decision,
        "note": _text(note),
        "edited_update": _text(edited_update),
        "reviewer": _text(reviewer) or "user",
        "reviewed_at": _now_iso(),
        "policy": "human_review_only_no_automatic_scoring_or_rag_change",
    }
    append_jsonl(reviews_path, review)
    return {"status": "recorded", "review": review}


def summarize_prediction_errors(
    *,
    events_path: Path = EVENTS_PATH,
    candidates_path: Path = CANDIDATES_PATH,
    reviews_path: Path = REVIEWS_PATH,
) -> dict[str, Any]:
    events = load_jsonl(events_path, newest_first=False)
    candidates = _apply_latest_reviews(load_jsonl(candidates_path, newest_first=False), reviews_path=reviews_path)
    by_type: dict[str, int] = {}
    by_belief: dict[str, int] = {}
    by_review: dict[str, int] = {}
    for event in events:
        error_type = str((event.get("prediction_error") or {}).get("type") or "unknown")
        by_type[error_type] = by_type.get(error_type, 0) + 1
    for candidate in candidates:
        belief = str(candidate.get("target_belief") or "unknown")
        by_belief[belief] = by_belief.get(belief, 0) + 1
        if candidate.get("review_decision"):
            decision = str(candidate.get("review_decision"))
            by_review[decision] = by_review.get(decision, 0) + 1
    return {
        "total_events": len(events),
        "total_candidates": len(candidates),
        "ready_for_review": sum(1 for item in candidates if item.get("status") == "ready_for_review"),
        "review_counts": by_review,
        "by_error_type": by_type,
        "top_target_beliefs": sorted(
            [{"target_belief": key, "count": value} for key, value in by_belief.items()],
            key=lambda item: item["count"],
            reverse=True,
        )[:10],
        "guardrail": "records_candidates_only_no_scoring_no_rag_no_prompt_change",
    }


def event_to_json(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, sort_keys=True)
