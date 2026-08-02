"""Chat language-material preservation and response-impact prediction."""

from __future__ import annotations

import datetime as _dt
import hashlib as _hashlib
import json
from pathlib import Path
from typing import Any, Callable


Redactor = Callable[..., str]
CloudRunRecorder = Callable[..., dict[str, Any]]


def capture_language_judgment_material(
    message: str,
    *,
    user_id: str,
    surface: str,
    language_materials_jsonl: Path,
    redactor: Redactor,
    cloudrun_event_recorder: CloudRunRecorder,
    response_mode: str = "",
    category: str = "",
) -> dict[str, Any]:
    """Preserve a user utterance as raw judgment-asset material."""
    raw_text = str(message or "").strip()
    if not raw_text:
        return {"captured": False, "reason": "empty_message"}
    preserved_text = redactor(raw_text, limit=1200)
    if not preserved_text:
        return {"captured": False, "reason": "empty_after_redaction"}
    now = _dt.datetime.now(_dt.timezone.utc)
    fingerprint = _hashlib.sha256(
        "\n".join([
            str(user_id or "default")[:80],
            str(surface or "chat")[:80],
            preserved_text,
        ]).encode("utf-8")
    ).hexdigest()
    entry = {
        "id": fingerprint[:16],
        "schema_version": 1,
        "asset_stage": "raw_language_material",
        "status": "preserved",
        "promotion_status": "not_promoted",
        "review_required": True,
        "source": "chat_user_message",
        "surface": str(surface or "chat")[:80],
        "user_id": str(user_id or "default")[:80],
        "response_mode": str(response_mode or "")[:40],
        "category": str(category or "")[:80],
        "captured_at": now.isoformat(),
        "text": preserved_text,
        "text_length": len(preserved_text),
        "fingerprint": fingerprint,
        "use_policy": (
            "言葉を一言たりとも無駄にしないための原文保全。"
            "判断資産候補・判断資産・紫苑中枢へ進めるには人間レビューを必須にする。"
        ),
    }
    try:
        language_materials_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with language_materials_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        writeback = cloudrun_event_recorder(
            event_type="language_judgment_material_preserved",
            surface=surface,
            payload=entry,
        )
        return {
            "captured": True,
            "material": {
                "id": entry["id"],
                "asset_stage": entry["asset_stage"],
                "status": entry["status"],
                "promotion_status": entry["promotion_status"],
            },
            "writeback": writeback,
        }
    except Exception as exc:
        print(f"[言葉判断資産原材料] 記録エラー: {exc}")
        return {"captured": False, "reason": str(exc)[:200]}


def score_response_impact(user_message: str, assistant_reply: str) -> dict[str, Any]:
    """Predict how Shion's wording may land, without calling an LLM."""
    user_text = str(user_message or "")
    reply = str(assistant_reply or "")
    reply_len = len(reply)
    risk_factors: list[str] = []
    positive_factors: list[str] = []
    rewrite_hints: list[str] = []

    strong_markers = ("絶対", "必ず", "間違いなく", "断言", "当然", "ありえない", "絶対に")
    negation_markers = ("違う", "ダメ", "無理", "危険", "やめた方がいい", "やめるべき", "誤り", "間違い")
    softening_markers = ("ただし", "一方で", "可能性", "かもしれ", "要確認", "前提", "状況次第", "現時点")
    action_markers = ("次", "まず", "確認", "保存", "検証", "見る", "進める", "残す")
    empathy_markers = ("わかる", "その通り", "大事", "怖さ", "受け止め", "懸念", "不安", "迷い")
    user_sensitive_markers = ("怖", "不安", "つら", "苦しい", "怒", "失敗", "危険", "大丈夫", "やめ")

    if reply_len > 900:
        risk_factors.append("回答が長く、要点が埋もれる可能性")
        rewrite_hints.append("結論・理由・次の一手を短く分ける")
    if reply_len < 24:
        risk_factors.append("短すぎて、相手の言葉を受け止めていない印象になる可能性")
        rewrite_hints.append("相手の意図を一文だけ受け止めてから結論を置く")
    if any(marker in reply for marker in strong_markers):
        risk_factors.append("断定が強く、過信または押し付けに見える可能性")
        rewrite_hints.append("断定を維持する場合も、前提条件か検証条件を添える")
    if any(marker in reply for marker in negation_markers) and not any(marker in reply for marker in softening_markers):
        risk_factors.append("否定が強く、突き放された印象になる可能性")
        rewrite_hints.append("否定の前に、相手が何を守ろうとしているかを短く受け止める")
    if any(marker in user_text for marker in user_sensitive_markers) and not any(marker in reply for marker in empathy_markers):
        risk_factors.append("相手の怖さや不安への受け止めが薄い可能性")
        rewrite_hints.append("怖さや懸念を否定せず、確認・検疫・説明責任へ変換する")
    if "?" in reply or "？" in reply:
        risk_factors.append("質問返しで終わると、相手に判断を戻しすぎる可能性")
        rewrite_hints.append("質問する場合も、紫苑側の仮判断と次の一手を先に置く")
    if any(marker in reply for marker in softening_markers):
        positive_factors.append("前提や不確実性を残しており、過信リスクを下げている")
    if any(marker in reply for marker in action_markers):
        positive_factors.append("次の行動に移しやすい")
    if any(marker in reply for marker in empathy_markers):
        positive_factors.append("相手の言葉や懸念を受け止めている")

    if len(risk_factors) >= 3:
        reaction_risk = "high"
        predicted_reaction = "内容は伝わるが、強さ・長さ・受け止め不足で反発や疲れが出る可能性"
    elif len(risk_factors) >= 1:
        reaction_risk = "medium"
        predicted_reaction = "概ね受け取られるが、一部で誤解・圧・突き放し感が出る可能性"
    else:
        reaction_risk = "low"
        predicted_reaction = "相手は要点を受け取りやすく、次の行動に移りやすい可能性"

    if not rewrite_hints:
        rewrite_hints.append("結論は維持し、短く次の一手を残す")

    return {
        "predicted_reaction": predicted_reaction,
        "reaction_risk": reaction_risk,
        "risk_factors": risk_factors,
        "positive_factors": positive_factors,
        "better_reply_policy": rewrite_hints[0],
        "rewrite_hints": rewrite_hints[:5],
        "shadow_mode": True,
    }


def record_response_impact_prediction(
    *,
    user_message: str,
    assistant_reply: str,
    user_id: str,
    surface: str,
    response_impact_predictions_jsonl: Path,
    redactor: Redactor,
    cloudrun_event_recorder: CloudRunRecorder,
    response_mode: str = "",
    category: str = "",
) -> dict[str, Any]:
    try:
        prediction = score_response_impact(user_message, assistant_reply)
        now = _dt.datetime.now(_dt.timezone.utc)
        user_preview = redactor(user_message, limit=360)
        reply_preview = redactor(assistant_reply, limit=700)
        fingerprint = _hashlib.sha256(
            "\n".join([
                str(user_id or "default")[:80],
                str(surface or "chat")[:80],
                user_preview,
                reply_preview,
            ]).encode("utf-8")
        ).hexdigest()
        entry = {
            "id": fingerprint[:16],
            "schema_version": 1,
            "event_type": "response_impact_prediction",
            "status": "shadow_only",
            "source": "shion_reply_self_check",
            "surface": str(surface or "chat")[:80],
            "user_id": str(user_id or "default")[:80],
            "response_mode": str(response_mode or "")[:40],
            "category": str(category or "")[:80],
            "captured_at": now.isoformat(),
            "user_message_preview": user_preview,
            "assistant_reply_preview": reply_preview,
            "fingerprint": fingerprint,
            **prediction,
            "use_policy": (
                "自分の言葉が相手にどう響くかを事前点検するshadow mode。"
                "使うが従わない、予測するが迎合しない。"
                "相手操作ではなく、必要な厳しさを責任ある言葉に整え、言葉のQリスク、誤解、過信、突き放し感を下げるために使う。"
            ),
        }
        response_impact_predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with response_impact_predictions_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        writeback = cloudrun_event_recorder(
            event_type="response_impact_prediction",
            surface=surface,
            payload=entry,
        )
        return {
            "captured": True,
            "prediction": {
                "id": entry["id"],
                "reaction_risk": entry["reaction_risk"],
                "predicted_reaction": entry["predicted_reaction"],
                "risk_factors": entry["risk_factors"][:4],
                "better_reply_policy": entry["better_reply_policy"],
                "shadow_mode": True,
            },
            "writeback": writeback,
        }
    except Exception as exc:
        print(f"[ResponseImpactPredictor] 記録エラー: {exc}")
        return {"captured": False, "reason": str(exc)[:200]}
