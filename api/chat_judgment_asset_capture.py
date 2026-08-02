"""Chat-side judgment asset capture helpers."""

from __future__ import annotations

import datetime as _dt
import hashlib as _hashlib
import json
from pathlib import Path
from typing import Any, Callable


CHAT_JUDGMENT_ASSET_TRIGGERS = (
    "審査では",
    "判断方法",
    "判断基準",
    "判断資産",
    "稟議では",
    "条件付き承認",
    "否決",
    "承認",
)
CHAT_JUDGMENT_ASSET_ACTIONS = (
    "見る",
    "確認",
    "注意",
    "警戒",
    "疑う",
    "止める",
    "登録",
    "残す",
)


class JudgmentAssetCandidateValidationError(ValueError):
    """Raised when a manual judgment asset candidate request is invalid."""


def load_autoresearch_judgment_asset_candidates(
    *,
    candidates_jsonl: Path,
    candidate_state_json: Path,
    limit: int = 500,
) -> list[dict[str, Any]]:
    state: dict[str, Any] = {}
    if candidate_state_json.exists():
        try:
            raw_state = json.loads(candidate_state_json.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(raw_state, dict):
                state = raw_state
        except (json.JSONDecodeError, OSError):
            state = {}
    rows: list[dict[str, Any]] = []
    if candidates_jsonl.exists():
        try:
            for line in candidates_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    candidate_state = state.get(str(item.get("id") or ""))
                    if isinstance(candidate_state, dict):
                        item = {**item, **candidate_state}
                    rows.append(item)
                    if len(rows) >= limit:
                        break
        except OSError:
            rows = []
    demo_candidates = [
        {
            "asset_quality": "actionable",
            "candidate_type": "condition_signal",
            "claim": "更新設備の申込では、既存設備の稼働実績と受注増の根拠を並べ、増額後も返済原資が説明できるかを確認する。",
            "edited_claim": "更新設備の申込では、既存設備の稼働実績と受注増の根拠を並べ、増額後も返済原資が説明できるかを確認する。",
            "effective_claim": "更新設備の申込では、既存設備の稼働実績と受注増の根拠を並べ、増額後も返済原資が説明できるかを確認する。",
            "edit_count": 1,
            "evidence_path": "manual://screening/demo-renewal-asset-candidate",
            "id": "demo-renewal-asset-candidate",
            "promotion_status": "not_promoted",
            "research_date": "2026-07-18",
            "research_title": "Manual Judgment Asset",
            "research_topic": "demo-renewal-asset",
            "review_status": "candidate",
            "source_section": "manual_input",
            "use_count": 0,
            "useful_count": 0,
            "rejected_count": 0,
            "verified_status": "unverified",
            "verification_note": "demo_candidate_for_screening_review",
        },
    ]
    for item in reversed(demo_candidates):
        demo_id = str(item.get("id") or "")
        rows = [row for row in rows if str(row.get("id") or "") != demo_id]
        rows.insert(0, item)
    return rows[:limit]


def create_manual_judgment_asset_candidate(
    req: Any,
    *,
    candidates_jsonl: Path,
    candidate_state_json: Path,
) -> dict[str, Any]:
    claim = str(req.claim or "").strip()
    if len(claim) < 8:
        raise JudgmentAssetCandidateValidationError("claim must be at least 8 characters")
    now = _dt.datetime.now(_dt.timezone.utc)
    now_iso = now.isoformat()
    topic = str(req.research_topic or "manual").strip() or "manual"
    candidate_type = str(req.candidate_type or "confirmation_question")
    seed = "|".join(["manual", now_iso, candidate_type, topic, claim, str(req.case_id or ""), str(req.review_id or "")])
    candidate_id = _hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    row = {
        "id": candidate_id,
        "candidate_type": candidate_type,
        "research_topic": topic,
        "research_title": "Manual Judgment Asset",
        "research_date": now.date().isoformat(),
        "claim": claim,
        "effective_claim": claim,
        "edited_claim": claim,
        "edit_count": 1,
        "last_edited_at": now_iso,
        "source_section": "manual_input",
        "evidence_path": f"manual://screening/{str(req.case_id or '')[:80]}",
        "review_status": "candidate",
        "asset_quality": "actionable",
        "quality_reasons": [],
        "promotion_status": "not_promoted",
        "use_count": 0,
        "useful_count": 0,
        "rejected_count": 0,
        "neutral_count": 0,
        "last_used_at": "",
        "last_feedback_at": "",
        "verified_status": "unverified",
        "verification_note": "manual_candidate_created",
        "requires_human_use_feedback": True,
        "requires_result_verification": True,
        "manual_created_at": now_iso,
        "manual_case_id": str(req.case_id or "")[:120],
        "manual_review_id": int(req.review_id or 0) if req.review_id else 0,
        "use_policy": "人間が追加した判断資産候補。案件レビューで使い、効いた/外したと結果検証で育てる。",
    }
    candidates_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with candidates_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        from scripts.build_autoresearch_judgment_asset_candidates import load_state, write_state

        state = load_state(candidate_state_json)
        state[candidate_id] = {
            "use_count": 0,
            "useful_count": 0,
            "rejected_count": 0,
            "neutral_count": 0,
            "last_used_at": "",
            "last_feedback_at": "",
            "verified_status": "unverified",
            "verification_note": "manual_candidate_created",
            "edited_claim": claim,
            "edit_count": 1,
            "last_edited_at": now_iso,
        }
        write_state(candidate_state_json, [{"id": candidate_id, **state[candidate_id]}], state)
    except Exception:
        pass
    return row


def extract_chat_judgment_asset_claim(message: str) -> str:
    text = " ".join(str(message or "").strip().split())
    if len(text) < 12 or len(text) > 600:
        return ""
    prompt_noise_markers = (
        "【審査分析画面からの紫苑レビュー依頼】",
        "【Vertex補助検索ヒント】",
        "この案件を、審査担当者の横にいる紫苑としてレビューしてください。",
    )
    if any(marker in text for marker in prompt_noise_markers):
        return ""
    if text.endswith("?") or text.endswith("？"):
        return ""
    if not any(trigger in text for trigger in CHAT_JUDGMENT_ASSET_TRIGGERS):
        return ""
    if not any(action in text for action in CHAT_JUDGMENT_ASSET_ACTIONS):
        return ""
    weak_markers = (
        "どう思う",
        "教えて",
        "とは",
        "なに",
        "何",
        "わからない",
        "分からない",
    )
    if any(marker in text for marker in weak_markers) and not any(marker in text for marker in ("判断資産に登録", "判断資産として")):
        return ""
    return text[:500]


def chat_judgment_asset_candidate_type(claim: str) -> str:
    if any(marker in claim for marker in ("注意", "警戒", "疑う", "止める", "危険")):
        return "caution"
    if any(marker in claim for marker in ("条件", "兆候", "サイン", "なら")):
        return "condition_signal"
    if any(marker in claim for marker in ("確認", "質問", "聞く")):
        return "confirmation_question"
    return "application_rule"


def capture_chat_judgment_asset_if_needed(
    message: str,
    *,
    user_id: str,
    surface: str,
    response_mode: str = "",
    candidates_loader: Callable[..., list[dict[str, Any]]],
    candidate_creator: Callable[[Any], dict[str, Any]],
    request_factory: Callable[..., Any],
    cloudrun_event_recorder: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    claim = extract_chat_judgment_asset_claim(message)
    if not claim:
        return {"captured": False, "reason": "not_judgment_asset_teaching"}
    try:
        normalized_claim = " ".join(claim.split())
        for item in candidates_loader(limit=1000):
            existing_claim = " ".join(str(item.get("edited_claim") or item.get("claim") or "").split())
            if existing_claim == normalized_claim:
                return {
                    "captured": True,
                    "duplicate": True,
                    "candidate": {
                        "id": item.get("id"),
                        "claim": item.get("claim"),
                        "candidate_type": item.get("candidate_type"),
                        "research_topic": item.get("research_topic"),
                    },
                }
        entry = candidate_creator(
            request_factory(
                claim=claim,
                candidate_type=chat_judgment_asset_candidate_type(claim),
                research_topic="chat_judgment_teaching",
                case_id=f"chat:{user_id}",
            )
        )
        writeback = cloudrun_event_recorder(
            event_type="judgment_asset_candidate_chat_capture",
            surface=surface,
            payload={
                **entry,
                "schema_version": 1,
                "user_id": user_id,
                "response_mode": response_mode,
                "source_message": claim,
            },
        )
        return {
            "captured": True,
            "candidate": {
                "id": entry.get("id"),
                "claim": entry.get("claim"),
                "candidate_type": entry.get("candidate_type"),
                "research_topic": entry.get("research_topic"),
            },
            "writeback": writeback,
        }
    except Exception as exc:
        print(f"[判断資産チャット登録] エラー: {exc}")
        return {"captured": False, "reason": str(exc)[:200], "claim": claim}
