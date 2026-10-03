"""Chat-side judgment asset capture helpers."""

from __future__ import annotations

import datetime as _dt
import hashlib as _hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable

from api.judgment_policy import classify_knowledge_kind


# チャット由来の候補は、人が /judgment-review で確かめるまで要確認にしておく（自動昇格しない）。
CHAT_TEACHING_TOPIC = "chat_judgment_teaching"
CHAT_TEACHING_PROMOTION_STATUS = "needs_review_quality"


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
    rows = [row for row in rows if str(row.get("id") or "") != "demo-renewal-asset-candidate"]
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
        "research_date": str(getattr(req, "research_date", "") or "") or now.date().isoformat(),
        "claim": claim,
        "knowledge_kind": classify_knowledge_kind(claim),  # 方針（社内ルール）/知見。想起時に方針を結論として効かせる
        "effective_claim": claim,
        "edited_claim": claim,
        "edit_count": 1,
        "last_edited_at": now_iso,
        "source_section": "manual_input",
        "evidence_path": f"manual://screening/{str(req.case_id or '')[:80]}",
        "review_status": "candidate",
        "asset_quality": "actionable",
        "quality_reasons": [],
        "promotion_status": CHAT_TEACHING_PROMOTION_STATUS if topic == CHAT_TEACHING_TOPIC else "not_promoted",
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
    """ノウハウ教示なら保存する本文を、そうでなければ空文字を返す。

    判定は ``memory_promotion_policy.classify_lease_teaching`` の1段だけで行う。
    以前のトリガー語×行動語の二段判定と「何」「とは」の部分一致除外は、
    「ということは」のような普通の文まで落としていたため廃止した。
    """
    from memory_promotion_policy import classify_lease_teaching

    text = " ".join(str(message or "").strip().split())
    if not classify_lease_teaching(text)[0]:
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


# --- Jev shadow（記録のみ。保存判定は変えない） ---------------------------------
# 2026-10-02 オフライン評価（昇格62/却下41のチャット由来候補）で、Jevの「審査ノウハウか」は
# AUC 0.83 [0.74–0.91]、キーワードルールの理由別は 0.55。将来「低確信は要確認・愚痴は保存しない」
# に使う前提で、まず確率だけを REV-424 判定ログへ残す。CHAT_TEACHING_JEV_SHADOW=0 で停止。
CHAT_TEACHING_JEV_QUESTION = {
    "type": "noul",
    "instructions": "`items[0]` は、リース審査の担当者が教えた『判断資産として保存すべき審査ノウハウ』か？",
    "criteria": {
        "true": "他の案件でも再利用できる審査上の判断基準・注意点・確認事項・業界/物件の知見を、言い切りまたは理由付きで述べている。",
        "false": "愚痴・感想・雑談・質問・依頼・システムやAIへの指示、または特定案件の状況報告だけで、他案件に使える審査判断を含まない。",
    },
}
_ORG_RE = re.compile(
    r"(株式会社|有限会社|合同会社|\(株\)|（株）|\(有\)|（有）)\s*[^\s、。,，「」]{1,15}|[^\s、。,，「」]{1,15}(株式会社|有限会社|合同会社)"
)
_PERSON_RE = re.compile(r"[一-龥ァ-ヶ]{1,6}(さん|様|氏|社長|部長|課長)")


def mask_for_jev(text: str) -> str:
    """企業名・人名らしき部分を伏せる。PII様の内容が残れば空文字（送らない）。"""
    from jev_safe_gateway import _find_sensitive

    masked = _PERSON_RE.sub("〈人物〉", _ORG_RE.sub("〈企業〉", text))
    masked = " ".join(masked.split())[:400]
    return "" if _find_sensitive(masked) else masked


def judge_chat_teaching_with_jev(claim: str) -> float | None:
    """Jevの「審査ノウハウか」確率を返す。送らない・不通・不正応答なら None（呼び出し側はルールのみで動く）。"""
    masked = mask_for_jev(claim)
    if not masked:
        return None
    try:
        import typesafe_dedup_guard as transport

        body = transport._default_request(
            {"state": {"items": [masked]}, "model": "jev-latest", "questions": {"knowhow": CHAT_TEACHING_JEV_QUESTION}}
        )
        value = float(body["answers"]["knowhow"]["noul"])
    except Exception as exc:  # noqa: BLE001
        print(f"[判断資産チャット登録] Jev shadow スキップ: {type(exc).__name__}")
        return None
    return value if 0.0 <= value <= 1.0 else None


def record_chat_teaching_jev_shadow(claim: str, *, judge: Callable[[str], float | None] = judge_chat_teaching_with_jev) -> None:
    probability = judge(claim)
    if probability is None:
        return
    import jev_judgment_log

    jev_judgment_log.append_records(
        jev_judgment_log.build_records(
            guard="chat_teaching_capture",
            run_id=jev_judgment_log.new_run_id(),
            mode="shadow",
            model="jev-latest",
            items=[
                {
                    "question": "chat_teaching_knowhow",
                    "subject": claim,
                    "probability": round(probability, 3),
                    "choice": probability >= 0.5,
                    "route": "saved_by_rule",
                    "thresholds": {"knowhow_min": 0.5},
                    "auto_passed": True,
                }
            ],
        )
    )


def _start_jev_shadow(claim: str) -> None:
    """回答を待たせないよう別スレッドで記録する。"""
    if os.environ.get("CHAT_TEACHING_JEV_SHADOW", "1").strip() == "0":
        return
    threading.Thread(target=record_chat_teaching_jev_shadow, args=(claim,), daemon=True).start()


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
    jev_shadow: Callable[[str], None] = _start_jev_shadow,
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
                research_topic=CHAT_TEACHING_TOPIC,
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
        try:
            jev_shadow(claim)
        except Exception as exc:  # noqa: BLE001 — shadowの失敗で保存結果を変えない
            print(f"[判断資産チャット登録] Jev shadow 起動失敗: {exc}")
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
