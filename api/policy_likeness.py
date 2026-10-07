"""判断資産の「方針らしさ」を Jev の確率で3段階に分ける（shadow: 回答には使わない）。

ユーザーいわく方針と知見に「明確な仕切りはない」ので、二値で切らず確率を方針らしさとして扱う。
  - policy（社内方針）: 決定的ルール（judgment_policy）が方針 / ユーザーが方針と決めた / Jev が POLICY_MIN 以上
  - guideline（社内の目安）: GUIDELINE_MIN 以上。回答に含めても冒頭で断定しない想定。/judgment-review で1タップ判定
  - insight（知見）: それ未満
誤って方針扱いして答えを断定しすぎる害の方が大きいので、閾値は保守的にする。
2026-10-04 の測定（experiments/policy_kind_jev/、ユーザー正解16件で AUC 0.81・95%CI 0.52–1.0）では
文単位の Jev の最大値は 0.88。資産単位の初回採点（active 69件）では 0.9 以上が4件（うち2件はユーザーが方針と判定した文）、
社内の目安16件、知見48件だった。

保存先はローカルの data/policy_likeness_queue.json（資産IDごと）。判定は jev_judgment_log に
guard=knowledge_kind_policy, mode=shadow で残し、ユーザーの1タップを human ラベルとして追記する。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from api.judgment_policy import POLICY, classify_knowledge_kind
from silent_failure_log import record_silent_failure

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_JSON = REPO_ROOT / "data" / "canonical_judgment_rules.json"
DEFAULT_QUEUE_JSON = REPO_ROOT / "data" / "policy_likeness_queue.json"
GUARD = "knowledge_kind_policy"
QUESTION_ID = "is_policy_v2"
POLICY_MIN = 0.9
GUIDELINE_MIN = 0.55
TIERS = ("policy", "guideline", "insight")
HUMAN_LABEL_SOURCE = "human:judgment_review"

# experiments/policy_kind_jev/measure.py の QUESTION_V2 と同じ文面（変えたら再測定する）
QUESTION = {
    "type": "noul",
    "instructions": "`items[{n}]` は、リース会社の担当者が教えた一文です。これは、特定の物件・業種・取引先（の状態）・取引形態や契約条件について、当社としての取扱い基準（取り扱える/取り扱えない/不向き、取り扱う条件や前提、審査で重点的に見ること）を決めていますか。",
    "criteria": {
        "true": "対象が具体的（ある物件や設備、業種、借手の状態、契約条件など）で、その対象をどう扱うか（可否・条件・前提・重点）を決めている。",
        "false": "対象を特定しない一般論、審査の心構え、汎用の確認手順や確認項目の列挙、市場や業界の傾向・相場・事実・制度の説明、理由づけ、感想・雑談。",
    },
}
_SAME_RE = re.compile(r"[（(]同旨[:：]")


def tier_for(probability: float | None, *, rule_kind: str, decision: bool | None = None) -> str:
    """ユーザーの判断 > 決定的ルール > Jev の順で決める。Jev が無い時は知見（断定しない側）。"""
    if decision is not None:
        return "policy" if decision else "insight"
    if rule_kind == POLICY:
        return "policy"
    if probability is None:
        return "insight"
    if probability >= POLICY_MIN:
        return "policy"
    if probability >= GUIDELINE_MIN:
        return "guideline"
    return "insight"


def main_statement(statement: str) -> str:
    """統合で追記された「（同旨: …）」を除いた代表文。"""
    match = _SAME_RE.search(statement)
    return (statement[: match.start()] if match else statement).strip()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_queue(path: Path = DEFAULT_QUEUE_JSON) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        record_silent_failure("judgment.policy_likeness.load_queue", "fallback", exc, detail="キューを読めず空として扱う")
        return {}
    return data.get("items", {}) if isinstance(data, dict) else {}


def save_queue(items: dict[str, dict[str, Any]], path: Path = DEFAULT_QUEUE_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {"schema_version": 1, "thresholds": {"policy_min": POLICY_MIN, "guideline_min": GUIDELINE_MIN}, "items": items}
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def judge_with_jev(texts: list[str], *, batch: int = 15) -> list[float | None]:
    """伏せ字にして Jev に送る。送れない文（個人情報らしい）や失敗は None。"""
    import typesafe_dedup_guard as transport
    from api.chat_judgment_asset_capture import mask_for_jev

    os.environ.setdefault("TYPESAFE_DEDUP_TIMEOUT_SECONDS", "90")
    # 改善パイプライン（launchd com.tunelease.improvement-pipeline）の plist には鍵の受け渡しが無く、
    # 10/5 の初回から毎回 TYPESAFE_API_KEY is not configured で34件とも失敗していた（REV-496）。
    # 他の Jev 呼び出し（lease_intelligence_mind 等）と同じ Keychain 項目を既定にする。
    os.environ.setdefault("TYPESAFE_API_KEYCHAIN_SERVICE", "typesafe-api-key")
    masked = [mask_for_jev(text) for text in texts]
    sendable = [k for k, m in enumerate(masked) if m]
    results: list[float | None] = [None] * len(texts)
    for start in range(0, len(sendable), batch):
        chunk = sendable[start : start + batch]
        questions = {
            f"item{n}": {**QUESTION, "instructions": QUESTION["instructions"].format(n=n)} for n in range(len(chunk))
        }
        try:
            body = transport._default_request(
                {"state": {"items": [masked[k] for k in chunk]}, "model": "jev-latest", "questions": questions}
            )
            for n, k in enumerate(chunk):
                results[k] = float(transport._noul(body["answers"], f"item{n}"))
        except Exception as exc:  # noqa: BLE001 - Jev 不通でも他の段は止めない（その分は未判定のまま次回再試行）
            record_silent_failure("judgment.policy_likeness.jev", "fallback", exc, detail="Jev 判定できず未判定のまま")
    return results


def _log_shadow(rows: list[dict[str, Any]]) -> list[str]:
    import jev_judgment_log

    records = jev_judgment_log.build_records(
        guard=GUARD,
        run_id=jev_judgment_log.new_run_id(),
        mode="shadow",
        model="jev-latest",
        items=[
            {
                "question": QUESTION_ID,
                "subject": row["statement"],
                "probability": row["probability"],
                "choice": row["probability"] >= GUIDELINE_MIN,
                "route": row["tier"],
                "thresholds": {"policy_min": POLICY_MIN, "guideline_min": GUIDELINE_MIN},
                # 中間（目安）だけ人が見る。方針・知見の端は自動扱い
                "auto_passed": row["tier"] != "guideline",
            }
            for row in rows
        ],
    )
    jev_judgment_log.append_records(records)
    return [record["judgment_id"] for record in records]


def score_canonical_rules(
    *,
    canonical_path: Path = DEFAULT_CANONICAL_JSON,
    queue_path: Path = DEFAULT_QUEUE_JSON,
    judge: Callable[[list[str]], list[float | None]] = judge_with_jev,
    now: dt.datetime | None = None,
) -> dict[str, int]:
    """active の判断資産のうち未判定・本文が変わったものだけ Jev にかけてキューを更新する。"""
    rules = json.loads(canonical_path.read_text(encoding="utf-8")).get("rules") or []
    queue = load_queue(queue_path)
    targets = []
    for rule in rules:
        if not isinstance(rule, dict) or rule.get("status") != "active":
            continue
        text = main_statement(str(rule.get("canonical_statement") or ""))
        rule_id = str(rule.get("id") or "")
        if not rule_id or not text:
            continue
        current = queue.get(rule_id)
        if current and current.get("text_hash") == _text_hash(text) and current.get("probability") is not None:
            continue
        targets.append((rule_id, text))
    probabilities = judge([text for _, text in targets]) if targets else []
    stamp = (now or dt.datetime.now()).isoformat(timespec="seconds")
    scored = []
    for (rule_id, text), probability in zip(targets, probabilities):
        if probability is None:
            continue
        previous = queue.get(rule_id) or {}
        # 本文が変わったらユーザーの判断はその文に対するものではなくなるので持ち越さない
        decision = previous.get("decision") if previous.get("text_hash") == _text_hash(text) else None
        rule_kind = classify_knowledge_kind(text)
        row = {
            "id": rule_id,
            "statement": text,
            "text_hash": _text_hash(text),
            "rule_kind": rule_kind,
            "probability": round(probability, 3),
            "tier": tier_for(probability, rule_kind=rule_kind, decision=decision),
            "decision": decision,
            "decided_at": previous.get("decided_at") if decision is not None else None,
            "scored_at": stamp,
        }
        queue[rule_id] = row
        scored.append(row)
    for row, judgment_id in zip(scored, _log_shadow(scored) if scored else []):
        row["judgment_id"] = judgment_id
    active_ids = {str(rule.get("id")) for rule in rules if isinstance(rule, dict) and rule.get("status") == "active"}
    for rule_id in list(queue):
        if rule_id not in active_ids:
            queue.pop(rule_id)  # 統合・降格されたものは目安候補から外す
    save_queue(queue, queue_path)
    return {"targets": len(targets), "scored": len(scored), "failed": len(targets) - len(scored), "queued": len(queue)}


def review_candidates(queue_path: Path = DEFAULT_QUEUE_JSON, *, limit: int = 30) -> dict[str, Any]:
    """「社内の目安」でまだユーザーが決めていないもの（方針らしい順）。"""
    items = [row for row in load_queue(queue_path).values() if row.get("tier") == "guideline" and row.get("decision") is None]
    items.sort(key=lambda row: -float(row.get("probability") or 0))
    return {"total_count": len(items), "candidates": items[: max(1, min(int(limit or 30), 100))]}


def record_decision(rule_id: str, is_policy: bool, queue_path: Path = DEFAULT_QUEUE_JSON) -> dict[str, Any]:
    """ユーザーの1タップを記録する。shadow 中は回答（knowledge_kind）を変えず、ラベルとして貯める。"""
    queue = load_queue(queue_path)
    row = queue.get(rule_id)
    if row is None:
        raise KeyError(rule_id)
    row["decision"] = bool(is_policy)
    row["decided_at"] = dt.datetime.now().isoformat(timespec="seconds")
    row["tier"] = tier_for(row.get("probability"), rule_kind=str(row.get("rule_kind") or ""), decision=row["decision"])
    save_queue(queue, queue_path)
    if row.get("judgment_id"):
        import jev_judgment_log

        jev_judgment_log.append_label(row["judgment_id"], bool(is_policy), label_source=HUMAN_LABEL_SOURCE)
    return row
