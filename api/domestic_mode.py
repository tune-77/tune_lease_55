from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from api.loop_engineering_common import call_gemini_json

_DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent / "data")))
_PROPOSALS_PATH = _DATA_DIR / "domestic_mode_proposals.jsonl"
_IMPROVEMENT_LOG_PATH = _DATA_DIR / "cloudrun_improvement_log.jsonl"
_ALLOWED_STATUSES = {"adopt", "hold", "reject", "observe"}
_STATUS_LABELS = {
    "adopt": "採用候補",
    "hold": "保留",
    "reject": "却下候補",
    "observe": "観測継続",
}
_APPROVED_STATUSES = {"approved", "applied", "adopt", "adopted", "promoted"}
_REJECTED_STATUSES = {"rejected", "reject", "deleted", "suppressed"}
_DEFERRED_STATUSES = {"deferred", "parked", "hold", "held", "observe", "needs_human_review"}


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _canonical_key(title: str, body: str = "") -> str:
    digest = hashlib.sha1(f"{title}\n{body}".encode("utf-8")).hexdigest()[:16]
    return f"proposal:{digest}"


def _memo_field(memo: str, field: str) -> str:
    for line in str(memo or "").splitlines():
        if line.startswith(f"{field}:"):
            return line.split(":", 1)[1].strip()
    return ""


def _selection_status(status: str) -> str:
    value = str(status or "").strip().lower()
    if value in _APPROVED_STATUSES:
        return "selected"
    if value in _REJECTED_STATUSES:
        return "selected_out"
    if value in _DEFERRED_STATUSES:
        return "pending_selection"
    return "unselected"


def build_genetic_profile(proposal: dict[str, Any], status: str = "") -> dict[str, Any]:
    """Return lightweight evolutionary metadata for domestic-mode proposals.

    This is deliberately heuristic. It does not auto-approve proposals; it only
    tells the next self-proposal prompt how much to preserve or mutate a lineage.
    """
    effective_status = status or str(
        proposal.get("human_decision_status") or proposal.get("domestic_status") or proposal.get("status") or ""
    )
    selection_status = _selection_status(effective_status)
    style = str(proposal.get("proposal_style") or "").strip()
    has_metric = bool(str(proposal.get("success_metric") or "").strip())
    has_risk = bool(str(proposal.get("risk") or "").strip())
    has_evidence = bool(str(proposal.get("evidence") or "").strip())

    fitness = 50
    if selection_status == "selected":
        fitness += 25
    elif selection_status == "selected_out":
        fitness -= 22
    elif selection_status == "pending_selection":
        fitness += 2
    if has_metric:
        fitness += 10
    if has_evidence:
        fitness += 8
    if has_risk:
        fitness += 4
    if style == "leap":
        fitness -= 4
    fitness = max(0, min(100, fitness))

    mutation_rate = 0.1
    if style == "leap":
        mutation_rate = 0.24
    if selection_status == "selected":
        mutation_rate = 0.06
    elif selection_status == "selected_out":
        mutation_rate = 0.22
    elif selection_status == "pending_selection":
        mutation_rate = 0.14
    if not has_metric:
        mutation_rate += 0.04
    mutation_rate = round(max(0.04, min(0.35, mutation_rate)), 2)

    if selection_status == "selected":
        inheritance_policy = "preserve_and_refine"
    elif selection_status == "selected_out":
        inheritance_policy = "suppress_or_mutate"
    elif mutation_rate >= 0.2:
        inheritance_policy = "explore_variant"
    else:
        inheritance_policy = "observe_then_select"

    return {
        "fitness_score": fitness,
        "mutation_rate": mutation_rate,
        "selection_status": selection_status,
        "inheritance_policy": inheritance_policy,
        "parent_ids": [str(proposal.get("source_canonical_key") or "").strip()] if proposal.get("source_canonical_key") else [],
        "derivation_reason": str(proposal.get("decision_layer") or proposal.get("hypothesis") or "").strip()[:160],
        "validation_status": selection_status,
    }


def build_domestic_mode_prompt(memo: str, heuristic: dict[str, Any] | None = None) -> str:
    heuristic_lines = "\n".join(f"- {key}: {value}" for key, value in (heuristic or {}).items() if value)
    if not heuristic_lines:
        heuristic_lines = "（画面内の一次判定なし）"
    return f"""あなたはリース審査AIシステム「紫苑」の内政モードです。
外向きの審査回答ではなく、システム内部の導線・情報配置・記憶運用・改善候補を見直します。

目的:
- 人間の内政メモを、採用 / 保留 / 却下 / 観測継続 のどれで扱うべきか再判定する
- 低利用だけで削除せず、価値不足 / 発見性不足 / 文脈不足 / 削除候補 / 再配置候補 / 観測継続 を切り分ける
- 人間が次に動ける粒度で、理由と次の扱いを書く

判断ルール:
- 「触らない線」や禁止条件がある場合は、原則として保留または範囲限定にする
- 意図、使う場面、残す理由、成功指標が揃う場合は採用候補にできる
- 削ってよい条件だけが強く、残す理由や使用場面が弱い場合は却下候補にできる
- 根拠が足りないが測定条件がある場合は観測継続にする
- 作業実行そのものは約束しない。人間確認を前提にする

【画面内の一次判定】
{heuristic_lines}

【内政メモ】
{memo.strip() or "（未入力）"}

以下のJSONオブジェクトだけを返してください。
{{
  "status": "adopt|hold|reject|observe",
  "label": "採用候補|保留|却下候補|観測継続",
  "reason": "判定理由を120字以内で具体的に書く",
  "next_action": "次に人間または紫苑がする扱いを120字以内で書く",
  "decision_layer": "価値不足|発見性不足|文脈不足|削除候補|再配置候補|観測継続 のうち近いもの",
  "missing_inputs": ["不足している入力項目"],
  "success_metric": "効果を見る指標。未指定なら提案する",
  "review_timing": "再判定時期。未指定なら30日後などを提案する"
}}"""


def normalize_domestic_mode_result(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Gemini応答がJSONオブジェクトではありません")

    status = str(raw.get("status") or "").strip()
    if status not in _ALLOWED_STATUSES:
        status = "hold"

    label = str(raw.get("label") or "").strip() or _STATUS_LABELS[status]
    missing_inputs = raw.get("missing_inputs")
    if not isinstance(missing_inputs, list):
        missing_inputs = []

    return {
        "generated": True,
        "status": status,
        "label": label,
        "reason": str(raw.get("reason") or "").strip()[:240],
        "next_action": str(raw.get("next_action") or "").strip()[:240],
        "decision_layer": str(raw.get("decision_layer") or "").strip()[:80],
        "missing_inputs": [str(item).strip()[:80] for item in missing_inputs if str(item).strip()][:8],
        "success_metric": str(raw.get("success_metric") or "").strip()[:160],
        "review_timing": str(raw.get("review_timing") or "").strip()[:120],
        "model": "gemini",
    }


def evaluate_domestic_mode_memo(memo: str, heuristic: dict[str, Any] | None = None) -> dict[str, Any]:
    if not str(memo or "").strip():
        return {
            "generated": False,
            "status": "hold",
            "label": "未入力",
            "reason": "内政メモが未入力です。",
            "next_action": "対象、意図、成功指標を最低限入れてください。",
            "decision_layer": "文脈不足",
            "missing_inputs": ["対象", "意図", "成功指標"],
            "success_metric": "",
            "review_timing": "",
            "model": "rule",
        }

    prompt = build_domestic_mode_prompt(memo, heuristic)
    raw = call_gemini_json(prompt, temperature=0.25, max_output_tokens=2048)
    return normalize_domestic_mode_result(raw)


def _build_proposal_entry(memo: str, heuristic: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
    target = _memo_field(memo, "対象") or "内政モード"
    intent = _memo_field(memo, "意図")
    do_not_touch = _memo_field(memo, "触らない線")
    delete_condition = _memo_field(memo, "削ってよい条件")
    generated_at = dt.datetime.now().isoformat(timespec="seconds")
    event_id = str(uuid.uuid4())
    label = str(verdict.get("label") or _STATUS_LABELS.get(str(verdict.get("status") or ""), "保留"))
    title = f"内政再判定: {target}を{label}で扱う"
    reason = str(verdict.get("reason") or "").strip()
    next_action = str(verdict.get("next_action") or "").strip()
    success_metric = str(verdict.get("success_metric") or "").strip()
    review_timing = str(verdict.get("review_timing") or "").strip() or "30日後"
    risk_parts = []
    if do_not_touch:
        risk_parts.append(f"触らない線: {do_not_touch}")
    if delete_condition:
        risk_parts.append(f"削ってよい条件: {delete_condition}")
    missing = verdict.get("missing_inputs") if isinstance(verdict.get("missing_inputs"), list) else []
    if missing:
        risk_parts.append("不足入力: " + " / ".join(str(item) for item in missing))
    risk = " / ".join(risk_parts)
    body = "\n\n".join(
        part
        for part in (
            f"## 仮説\n{reason}" if reason else "",
            f"## 根拠\n内政メモ、画面内一次判定、Gemini再判定を突き合わせた。",
            f"## 変更案\n{next_action}" if next_action else "",
            f"## 成功指標\n{success_metric}" if success_metric else "",
            f"## 検証方法\n{review_timing}に採用/保留/却下後の効果を確認する。",
            f"## リスク\n{risk}" if risk else "",
        )
        if part
    )
    priority = {
        "adopt": "high",
        "observe": "medium",
        "hold": "medium",
        "reject": "low",
    }.get(str(verdict.get("status") or ""), "medium")
    entry = {
        "event_id": event_id,
        "generated_at": generated_at,
        "ts": generated_at,
        "title": title,
        "target": target,
        "target_page": "",
        "source": "domestic_mode",
        "kind": "内政モード",
        "status": "needs_human_review",
        "priority": priority,
        "hypothesis": reason or intent,
        "evidence": "内政メモ + 画面内一次判定 + Gemini再判定",
        "proposed_change": next_action,
        "success_metric": success_metric,
        "verification_plan": f"{review_timing}に採用/保留/却下後の効果を確認する。",
        "risk": risk,
        "proposal_schema": "shion_self_hypothesis_v1",
        "proposal_style": str(verdict.get("proposal_style") or "").strip(),
        "human_decision_status": "needs_human_review",
        "domestic_status": str(verdict.get("status") or ""),
        "domestic_label": label,
        "decision_layer": str(verdict.get("decision_layer") or ""),
        "missing_inputs": missing,
        "review_timing": review_timing,
        "memo": str(memo or "").strip()[:4000],
        "heuristic": heuristic,
        "model": str(verdict.get("model") or "gemini"),
        "canonical_key": _canonical_key(title, body),
        "body": body,
    }
    entry["genetic_profile"] = build_genetic_profile(entry, str(verdict.get("status") or ""))
    return entry


def save_domestic_mode_proposal(memo: str, heuristic: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
    entry = _build_proposal_entry(memo, heuristic, verdict)
    _append_jsonl(_PROPOSALS_PATH, entry)
    _append_jsonl(
        _IMPROVEMENT_LOG_PATH,
        {
            **entry,
            "surface": "shion_self_proposal",
            "proposed_by": "shion",
            "source": "domestic_mode",
            "event_type": "domestic_mode_self_proposal",
            "payload": {
                "canonical_key": entry["canonical_key"],
                "domestic_status": entry["domestic_status"],
                "decision_layer": entry["decision_layer"],
                "success_metric": entry["success_metric"],
                "review_timing": entry["review_timing"],
                "genetic_profile": entry["genetic_profile"],
            },
        },
    )
    return {
        "saved": True,
        "event_id": entry["event_id"],
        "title": entry["title"],
        "canonical_key": entry["canonical_key"],
        "proposal_path": str(_PROPOSALS_PATH),
        "improvement_log_path": str(_IMPROVEMENT_LOG_PATH),
    }


def _existing_domestic_keys() -> set[str]:
    keys: set[str] = set()
    for row in _load_jsonl(_PROPOSALS_PATH, limit=500):
        key = str(row.get("canonical_key") or row.get("source_canonical_key") or "").strip()
        if key:
            keys.add(key)
    return keys


def connect_self_proposal_to_domestic_mode(
    proposal: dict[str, Any],
    *,
    source: str,
    kind: str,
) -> dict[str, Any]:
    """Record an ordinary self proposal as domestic-mode intake without a button."""
    title = str(proposal.get("title") or proposal.get("topic") or "").strip()
    if not title:
        return {"saved": False, "reason": "title is empty"}

    body = "\n".join(
        str(proposal.get(key) or "")
        for key in (
            "hypothesis",
            "evidence",
            "proposed_change",
            "suggestion",
            "reason",
            "review_point",
            "observation",
            "search_hint",
        )
        if str(proposal.get(key) or "").strip()
    )
    source_key = str(proposal.get("canonical_key") or "").strip() or _canonical_key(title, body)
    domestic_key = f"domestic:{source_key}"
    if domestic_key in _existing_domestic_keys():
        return {"saved": False, "reason": "already connected", "canonical_key": domestic_key}

    target = str(proposal.get("target_page") or proposal.get("target") or proposal.get("topic") or title).strip()
    intent = str(
        proposal.get("hypothesis")
        or proposal.get("proposed_change")
        or proposal.get("suggestion")
        or proposal.get("review_point")
        or proposal.get("reason")
        or ""
    ).strip()
    success_metric = str(proposal.get("success_metric") or "").strip()
    review_timing = str(proposal.get("review_timing") or "").strip() or "30日後"
    risk = str(proposal.get("risk") or "").strip()
    generated_at = str(proposal.get("generated_at") or dt.datetime.now().isoformat(timespec="seconds"))
    event_id = str(uuid.uuid4())
    entry_title = f"内政接続: {title}"
    memo = "\n".join(
        [
            "内政メモ:",
            f"対象: {target}",
            f"意図: {intent}",
            "使う場面: 自己提案モードから自動接続",
            "残す理由: 紫苑が出した改善仮説を内政基準で再利用するため",
            "削ってよい条件: 人間が却下した場合、同系統の浅い提案を抑制する",
            f"触らない線: {risk}",
            f"成功指標: {success_metric}",
            f"再判定時期: {review_timing}",
        ]
    )
    entry = {
        "event_id": event_id,
        "generated_at": generated_at,
        "ts": generated_at,
        "title": entry_title,
        "target": target,
        "target_page": str(proposal.get("target_page") or ""),
        "source": "domestic_mode_auto_connect",
        "source_proposal_source": source,
        "source_canonical_key": source_key,
        "kind": "内政モード",
        "source_kind": kind,
        "status": "needs_human_review",
        "priority": str(proposal.get("priority") or "medium").strip(),
        "hypothesis": intent,
        "evidence": str(proposal.get("evidence") or proposal.get("reason") or "自己提案モードから自動接続").strip(),
        "proposed_change": str(proposal.get("proposed_change") or proposal.get("suggestion") or proposal.get("review_point") or "").strip(),
        "success_metric": success_metric,
        "verification_plan": str(proposal.get("verification_plan") or f"{review_timing}に採用/保留/却下後の効果を確認する。").strip(),
        "risk": risk,
        "proposal_schema": "shion_self_hypothesis_v1",
        "human_decision_status": "needs_human_review",
        "domestic_status": "observe",
        "domestic_label": "観測継続",
        "decision_layer": "観測継続",
        "missing_inputs": [],
        "review_timing": review_timing,
        "memo": memo[:4000],
        "heuristic": {"source": source, "kind": kind, "auto_connected": True},
        "model": "self_proposal_auto_connect",
        "canonical_key": domestic_key,
        "body": body,
    }
    entry["genetic_profile"] = build_genetic_profile(entry, "observe")
    _append_jsonl(_PROPOSALS_PATH, entry)
    return {
        "saved": True,
        "event_id": event_id,
        "title": entry_title,
        "canonical_key": domestic_key,
        "source_canonical_key": source_key,
    }


def _load_jsonl(path: Path, limit: int = 80) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def build_domestic_memory_context(limit: int = 6) -> str:
    """Summarize domestic-mode judgment memory for self-proposal prompts."""
    proposal_rows = _load_jsonl(_PROPOSALS_PATH)
    ledger_rows = _load_jsonl(_IMPROVEMENT_LOG_PATH)
    latest_status_by_key: dict[str, str] = {}
    for row in ledger_rows:
        key = str(row.get("canonical_key") or row.get("key") or "").strip()
        status = str(row.get("status") or row.get("action") or "").strip()
        if key and status:
            latest_status_by_key[key] = status

    selected: list[str] = []
    for row in reversed(proposal_rows):
        if len(selected) >= limit:
            break
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        key = str(row.get("canonical_key") or "").strip()
        status = latest_status_by_key.get(key) or str(row.get("human_decision_status") or row.get("status") or "")
        layer = str(row.get("decision_layer") or "").strip()
        metric = str(row.get("success_metric") or "").strip()
        risk = str(row.get("risk") or "").strip()
        genetic_profile = row.get("genetic_profile") if isinstance(row.get("genetic_profile"), dict) else {}
        if not genetic_profile:
            genetic_profile = build_genetic_profile(row, status)
        line = f"- {title}: status={status or '-'}"
        line += (
            f" / fitness={genetic_profile.get('fitness_score', '-')}"
            f" / mutation_rate={genetic_profile.get('mutation_rate', '-')}"
            f" / inheritance={genetic_profile.get('inheritance_policy', '-')}"
        )
        if layer:
            line += f" / layer={layer}"
        if metric:
            line += f" / metric={metric[:80]}"
        if risk:
            line += f" / guardrail={risk[:80]}"
        selected.append(line)

    if not selected:
        return ""

    return """【内政モード判断メモリ】
以下はUserと紫苑が過去に内政モードで判断した改善メモリです。
- approved は似た判断軸を優先する
- rejected は同じ浅い提案を繰り返さない
- deferred/parked/observe は追加根拠か成功指標を添えて提案する
- guardrail/触らない線がある場合は必ず守る
- fitness が高い系統は小さく継承・改善し、mutation_rate が高い系統は別方向の検証可能な変異案として出す
- 跳躍提案は最大1件まで。突然変異は面白さではなく、測れる違和感として出す
""" + "\n".join(selected)
