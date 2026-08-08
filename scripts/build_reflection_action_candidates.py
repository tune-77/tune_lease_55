#!/usr/bin/env python3
"""Convert Shion private reflection into reviewable operational action candidates.

This script is deliberately conservative:
- It writes candidates only to data/reflection_action_candidates.jsonl.
- It does not promote judgment assets, alter prompts, scoring, or deploy state.
- User selection is tracked later through the existing improvement triage flow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DELTA = REPO_ROOT / "data" / "shion_reflection_delta.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "reflection_action_candidates.jsonl"
DEFAULT_REPORT = REPO_ROOT / "reports" / "reflection_action_candidates_latest.md"

ACTION_SCHEMA = "shion_reflection_action_candidate_v1"
PROPOSAL_SCHEMA = "shion_self_hypothesis_v1"
ACTION_TERMS = (
    "確認",
    "次回",
    "変える",
    "改善",
    "判断資産",
    "審査",
    "稟議",
    "過去案件",
    "情報提供",
    "レビュー",
)
PRACTICAL_TERMS = (
    "リース",
    "審査",
    "稟議",
    "条件付き承認",
    "否決",
    "過去案件",
    "判断資産",
    "確認項目",
    "情報提供",
    "ユーザー",
    "User",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _clean_text(value: Any, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" -")
    return text[:limit].rstrip()


def _canonical_key(title: str, body: str) -> str:
    digest = hashlib.sha1(f"{title}\n{body}".encode("utf-8")).hexdigest()[:16]
    return f"reflection_action:{digest}"


def _source_ref(delta: dict[str, Any], name: str) -> str:
    sources = delta.get("sources") if isinstance(delta.get("sources"), dict) else {}
    source = sources.get(name) if isinstance(sources.get(name), dict) else {}
    return str(source.get("path") or source.get("status") or "")


def _items_from_delta(delta: dict[str, Any]) -> list[dict[str, str]]:
    handoff = delta.get("operational_handoff") if isinstance(delta.get("operational_handoff"), dict) else {}
    judgment = delta.get("judgment_change_log") if isinstance(delta.get("judgment_change_log"), dict) else {}
    quality = delta.get("quality") if isinstance(delta.get("quality"), dict) else {}
    metrics = delta.get("metrics") if isinstance(delta.get("metrics"), dict) else {}

    raw_items: list[dict[str, str]] = []
    for action in handoff.get("shion_next_actions") or []:
        text = _clean_text(action)
        if text:
            raw_items.append({"kind": "審査改善", "text": text, "origin": "operational_handoff.shion_next_actions"})
    for request in handoff.get("user_requests") or []:
        text = _clean_text(request)
        if text:
            raw_items.append({"kind": "ユーザー情報提供", "text": text, "origin": "operational_handoff.user_requests"})

    next_check = _clean_text(judgment.get("次回から変える確認事項"))
    if next_check:
        raw_items.append({"kind": "審査改善", "text": next_check, "origin": "judgment_change_log.next_check"})
    asset = _clean_text(judgment.get("判断資産候補"))
    if asset:
        raw_items.append({"kind": "過去案件からの学び", "text": asset, "origin": "judgment_change_log.judgment_asset_candidate"})
    missed = _clean_text(judgment.get("紫苑が外した点"))
    if missed:
        raw_items.append({"kind": "審査改善", "text": f"次回はこの見落としを先に確認する: {missed}", "origin": "judgment_change_log.missed_point"})

    filtered: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_items:
        text = _clean_text(item.get("text"))
        if len(text) < 10 or not any(term in text for term in ACTION_TERMS):
            continue
        signature = re.sub(r"\s+", "", text)[:160]
        if signature in seen:
            continue
        seen.add(signature)
        item["text"] = text
        item["quality_status"] = str(quality.get("status") or "")
        item["reflection_similarity_to_yesterday"] = str(metrics.get("reflection_similarity_to_yesterday") or "")
        filtered.append(item)
    return filtered[:5]


def _operationalize_text(text: str, kind: str) -> str:
    """Turn abstract reflection wording into a concrete reviewable action."""
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    if any(term in cleaned for term in PRACTICAL_TERMS):
        return cleaned
    if "退屈" in cleaned or "観測レポート" in cleaned or "小さく変える" in cleaned:
        return (
            "内省が抽象評価で止まった日は、当日ログから"
            "リース審査の確認項目・Userへの情報提供・過去案件からの学びのどれか1つへ変換して記録する。"
        )
    if kind == "ユーザー情報提供":
        return f"Userへの次回情報提供として、{cleaned}"
    if kind == "過去案件からの学び":
        return f"過去案件からの学びとして、{cleaned}"
    return f"リース審査業務の改善候補として、{cleaned}"


def build_candidates(delta: dict[str, Any], *, generated_at: str | None = None) -> list[dict[str, Any]]:
    target_date = str(delta.get("target_date") or date.today().isoformat())
    generated = generated_at or datetime.now().isoformat(timespec="seconds")
    today_reflection = _source_ref(delta, "today_reflection")
    today_memory = _source_ref(delta, "today_memory")
    candidates: list[dict[str, Any]] = []
    seen_actions: set[str] = set()

    for idx, item in enumerate(_items_from_delta(delta), start=1):
        kind = item["kind"]
        text = _operationalize_text(item["text"], kind)
        if not text:
            continue
        signature = re.sub(r"\s+", "", text)
        if signature in seen_actions:
            continue
        seen_actions.add(signature)
        title = f"内省アクション: {text[:42]}"
        success_metric = "採用(修正)/保留/却下の判断率と、採用後30日以内の再利用・thin評価低下・追加質問発生率"
        verification_plan = "既存の改善トリアージで採用(修正)/保留/却下を記録し、30日後にsuccess_metricの変化を見る。"
        risk = "内省が的外れなら回答が冗長化するため、User判断なしにプロンプト・RAG・判断資産へ昇格しない。"
        hypothesis = f"内省から出た「{text}」を実務アクション候補として扱うと、自己言及で終わらず次の審査・対話に戻せる。"
        proposed_change = text
        body = "\n".join([hypothesis, proposed_change, success_metric, verification_plan, risk])
        canonical_key = _canonical_key(title, body)
        candidates.append(
            {
                "schema_version": 1,
                "proposal_schema": PROPOSAL_SCHEMA,
                "action_schema": ACTION_SCHEMA,
                "event_id": f"{target_date}-reflection-action-{idx}",
                "generated_at": generated,
                "ts": generated,
                "title": title,
                "target": "紫苑の内省運用",
                "target_page": "/lease-intelligence",
                "source": "reflection_action_candidates",
                "kind": kind,
                "status": "needs_human_review",
                "priority": "medium",
                "hypothesis": hypothesis,
                "evidence": f"Private Reflection delta由来: {item.get('origin')} / target_date={target_date}",
                "proposed_change": proposed_change,
                "success_metric": success_metric,
                "verification_plan": verification_plan,
                "risk": risk,
                "human_decision_status": "needs_human_review",
                "proposal_style": "reflection_to_action",
                "review_policy": "human_decision_required",
                "effect_tracking": "採用(修正)/保留/却下と、success_metric の事後変化で確認する",
                "source_paths": [path for path in (today_reflection, today_memory) if path],
                "target_date": target_date,
                "origin": item.get("origin"),
                "canonical_key": canonical_key,
                "body": body,
            }
        )
    return candidates


def _existing_keys(path: Path) -> set[str]:
    return {str(row.get("canonical_key") or "") for row in _load_jsonl(path) if row.get("canonical_key")}


def write_report(candidates: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Reflection Action Candidates",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- count: {len(candidates)}",
        "- policy: human_decision_required; no prompt/RAG/scoring promotion",
        "",
    ]
    if candidates:
        lines.append("## Candidates")
        for item in candidates:
            lines.append(
                f"- {item['title']} / status={item['human_decision_status']} / metric={item['success_metric']}"
            )
    else:
        lines.append("候補なし")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, default=DEFAULT_DELTA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    delta = _load_json(args.delta)
    candidates = build_candidates(delta)
    existing = _existing_keys(args.output)
    new_candidates = [item for item in candidates if item["canonical_key"] not in existing]
    _append_jsonl(args.output, new_candidates)
    write_report(candidates, args.report)
    print(f"reflection_action_candidates={len(candidates)} new={len(new_candidates)}")
    print(args.output)
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
