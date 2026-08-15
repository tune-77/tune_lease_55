#!/usr/bin/env python3
"""Convert experience replay failures into response checklist candidates.

The output is a review artifact, not an active prompt rule. It summarizes which
practical concepts were missing from historical answers and proposes compact
checklist items that can later be reviewed before promotion.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORICAL_REPORT = REPO_ROOT / "reports" / "experience_replay_historical_quality_latest.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports" / "experience_replay_checklist_candidates_latest.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "experience_replay_checklist_candidates_latest.md"


def load_report(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def build_checklist_candidates(report: dict[str, Any]) -> dict[str, Any]:
    cases = [
        case
        for case in ((report.get("final") or {}).get("cases") or [])
        if isinstance(case, dict) and not case.get("passed")
    ]
    candidates: list[dict[str, Any]] = []
    missing_counter: Counter[str] = Counter()
    uncertainty_miss_count = 0
    topic_counter: Counter[str] = Counter()

    for case in cases:
        missing_groups = missing_concept_groups(case)
        for group in missing_groups:
            missing_counter[group] += 1
        uncertainty_missing = bool(case.get("uncertainty_required")) and not bool(case.get("uncertainty_present"))
        if uncertainty_missing:
            uncertainty_miss_count += 1
        topic = infer_topic(str(case.get("query") or ""))
        topic_counter[topic] += 1
        checklist_items = propose_case_checklist_items(case, missing_groups, uncertainty_missing)
        if not checklist_items:
            continue
        candidates.append(
            {
                "id": f"exp_replay_check_{len(candidates) + 1:02d}",
                "status": "candidate_needs_human_review",
                "source_case_id": str(case.get("id") or ""),
                "query": str(case.get("query") or ""),
                "topic": topic,
                "score": case.get("score"),
                "missing_concepts": missing_groups,
                "uncertainty_missing": uncertainty_missing,
                "checklist_items": checklist_items,
                "priority": priority_for(case, missing_groups, uncertainty_missing),
            }
        )

    global_items = build_global_items(missing_counter, uncertainty_miss_count)
    candidates = sorted(candidates, key=lambda item: (item["priority"], len(item["missing_concepts"])), reverse=True)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "experience_replay_response_checklist_candidates",
        "guardrail": "review artifact only; no active prompt injection; no automatic memory or skill promotion",
        "source_report": report.get("eval_set") or "",
        "summary": {
            "failed_cases": len(cases),
            "candidate_count": len(candidates),
            "uncertainty_miss_count": uncertainty_miss_count,
            "top_missing_concepts": [
                {"concept_group": concept, "count": count}
                for concept, count in missing_counter.most_common(10)
            ],
            "by_topic": dict(sorted(topic_counter.items())),
        },
        "global_checklist_candidates": global_items,
        "case_checklist_candidates": candidates,
    }


def missing_concept_groups(case: dict[str, Any]) -> list[str]:
    groups: list[str] = []
    for result in case.get("concept_results") or []:
        if not isinstance(result, dict) or result.get("matched"):
            continue
        aliases = [str(alias) for alias in result.get("aliases") or [] if str(alias)]
        if aliases:
            groups.append(" / ".join(aliases))
    return groups


def propose_case_checklist_items(
    case: dict[str, Any],
    missing_groups: list[str],
    uncertainty_missing: bool,
) -> list[str]:
    query = str(case.get("query") or "")
    topic = infer_topic(query)
    items: list[str] = []
    if missing_groups:
        items.append(f"{topic}では、抜けやすい確認軸（{'; '.join(missing_groups[:3])}）を最低1つずつ入れる。")
    if uncertainty_missing:
        items.append("制度・可否・残価・事故/法令が絡む時は、断定せず「要確認」または「場合がある」を明示する。")
    topic_specific = topic_specific_item(topic)
    if topic_specific:
        items.append(topic_specific)
    return dedupe(items)[:4]


def build_global_items(missing_counter: Counter[str], uncertainty_miss_count: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for concept, count in missing_counter.most_common(5):
        items.append(
            {
                "id": f"global_missing_{len(items) + 1:02d}",
                "status": "candidate_needs_human_review",
                "trigger": "lease_practical_answer",
                "checklist_item": f"回答前に `{concept}` が今回の判断に関係するか確認し、関係するなら1文で触れる。",
                "evidence_count": count,
            }
        )
    if uncertainty_miss_count:
        items.append(
            {
                "id": "global_uncertainty_01",
                "status": "candidate_needs_human_review",
                "trigger": "制度・可否・残価・法令・特殊物件",
                "checklist_item": "可否や制度を断定する前に、対象条件・契約形態・最新ルールの確認余地を明示する。",
                "evidence_count": uncertainty_miss_count,
            }
        )
    return items


def infer_topic(query: str) -> str:
    if "飲食" in query:
        return "飲食業リース"
    if "事業計画" in query:
        return "事業計画"
    if "メイン先" in query or "メインバンク" in query:
        return "メイン先・銀行支援"
    if "電車" in query or "鉄道" in query:
        return "鉄道車両"
    if "漁業" in query:
        return "漁業"
    if "焼却炉" in query:
        return "焼却炉"
    if "空調" in query or "ダクト" in query or "工事費" in query:
        return "工事費・建物付属設備"
    if "車検" in query or "レンタカー" in query or "くるま" in query:
        return "自動車周辺リスク"
    return "リース実務一般"


def topic_specific_item(topic: str) -> str:
    mapping = {
        "飲食業リース": "飲食業は初期投資メリットだけで終えず、撤退時の物件回収・中古市場・原状回復も見る。",
        "事業計画": "事業計画は項目列挙で終えず、売上/費用の前提と根拠資料まで確認する。",
        "メイン先・銀行支援": "メイン先でも、銀行支援の有無と返済原資を分けて確認する。",
        "鉄道車両": "大型車両は調達メリットだけでなく、長期契約・保守・資金調達/返済原資をセットで見る。",
        "漁業": "漁業は対象物件と季節性に加え、保険・担保・保全で荒天/漁獲変動を受け止められるか見る。",
        "焼却炉": "焼却炉はリース可否だけでなく、許認可、設置場所、撤去費、保守、処分、残価を確認する。",
        "工事費・建物付属設備": "工事費やダクト付き設備は、物件本体と付帯工事、所有権、撤去/原状回復を分ける。",
        "自動車周辺リスク": "自動車系は車検・登録・所有者/使用者・保険/整備/事故対応を分けて確認する。",
    }
    return mapping.get(topic, "")


def priority_for(case: dict[str, Any], missing_groups: list[str], uncertainty_missing: bool) -> int:
    score = float(case.get("score") or 0)
    base = 100 - min(100, int(score))
    return base + len(missing_groups) * 10 + (15 if uncertainty_missing else 0)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# Experience Replay Checklist Candidates",
        "",
        f"- Generated at: `{payload.get('generated_at')}`",
        f"- Mode: `{payload.get('mode')}`",
        f"- Guardrail: {payload.get('guardrail')}",
        f"- Failed cases: {summary.get('failed_cases', 0)}",
        f"- Candidates: {summary.get('candidate_count', 0)}",
        f"- Uncertainty misses: {summary.get('uncertainty_miss_count', 0)}",
        "",
        "## Global Candidates",
    ]
    global_items = payload.get("global_checklist_candidates") or []
    if not global_items:
        lines.append("- None")
    for item in global_items:
        lines.append(f"- `{item['id']}` {item['checklist_item']} (evidence={item['evidence_count']})")
    lines.extend(["", "## Case Candidates"])
    case_items = payload.get("case_checklist_candidates") or []
    if not case_items:
        lines.append("- None")
    for item in case_items[:20]:
        lines.append(f"- `{item['id']}` priority={item['priority']} topic={item['topic']} / {item['query']}")
        for checklist in item.get("checklist_items") or []:
            lines.append(f"  - {checklist}")
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-report", type=Path, default=DEFAULT_HISTORICAL_REPORT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    payload = build_checklist_candidates(load_report(args.historical_report.expanduser()))
    write_outputs(payload, output_json=args.output_json.expanduser(), output_md=args.output_md.expanduser())
    print(
        "experience replay checklist candidates: "
        f"{payload['summary']['candidate_count']} -> {args.output_md.expanduser()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
