#!/usr/bin/env python3
"""Build an answer-quality eval set from experience flywheel replay candidates.

This converts real conversation traces into a small benchmark compatible with
scripts/evaluate_answer_quality.py. It is deterministic and conservative:
non-lease casual queries are skipped, generated cases remain in a separate eval
set, and no existing benchmark is modified.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLYWHEEL_REPORT = REPO_ROOT / "reports" / "experience_flywheel_latest.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "api" / "knowledge" / "experience_replay_eval_set.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "experience_replay_eval_set_latest.md"

LEASE_RELEVANCE_TERMS = (
    "リース",
    "審査",
    "稟議",
    "承認",
    "否決",
    "要審議",
    "条件",
    "料率",
    "物件",
    "設備",
    "補助金",
    "耐用年数",
    "再リース",
    "工事費",
    "車検",
    "レンタカー",
    "漁業",
    "焼却炉",
    "空調",
    "事業計画",
    "メイン先",
    "メインバンク",
)

CASUAL_ONLY_TERMS = (
    "釣り",
    "サバ",
    "横須賀",
    "横浜",
    "美味しい",
    "お店",
    "よかった",
    "気をつけよう",
)

META_CHAT_TERMS = (
    "リース以外の話",
    "なんでも聞ける",
    "できるのね",
    "よかった",
    "プロジェクトが開始",
    "協力してあげて",
    "推奨フレーズ(top3)",
)

DEFAULT_REQUIRED = [
    ["返済原資", "支払原資", "資金繰り"],
    ["物件", "保全", "残価"],
    ["確認事項", "追加確認", "条件"],
]

DEFAULT_FORBIDDEN = [
    "必ず承認",
    "無条件で承認",
    "確認不要",
]


def load_flywheel(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def build_replay_eval_set(
    flywheel_report: dict[str, Any],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    candidates = list(flywheel_report.get("replay_eval_candidates") or [])
    candidates = sorted(
        [item for item in candidates if isinstance(item, dict)],
        key=lambda item: (
            int(((item.get("scores") or {}).get("total") or 0)),
            int(((item.get("gate") or {}).get("priority") or 0)),
        ),
        reverse=True,
    )

    for candidate in candidates:
        query = extract_query(candidate)
        normalized = normalize_query(query)
        if not normalized or normalized in seen_queries:
            continue
        if not is_lease_relevant(normalized):
            continue
        case = build_case(candidate, normalized, len(cases) + 1)
        cases.append(case)
        seen_queries.add(normalized)
        if len(cases) >= limit:
            break
    return cases


def build_case(candidate: dict[str, Any], query: str, index: int) -> dict[str, Any]:
    concepts, forbidden, uncertainty = concepts_for_query(query)
    source = str(candidate.get("source") or "experience")
    source_hash = re.sub(r"[^a-z0-9_]+", "_", f"{source}_{index:02d}".lower()).strip("_")
    return {
        "id": f"experience_replay_{source_hash}",
        "query": query,
        "required_concepts": concepts,
        "forbidden_claims": forbidden,
        "require_uncertainty": uncertainty,
        "_source": source,
        "_source_id": str(candidate.get("source_id") or ""),
        "_flywheel_candidate_id": str(candidate.get("id") or ""),
        "_gate_reason": str((candidate.get("gate") or {}).get("reason") or ""),
    }


def concepts_for_query(query: str) -> tuple[list[list[str]], list[str], bool]:
    q = query
    if "補助金" in q:
        return (
            [["公募要領", "補助対象"], ["採択", "未採択"], ["返済原資", "支払原資"]],
            ["必ず対象になる", "無条件で使える", "採択前提で問題ない"],
            True,
        )
    if "工事費" in q or "空調" in q or "ダクト" in q:
        return (
            [["工事費", "付帯工事"], ["物件本体", "設備本体"], ["所有権", "原状回復", "撤去"]],
            ["工事費はすべてリース対象", "確認不要", "無条件で対象"],
            True,
        )
    if "電車" in q or "鉄道" in q:
        return (
            [["車両", "設備"], ["長期契約", "保守"], ["資金調達", "返済原資"]],
            ["必ずオフバランス", "会計影響はない"],
            True,
        )
    if "車検" in q or "くるま" in q or "レンタカー" in q or "自動車" in q:
        return (
            [["車検", "登録", "法令"], ["所有者", "使用者"], ["保険", "整備", "事故"]],
            ["車検切れでも問題ない", "登録確認は不要"],
            True,
        )
    if "事業計画" in q:
        return (
            [["売上計画", "収支計画"], ["資金繰り", "返済原資"], ["前提", "根拠"]],
            ["計画だけで承認", "根拠不要"],
            False,
        )
    if "メイン先" in q or "メインバンク" in q:
        return (
            [["メインバンク", "取引状況"], ["支援姿勢", "融資"], ["返済原資", "資金繰り"]],
            ["メイン先なら無条件で通る", "審査不要"],
            False,
        )
    if "飲食" in q:
        return (
            [["厨房設備", "店舗設備"], ["資金繰り", "返済原資"], ["撤退", "中古市場", "保全"]],
            ["飲食業なら一律否決", "開業なら必ず承認"],
            True,
        )
    if "漁業" in q:
        return (
            [["船舶", "漁具", "設備"], ["漁獲", "季節性", "収入変動"], ["保険", "担保", "保全"]],
            ["漁業はリース不可", "季節性は関係ない"],
            True,
        )
    if "焼却炉" in q:
        return (
            [["許認可", "環境規制"], ["設置場所", "撤去費"], ["保守", "処分", "残価"]],
            ["許認可確認は不要", "必ず高く売れる"],
            True,
        )
    if "リース" in q:
        return (DEFAULT_REQUIRED, DEFAULT_FORBIDDEN, True)
    return (DEFAULT_REQUIRED, DEFAULT_FORBIDDEN, False)


def extract_query(candidate: dict[str, Any]) -> str:
    context = candidate.get("context") if isinstance(candidate.get("context"), dict) else {}
    decision = candidate.get("decision") if isinstance(candidate.get("decision"), dict) else {}
    query = (
        str(context.get("question") or "").strip()
        or str(decision.get("issue_text") or "").strip()
        or str(context.get("rule_id") or "").strip()
    )
    return query


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "").strip())[:140]


def is_lease_relevant(query: str) -> bool:
    if any(term in query for term in META_CHAT_TERMS):
        return False
    if not any(term in query for term in LEASE_RELEVANCE_TERMS):
        return False
    if any(term in query for term in CASUAL_ONLY_TERMS) and "リース" not in query:
        return False
    return True


def render_markdown(cases: list[dict[str, Any]]) -> str:
    lines = [
        "# Experience Replay Eval Set",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Cases: {len(cases)}",
        "- Source: `reports/experience_flywheel_latest.json`",
        "- Format: compatible with `scripts/evaluate_answer_quality.py`",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"## {case['id']}",
                f"- Query: {case['query']}",
                f"- Required concepts: {', '.join('/'.join(group) for group in case['required_concepts'])}",
                f"- Forbidden claims: {', '.join(case['forbidden_claims'])}",
                f"- Require uncertainty: {case['require_uncertainty']}",
                f"- Source: {case.get('_source', 'manual')} `{case.get('_flywheel_candidate_id', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(cases: list[dict[str, Any]], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(cases), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flywheel-report", type=Path, default=DEFAULT_FLYWHEEL_REPORT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    cases = build_replay_eval_set(load_flywheel(args.flywheel_report.expanduser()), limit=args.limit)
    write_outputs(cases, output_json=args.output_json.expanduser(), output_md=args.output_md.expanduser())
    print(f"experience replay eval cases: {len(cases)} -> {args.output_json.expanduser()}")
    print(f"saved: {args.output_md.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
