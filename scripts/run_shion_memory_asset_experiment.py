"""Run a small local experiment for Shion memory-as-judgment-asset.

The experiment is intentionally deterministic and deploy-free. It compares a
generic baseline review with a memory-aware review that uses the timeline
delta's short/mid/long layers as judgment assets.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMELINE = REPO_ROOT / "data" / "shion_timeline_delta.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "shion_memory_asset_experiment_latest.json"
DEFAULT_REPORT = REPO_ROOT / "reports" / "shion_memory_asset_experiment_latest.md"


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    industry: str
    asset: str
    score: float
    q_risk: float
    customer_type: str
    purpose: str
    memo: str
    actual_result: str


DEMO_CASES = [
    DemoCase(
        case_id="logistics_route_expansion",
        industry="道路貨物運送業",
        asset="車両・運搬車",
        score=83.4,
        q_risk=10.0,
        customer_type="既存先",
        purpose="新規配送ルート対応の増車",
        memo="利益率が薄く燃料費・人件費上昇の影響確認が必要。競合条件との差分も確認したい。",
        actual_result="成約",
    ),
    DemoCase(
        case_id="food_new_store",
        industry="飲食店",
        asset="飲食店設備",
        score=62.8,
        q_risk=100.0,
        customer_type="新規先",
        purpose="新店舗開業に伴う初期設備",
        memo="赤字。出店計画の根拠、自己資金、撤退時の物件処分可能性を確認しないと通しにくい。",
        actual_result="未定",
    ),
    DemoCase(
        case_id="precision_machine_capacity",
        industry="金属製品製造業",
        asset="工作機械",
        score=79.4,
        q_risk=1.8,
        customer_type="既存メイン先",
        purpose="受注増に伴う加工能力増強",
        memo="返済原資と物件用途の説明がしやすい。競合というより稼働計画と受注継続を見たい。",
        actual_result="成約",
    ),
]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def baseline_review(case: DemoCase) -> dict[str, Any]:
    """Generic review that does not use layered memory."""
    questions = [
        "決算内容と返済原資を確認する。",
        "物件の用途と保全性を確認する。",
        "必要に応じて追加資料を依頼する。",
    ]
    reasons = [
        f"スコアは{case.score:.1f}点で、業種・物件・財務の基本確認が必要。",
        "リース審査では返済原資と物件保全を外せない。",
    ]
    return {
        "mode": "baseline",
        "questions": questions,
        "reasons": reasons,
        "uses_explicit_memory_bridge": True,
        "opening": "前回の観点を踏まえて、この案件も確認します。",
    }


def memory_aware_review(case: DemoCase, timeline: dict[str, Any]) -> dict[str, Any]:
    """Use timeline layers as judgment assets, not as visible memory claims."""
    layers = timeline.get("memory_layers") or {}
    long_text = " ".join((layers.get("long_term") or {}).get("promotion_candidates") or [])
    mid_text = " ".join((layers.get("mid_term") or {}).get("items") or [])

    risk_origin = classify_risk_origin(case, long_text=long_text, mid_text=mid_text)
    questions = build_questions(case, risk_origin=risk_origin)
    reasons = build_reasons(case, risk_origin=risk_origin)
    judgment_asset_use = [
        "Q_riskを自動減点ではなく論点分解に使う",
        "事前判断を人間の仮説として扱い、結果登録で検証する",
        "記憶差分は明示せず、確認質問と判断理由に変換する",
    ]
    return {
        "mode": "memory_aware",
        "risk_origin": risk_origin,
        "questions": questions,
        "reasons": reasons,
        "judgment_asset_use": judgment_asset_use,
        "uses_explicit_memory_bridge": False,
        "opening": natural_opening(case, risk_origin),
    }


def classify_risk_origin(case: DemoCase, *, long_text: str, mid_text: str) -> str:
    text = f"{case.memo} {case.purpose} {long_text} {mid_text}"
    if case.score < 65 and case.q_risk >= 70:
        return "credit_and_contract_recovery"
    if case.q_risk < 20 and _contains_any(text, ("稼働", "受注", "返済原資", "用途")):
        return "repayment_source_and_asset_purpose"
    if _contains_any(text, ("競合", "成約", "価格", "条件差", "営業導線")) or case.q_risk >= 30:
        return "competition_or_contract"
    if _contains_any(text, ("稼働", "受注", "返済原資", "用途")):
        return "repayment_source_and_asset_purpose"
    return "standard_credit"


def build_questions(case: DemoCase, *, risk_origin: str) -> list[str]:
    if risk_origin == "credit_and_contract_recovery":
        return [
            "出店計画の損益分岐点、自己資金、運転資金の不足月を確認する。",
            "撤退時に設備をどう処分できるか、保証・頭金・短期化でどこまで補えるかを確認する。",
            "低スコアでも進める外部支援があるなら、銀行支援・補助金・親族/本部支援を分けて確認する。",
        ]
    if risk_origin == "competition_or_contract":
        return [
            "信用悪化ではなく競合・成約リスクかを分けるため、他社条件と当社が取れる条件差を確認する。",
            "薄利でも返済できるか、燃料費・人件費上昇後のルート別採算を確認する。",
            "成約できる場合でも、条件を落としすぎていないか採算下限を確認する。",
        ]
    if risk_origin == "repayment_source_and_asset_purpose":
        return [
            "受注増が一過性でないか、主要受注先・受注残・稼働開始時期を確認する。",
            "工作機械の用途、転用可能性、既存設備との差し替え範囲を確認する。",
            "返済原資が設備導入効果とつながるか、月次の加工能力・粗利改善で確認する。",
        ]
    return [
        "返済原資、物件保全、代表者保証の基本確認に絞る。",
        "追加資料が必要な場合は、判定を変える資料だけに絞る。",
    ]


def build_reasons(case: DemoCase, *, risk_origin: str) -> list[str]:
    if risk_origin == "credit_and_contract_recovery":
        return [
            "低スコアかつ高Q_riskなので、否決前提ではなく救える外部要因と信用悪化を分けて検証する。",
            "新規・赤字・新店舗は、事業計画と撤退時保全が説明できないと条件付き承認に寄せにくい。",
        ]
    if risk_origin == "competition_or_contract":
        return [
            "高スコアでも、Q_riskや営業メモが示すのは信用より成約条件の歪みである可能性が高い。",
            "競合条件を信用リスクとして扱うと、見るべき採算下限と受注確度を外す。",
        ]
    if risk_origin == "repayment_source_and_asset_purpose":
        return [
            "既存メイン先でスコアも高いため、論点は信用悪化より投資効果と稼働根拠に寄る。",
            "返済原資と物件用途がつながれば、稟議では承認理由として再利用しやすい。",
        ]
    return [
        "信用・返済原資・物件保全の基本軸で足りる案件。",
    ]


def natural_opening(case: DemoCase, risk_origin: str) -> str:
    if risk_origin == "credit_and_contract_recovery":
        return "この案件は、否決理由を潰せる材料があるかを先に分けた方がいいです。"
    if risk_origin == "competition_or_contract":
        return "これは信用リスクより、競合・成約条件を先に切り分ける案件です。"
    if risk_origin == "repayment_source_and_asset_purpose":
        return "見るべき中心は、信用不安より投資効果と返済原資のつながりです。"
    return "基本確認を絞れば足りる案件です。"


def evaluate_pair(baseline: dict[str, Any], memory_aware: dict[str, Any]) -> dict[str, Any]:
    base_questions = baseline.get("questions") or []
    aware_questions = memory_aware.get("questions") or []
    return {
        "baseline_question_count": len(base_questions),
        "memory_question_count": len(aware_questions),
        "memory_has_no_explicit_bridge": not bool(memory_aware.get("uses_explicit_memory_bridge")),
        "memory_has_risk_origin": bool(memory_aware.get("risk_origin")),
        "memory_questions_capped": len(aware_questions) <= 3,
        "judgment_asset_count": len(memory_aware.get("judgment_asset_use") or []),
    }


def run_experiment(timeline_path: Path = DEFAULT_TIMELINE) -> dict[str, Any]:
    timeline = _load_json(timeline_path)
    cases = []
    for case in DEMO_CASES:
        baseline = baseline_review(case)
        aware = memory_aware_review(case, timeline)
        cases.append(
            {
                "case": case.__dict__,
                "baseline": baseline,
                "memory_aware": aware,
                "evaluation": evaluate_pair(baseline, aware),
            }
        )
    summary = {
        "case_count": len(cases),
        "no_explicit_bridge_count": sum(1 for item in cases if item["evaluation"]["memory_has_no_explicit_bridge"]),
        "risk_origin_count": sum(1 for item in cases if item["evaluation"]["memory_has_risk_origin"]),
        "question_cap_count": sum(1 for item in cases if item["evaluation"]["memory_questions_capped"]),
    }
    summary["passes_minimum_bar"] = (
        summary["no_explicit_bridge_count"] == summary["case_count"]
        and summary["risk_origin_count"] == summary["case_count"]
        and summary["question_cap_count"] == summary["case_count"]
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "timeline_source": str(timeline_path),
        "purpose": "記憶差分を、露骨な記憶アピールではなく確認質問・判断理由へ変換できるかを見る。",
        "summary": summary,
        "cases": cases,
        "next_action": "実案件/デモ案件で人間が質問の有用性を採点し、当たった確認観点だけを長期判断基準候補に残す。",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Shion Memory Asset Experiment",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- purpose: {payload['purpose']}",
        f"- passes_minimum_bar: {payload['summary']['passes_minimum_bar']}",
        "",
        "## Summary",
        f"- cases: {payload['summary']['case_count']}",
        f"- no explicit bridge: {payload['summary']['no_explicit_bridge_count']}",
        f"- risk origin separated: {payload['summary']['risk_origin_count']}",
        f"- questions capped: {payload['summary']['question_cap_count']}",
        "",
        "## Cases",
    ]
    for item in payload["cases"]:
        case = item["case"]
        aware = item["memory_aware"]
        lines.extend(
            [
                f"### {case['case_id']}",
                f"- industry: {case['industry']} / asset: {case['asset']} / score: {case['score']} / q_risk: {case['q_risk']}",
                f"- natural opening: {aware['opening']}",
                f"- risk origin: {aware['risk_origin']}",
                "- questions:",
                *[f"  - {q}" for q in aware["questions"]],
                "- reasons:",
                *[f"  - {r}" for r in aware["reasons"]],
                "",
            ]
        )
    lines.extend(["## Next Action", f"- {payload['next_action']}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Shion memory asset experiment.")
    parser.add_argument("--timeline", type=Path, default=DEFAULT_TIMELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = run_experiment(args.timeline)
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote={args.output}")
    print(f"report={args.report}")
    print(f"passes_minimum_bar={payload['summary']['passes_minimum_bar']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
