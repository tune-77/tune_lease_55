#!/usr/bin/env python3
"""Build a read-only architecture layer audit for Shion.

This report borrows the harness / agents / improver lens from another domain,
but it does not change runtime behavior. It only reads local source/report
artifacts and writes a local JSON/Markdown report.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_JSON = PROJECT_ROOT / "data" / "canonical_judgment_rules.json"
DEFAULT_FIELD_REVIEW_JSON = PROJECT_ROOT / "reports" / "judgment_asset_field_review_latest.json"
DEFAULT_GROWTH_JSON = PROJECT_ROOT / "reports" / "shion_growth_evaluation_latest.json"
DEFAULT_MEMORY_EFFECTIVENESS_JSON = PROJECT_ROOT / "reports" / "obsidian_memory_effectiveness_latest.json"
DEFAULT_LOOP_ENGINEERING_JSON = PROJECT_ROOT / "reports" / "loop_engineering_latest.json"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "shion_architecture_layer_audit_latest.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "shion_architecture_layer_audit_latest.md"

GUARDRAIL = "read_only_no_prompt_no_scoring_no_rag_no_memory_write_no_auto_promotion"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _canonical_rules(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    rules = canonical.get("rules")
    if isinstance(rules, list):
        return [item for item in rules if isinstance(item, dict)]
    rules = canonical.get("canonical_rules")
    if isinstance(rules, list):
        return [item for item in rules if isinstance(item, dict)]
    return []


def _active_rules(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    return [rule for rule in _canonical_rules(canonical) if str(rule.get("status") or "active") == "active"]


def _status_counts(canonical: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rule in _canonical_rules(canonical):
        status = str(rule.get("status") or "active").strip() or "active"
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _lineage_count(rules: list[dict[str, Any]]) -> int:
    count = 0
    concept_counts: dict[str, int] = {}
    for rule in rules:
        parent_ids = rule.get("parent_ids")
        if isinstance(parent_ids, list) and any(str(item).strip() for item in parent_ids):
            count += 1
            continue
        concept = str(rule.get("concept") or "").strip()
        if concept:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
    for total in concept_counts.values():
        if total > 1:
            count += total - 1
    return count


def _path_flags(project_root: Path, present_paths: set[str] | None = None) -> dict[str, bool]:
    paths = {
        "shion_eval_health_ui": "frontend/src/app/shion-eval-health/page.tsx",
        "judgment_asset_graph_ui": "frontend/src/app/judgment-asset-graph/page.tsx",
        "judgment_asset_graph_generator": "scripts/build_judgment_asset_graph.py",
        "canonical_promotion": "scripts/promote_canonical_judgment_rules.py",
        "field_feedback_recorder": "scripts/record_judgment_asset_feedback.py",
        "preflight_guard": "scripts/preflight_pr_guard.py",
        "screening_ui": "frontend/src/app/screening/page.tsx",
        "lease_intelligence_ui": "frontend/src/app/lease-intelligence/page.tsx",
        "improvement_log_ui": "frontend/src/app/improvement-log/page.tsx",
    }
    if present_paths is None:
        return {key: (project_root / rel).exists() for key, rel in paths.items()}
    return {key: rel in present_paths for key, rel in paths.items()}


def _score_layer(strength_count: int, risk_count: int) -> str:
    if risk_count >= 2:
        return "attention"
    if strength_count >= 3 and risk_count == 0:
        return "strong"
    if strength_count >= 2:
        return "usable"
    return "thin"


def build_audit(
    *,
    target_date: str,
    canonical: dict[str, Any],
    field_review: dict[str, Any],
    growth: dict[str, Any],
    memory_effectiveness: dict[str, Any],
    loop_engineering: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
    present_paths: set[str] | None = None,
) -> dict[str, Any]:
    rules = _active_rules(canonical)
    path_flags = _path_flags(project_root, present_paths)
    field_summary = field_review.get("summary") if isinstance(field_review.get("summary"), dict) else {}
    growth_judgment = growth.get("judgment") if isinstance(growth.get("judgment"), dict) else {}
    growth_dimensions = growth.get("dimensions") if isinstance(growth.get("dimensions"), dict) else {}
    field_validation = growth_dimensions.get("field_validation") if isinstance(growth_dimensions.get("field_validation"), dict) else {}
    memory_summary = memory_effectiveness.get("summary") if isinstance(memory_effectiveness.get("summary"), dict) else {}
    loop_status = str(loop_engineering.get("status") or "")

    sleeping = int(field_summary.get("sleeping") or 0)
    active_rule_count = len(rules)
    lineage_count = _lineage_count(rules)
    field_validation_score = float(field_validation.get("score") or 0)

    harness_strengths = [
        "評価ヘルス画面で成長・記憶・改善ログを読める" if path_flags["shion_eval_health_ui"] else "",
        "判断資産昇格スクリプトがあり、正式資産の入口を分けられる" if path_flags["canonical_promotion"] else "",
        "preflight guard により実装前後の最低限の破綻を検出できる" if path_flags["preflight_guard"] else "",
        "判断資産系統樹で親子関係を読み取り専用に可視化している" if path_flags["judgment_asset_graph_ui"] else "",
    ]
    harness_risks = [
        "candidate/proposed/approved/active/deprecated の承認状態が運用全体のゲートとしてはまだ弱い"
        if set(_status_counts(canonical)) <= {"active"}
        else "",
        "正式判断資産への反映可否を人間が確認する専用画面はまだ薄い",
    ]

    agents_strengths = [
        "審査画面・リース知性体画面があり、判断生成と対話の入口が分かれている"
        if path_flags["screening_ui"] and path_flags["lease_intelligence_ui"]
        else "",
        "評価GUIにより回答品質を実行後に点検できる" if path_flags["shion_eval_health_ui"] else "",
        "記憶・RAG参照の効果測定レポートがあり、参照しただけで終わらない" if memory_summary else "",
    ]
    agents_risks = [
        "作成AI・レビューAI・調査AIの役割境界は概念としてはあるが、harnessが強制する構造にはまだ寄せ切っていない",
        "どの判断資産を使ったかを案件ごとの manifest として保存する仕組みは今後の課題",
    ]

    improver_strengths = [
        "改善ログと loop engineering により改善ループの状態を読める" if path_flags["improvement_log_ui"] else "",
        "判断資産フィードバック記録スクリプトがあり、実案件反応を後から残せる" if path_flags["field_feedback_recorder"] else "",
        "判断資産に親子系統を持たせ始めており、派生の履歴を追える" if lineage_count else "",
    ]
    improver_risks = [
        f"active 判断資産 {active_rule_count} 件のうち、実案件フィードバック未記録が {sleeping} 件ある" if sleeping else "",
        "field_validation が 0 のため、資産は増えているが実戦で効いた証拠がまだ弱い" if field_validation_score == 0 else "",
        "改善提案を正式資産にする承認ゲートは、まず read-only 設計から固めるべき",
    ]

    layers = [
        {
            "id": "harness",
            "name": "harness: 型と安全ゲート",
            "role": "必須確認、様式、承認、記憶汚染防止、評価ゲートを決定的に管理する層。",
            "status": _score_layer(len([x for x in harness_strengths if x]), len([x for x in harness_risks if x])),
            "strengths": [x for x in harness_strengths if x],
            "risks": [x for x in harness_risks if x],
            "safe_next_step": "正式判断資産への昇格条件を read-only レポートで定義し、いきなり自動適用しない。",
        },
        {
            "id": "agents",
            "name": "agents: 判断を作るAI",
            "role": "審査コメント、確認質問、調査、レビューを担うAIの層。",
            "status": _score_layer(len([x for x in agents_strengths if x]), len([x for x in agents_risks if x])),
            "strengths": [x for x in agents_strengths if x],
            "risks": [x for x in agents_risks if x],
            "safe_next_step": "各回答で使った判断資産・根拠・質問を manifest として観測する案を作る。まだ回答生成には接続しない。",
        },
        {
            "id": "improver",
            "name": "improver: 判断資産を育てる層",
            "role": "実案件フィードバック、改善ログ、系統樹から再利用できる勘所を提案する層。",
            "status": _score_layer(len([x for x in improver_strengths if x]), len([x for x in improver_risks if x])),
            "strengths": [x for x in improver_strengths if x],
            "risks": [x for x in improver_risks if x],
            "safe_next_step": "未使用判断資産を削除せず、次の実案件で1件ずつ helped/neutral/challenged を記録する。",
        },
    ]

    overall_status = "attention" if any(layer["status"] == "attention" for layer in layers) else "ok"
    guardrails = [
        "既存の回答生成、RAG検索、スコアリング、日次改善パイプラインには接続しない。",
        "AI提案は正式判断資産に自動昇格しない。",
        "未承認のAI出力を長期記憶へ同期しない。",
        "まず観測レポートとして不足を見える化し、人間承認後に小さく実装する。",
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date": target_date,
        "schema_version": 1,
        "mode": "read_only_architecture_audit",
        "guardrail": GUARDRAIL,
        "inspiration": "harness_agents_improver_layering",
        "overall_status": overall_status,
        "summary": {
            "active_judgment_assets": active_rule_count,
            "lineage_assets": lineage_count,
            "field_feedback_sleeping": sleeping,
            "field_validation_score": field_validation_score,
            "growth_label": growth_judgment.get("label", ""),
            "loop_engineering_status": loop_status,
            "memory_records_used": memory_summary.get("used", ""),
        },
        "source_status": {
            "canonical_status_counts": _status_counts(canonical),
            "path_flags": path_flags,
        },
        "layers": layers,
        "guardrails": guardrails,
        "safe_sequence": [
            "1. read-only 監査レポートで層の不足を見る。",
            "2. 実案件フィードバックを1件ずつ記録する。",
            "3. 承認状態ラベル案をレポートで固める。",
            "4. 人間承認後にだけ判断資産へ反映する。",
        ],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# 紫苑 アーキテクチャ層監査",
        "",
        f"- Date: {payload.get('date')}",
        f"- Mode: {payload.get('mode')}",
        f"- Guardrail: {payload.get('guardrail')}",
        f"- Overall status: {payload.get('overall_status')}",
        "",
        "## これは何か",
        "",
        "HOIKUAGENT 型の `harness / agents / improver` という見方を、紫苑に壊さず当てはめた read-only 監査です。",
        "既存の回答生成、RAG、スコアリング、日次改善パイプラインには接続しません。",
        "",
        "## Summary",
        "",
        f"- Active judgment assets: {summary.get('active_judgment_assets', 0)}",
        f"- Lineage assets: {summary.get('lineage_assets', 0)}",
        f"- Field feedback sleeping: {summary.get('field_feedback_sleeping', 0)}",
        f"- Field validation score: {summary.get('field_validation_score', 0)}",
        f"- Growth: {summary.get('growth_label') or 'n/a'}",
        f"- Loop engineering: {summary.get('loop_engineering_status') or 'n/a'}",
        "",
        "## 3層の見立て",
        "",
    ]
    for layer in payload.get("layers") or []:
        lines += [
            f"### {layer.get('name')}",
            "",
            f"- Status: {layer.get('status')}",
            f"- Role: {layer.get('role')}",
            "- 強み:",
        ]
        strengths = layer.get("strengths") or []
        lines += [f"  - {item}" for item in strengths] if strengths else ["  - なし"]
        lines.append("- リスク:")
        risks = layer.get("risks") or []
        lines += [f"  - {item}" for item in risks] if risks else ["  - なし"]
        lines += [
            f"- 安全な次の一手: {layer.get('safe_next_step')}",
            "",
        ]
    lines += [
        "## 壊さないためのガードレール",
        "",
    ]
    for item in payload.get("guardrails") or []:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Safe Sequence",
        "",
    ]
    for item in payload.get("safe_sequence") or []:
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def write_reports(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> dict[str, str]:
    _write_json(output_json, payload)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown(payload), encoding="utf-8")
    return {"json": str(output_json), "markdown": str(output_md)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--field-review-json", type=Path, default=DEFAULT_FIELD_REVIEW_JSON)
    parser.add_argument("--growth-json", type=Path, default=DEFAULT_GROWTH_JSON)
    parser.add_argument("--memory-effectiveness-json", type=Path, default=DEFAULT_MEMORY_EFFECTIVENESS_JSON)
    parser.add_argument("--loop-engineering-json", type=Path, default=DEFAULT_LOOP_ENGINEERING_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--date", default=date.today().isoformat())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_audit(
        target_date=str(args.date),
        canonical=_read_json(args.canonical_json),
        field_review=_read_json(args.field_review_json),
        growth=_read_json(args.growth_json),
        memory_effectiveness=_read_json(args.memory_effectiveness_json),
        loop_engineering=_read_json(args.loop_engineering_json),
    )
    paths = write_reports(payload, output_json=args.output_json, output_md=args.output_md)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "active_judgment_assets": payload["summary"]["active_judgment_assets"],
                "field_feedback_sleeping": payload["summary"]["field_feedback_sleeping"],
                "paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
