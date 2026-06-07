#!/usr/bin/env python3
"""Lease system gap analyzer.

Read-only program that inventories missing or improvable areas in the lease
screening system and writes a prioritized report. It intentionally does not
modify scoring logic, databases, models, or Obsidian notes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPORTS_DIR = PROJECT_ROOT / "reports"
LATEST_REPORT = REPORTS_DIR / "latest.json"
SIDECAR_JSON = REPORTS_DIR / "agent_sidecar_brief.json"
RAG_EVAL_SET = PROJECT_ROOT / "api" / "knowledge" / "rag_eval_set.json"
TESTS_DIR = PROJECT_ROOT / "tests"
SPECS_DIR = PROJECT_ROOT / "specs"

DEFAULT_OUT_MD = REPORTS_DIR / "lease_system_gap_analysis.md"
DEFAULT_OUT_JSON = REPORTS_DIR / "lease_system_gap_analysis.json"


@dataclass
class GapItem:
    id: str
    title: str
    priority: str
    category: str
    evidence: list[str]
    impact: str
    recommended_action: str
    suggested_program: str
    guardrail: str = "本体スコア・DB・モデルを直接変更しない。まずレポート化して人間確認。"
    source_refs: list[str] = field(default_factory=list)


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob(pattern))


def _latest_improvement_gaps() -> list[GapItem]:
    data = _load_json(LATEST_REPORT)
    if not isinstance(data, dict):
        return []
    items: list[GapItem] = []
    needs_review = data.get("needs_review") or []
    needs_review_count = int(data.get("needs_review_count") or len(needs_review))
    if needs_review_count >= 20:
        top_titles = [
            str(item.get("title") or item.get("id") or "不明")
            for item in needs_review[:5]
            if isinstance(item, dict)
        ]
        items.append(
            GapItem(
                id="GAP-001",
                title="改善候補が滞留し、優先順位と着手可否が見えにくい",
                priority="high",
                category="improvement-ops",
                evidence=[
                    f"reports/latest.json needs_review_count={needs_review_count}",
                    "上位候補: " + " / ".join(top_titles) if top_titles else "上位候補なし",
                ],
                impact="重要なUI・データ・審査改善が埋もれ、毎日の改善パイプラインが提案過多になる。",
                recommended_action=(
                    "needs_review を、審査精度・データ品質・RAG・UI・運用に分類し、"
                    "各カテゴリから最大3件だけ今週のFocusに昇格する。"
                ),
                suggested_program="scripts/lease_system_gap_analyzer.py + reports/lease_system_gap_analysis.md",
                source_refs=["reports/latest.json"],
            )
        )
    high_risk = [
        item
        for item in needs_review
        if isinstance(item, dict)
        and str((item.get("auto_fix_policy") or {}).get("risk") or "").lower() == "high"
    ]
    if high_risk:
        items.append(
            GapItem(
                id="GAP-002",
                title="高リスク改善項目の扱いが手動確認止まり",
                priority="high",
                category="governance",
                evidence=[f"high-risk needs_review={len(high_risk)}"],
                impact="ポートフォリオ・スコア・データ系の改善が安全ゲート不足で放置されやすい。",
                recommended_action="高リスク項目はSPEC化して、承認ゲート・テスト・ロールバック条件を先に作る。",
                suggested_program="scripts/gen_tests_from_spec.py と連携した SPEC skeleton 生成",
                source_refs=["reports/latest.json", "docs/plan.md"],
            )
        )
    return items


def _sidecar_gaps() -> list[GapItem]:
    data = _load_json(SIDECAR_JSON)
    if not isinstance(data, dict):
        return []
    reports = [r for r in data.get("reports", []) if isinstance(r, dict)]
    items: list[GapItem] = []
    stale = [r for r in reports if r.get("stale")]
    risk_text = "\n".join(str(r.get("risks") or "") for r in reports)
    if stale:
        items.append(
            GapItem(
                id="GAP-003",
                title="Sidecarエージェントの監査情報が古い",
                priority="medium",
                category="agent-sidecar",
                evidence=[f"stale sidecar reports={len(stale)}/{len(reports)}"],
                impact="古い指摘を現在の真実として扱う危険がある。一方で再監査のチェックリストとしては有用。",
                recommended_action="古いレポートは自動で『再確認TODO』へ落とし、最新コードで再検証したものだけ有効扱いにする。",
                suggested_program="scripts/agent_sidecar_reader.py に freshness gate / recheck queue を追加",
                source_refs=["reports/agent_sidecar_brief.json"],
            )
        )
    if any(term in risk_text for term in ("テストデータ", "ダミーデータ", "データ不足", "過学習", "判定論理逆転")):
        items.append(
            GapItem(
                id="GAP-004",
                title="スコア根拠・データ品質・モデル信頼性の継続監査が不足",
                priority="critical",
                category="risk-scoring",
                evidence=[
                    "Sidecar reports mention test/dummy data, overfitting, or decision logic inversion.",
                ],
                impact="審査判断の説明性とモデル信頼性を損なう。特に再学習や過去案件比較に影響する。",
                recommended_action=(
                    "スコア計算・学習データ・DB品質を毎回読む監査プログラムを独立運用し、"
                    "本体画面には警告のみ表示する。"
                ),
                suggested_program="scripts/lease_system_gap_analyzer.py の scoring/data checks 拡張",
                source_refs=["reports/agent_sidecar_brief.md"],
            )
        )
    return items


def _rag_gaps(run_rag_eval: bool = False) -> list[GapItem]:
    items: list[GapItem] = []
    cases = _load_json(RAG_EVAL_SET)
    case_count = len(cases) if isinstance(cases, list) else 0
    if case_count < 20:
        items.append(
            GapItem(
                id="GAP-005",
                title="RAG評価セットが小さく、検索品質の継続評価が弱い",
                priority="medium",
                category="rag",
                evidence=[f"rag_eval_set cases={case_count}"],
                impact="検索改善後に、どの質問で悪化したかを検知しにくい。",
                recommended_action="業種別・物件別・補助金・過去案件・否認理由など最低30〜50問へ拡張する。",
                suggested_program="scripts/evaluate_obsidian_rag.py を日次パイプラインに read-only で組み込む",
                source_refs=["api/knowledge/rag_eval_set.json", "scripts/evaluate_obsidian_rag.py"],
            )
        )
    if run_rag_eval:
        try:
            from mobile_app.obsidian_bridge import search_notes

            misses: list[str] = []
            noise: list[str] = []
            if isinstance(cases, list):
                for case in cases:
                    query = str(case.get("query") or "")
                    expected = list(case.get("expected_path_any") or [])
                    forbidden = list(case.get("forbidden_path_any") or [])
                    hits = search_notes(query, limit=5, max_chars=300)
                    paths = [str(hit.get("path") or "") for hit in hits]
                    if expected and not any(any(pattern in path for pattern in expected) for path in paths):
                        misses.append(str(case.get("id") or query))
                    if forbidden and any(any(pattern in path for pattern in forbidden) for path in paths):
                        noise.append(str(case.get("id") or query))
            if misses or noise:
                items.append(
                    GapItem(
                        id="GAP-006",
                        title="RAG検索に期待パス未ヒットまたはノイズ混入がある",
                        priority="high",
                        category="rag",
                        evidence=[
                            f"misses={len(misses)}: {', '.join(misses[:5])}",
                            f"noise={len(noise)}: {', '.join(noise[:5])}",
                        ],
                        impact="AIチャットや審査コメントが根拠の薄いノートを参照する可能性がある。",
                        recommended_action="再ランキングの重み、expected_path の整備、古い/低優先ノートの減点を調整する。",
                        suggested_program="scripts/evaluate_obsidian_rag.py + mobile_app/obsidian_bridge.py rerank tuning",
                        source_refs=["api/knowledge/rag_eval_set.json", "mobile_app/obsidian_bridge.py"],
                    )
                )
        except Exception as exc:
            items.append(
                GapItem(
                    id="GAP-006",
                    title="RAG評価の実行が不安定",
                    priority="medium",
                    category="rag",
                    evidence=[f"rag eval error: {type(exc).__name__}: {exc}"],
                    impact="検索品質を自動検証できない。",
                    recommended_action="RAG評価をChroma依存ではなく obsidian_bridge.search_notes でも走るように保つ。",
                    suggested_program="scripts/evaluate_obsidian_rag.py のフォールバック追加",
                    source_refs=["scripts/evaluate_obsidian_rag.py"],
                )
            )
    return items


def _test_and_spec_gaps() -> list[GapItem]:
    items: list[GapItem] = []
    test_count = _count_files(TESTS_DIR, "test_*.py")
    spec_count = _count_files(SPECS_DIR, "*.md")
    if test_count and spec_count:
        items.append(
            GapItem(
                id="GAP-007",
                title="SPECとテストはあるが、改善候補とのトレースが弱い",
                priority="medium",
                category="quality",
                evidence=[f"test files={test_count}", f"spec files={spec_count}"],
                impact="REV候補が実装された時、どの受入条件を満たしたか追跡しにくい。",
                recommended_action="REV-ID / SPEC-ID / test file の対応表を自動生成し、未対応REVを見える化する。",
                suggested_program="scripts/lease_system_gap_analyzer.py に traceability matrix 出力を追加",
                source_refs=["tests/", "specs/", "reports/latest.json"],
            )
        )
    return items


def _static_code_gaps() -> list[GapItem]:
    items: list[GapItem] = []
    frontend_review = PROJECT_ROOT / "docs" / "nextjs_review.md"
    if frontend_review.exists():
        text = frontend_review.read_text(encoding="utf-8", errors="ignore")
        if "dangerouslySetInnerHTML" in text or "any" in text:
            items.append(
                GapItem(
                    id="GAP-008",
                    title="フロントエンドの安全性・型安全性レビューが本体改善に未接続",
                    priority="high",
                    category="frontend-security",
                    evidence=["docs/nextjs_review.md mentions dangerouslySetInnerHTML or any type risks"],
                    impact="AI回答表示や審査画面でXSS・型崩れ・表示不整合のリスクが残る。",
                    recommended_action="危険HTML表示、APIレスポンス型、モバイル表示崩れを別トラックで優先修正する。",
                    suggested_program="scripts/lease_system_gap_analyzer.py から frontend risk section を出力",
                    source_refs=["docs/nextjs_review.md"],
                )
            )
    db_artifacts = [p for p in (PROJECT_ROOT / "data").glob("*.db-wal")] + [p for p in (PROJECT_ROOT / "data").glob("*.db-shm")]
    if db_artifacts:
        items.append(
            GapItem(
                id="GAP-009",
                title="SQLite WAL/SHMなど実行時ファイルが作業ツリーに残っている",
                priority="low",
                category="repo-hygiene",
                evidence=[_repo_rel(p) for p in db_artifacts[:5]],
                impact="誤コミットやバックアップ混乱の原因になる。",
                recommended_action=".gitignore確認と、DBバックアップ/実行時ファイルの扱いを明文化する。",
                suggested_program="scripts/backup_case_data.py と repo hygiene check の連携",
                source_refs=["data/"],
            )
        )
    return items


def collect_gaps(run_rag_eval: bool = False) -> list[GapItem]:
    items: list[GapItem] = []
    items.extend(_latest_improvement_gaps())
    items.extend(_sidecar_gaps())
    items.extend(_rag_gaps(run_rag_eval=run_rag_eval))
    items.extend(_test_and_spec_gaps())
    items.extend(_static_code_gaps())
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    items.sort(key=lambda item: (priority_order.get(item.priority, 9), item.id))
    return items


def build_markdown(items: list[GapItem]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    counts: dict[str, int] = {}
    for item in items:
        counts[item.priority] = counts.get(item.priority, 0) + 1
    lines = [
        "# Lease System Gap Analysis",
        "",
        f"> Generated: {now} | mode: read-only diagnostics",
        "",
        "## Summary",
        f"- Total gaps: {len(items)}",
        f"- Critical: {counts.get('critical', 0)} / High: {counts.get('high', 0)} / Medium: {counts.get('medium', 0)} / Low: {counts.get('low', 0)}",
        "",
        "## Recommended First Program Track",
        "1. Data/scoring audit warnings: do not change scores automatically; surface warnings only.",
        "2. RAG eval expansion: add more expected-path cases and run retrieval checks daily.",
        "3. Improvement triage: reduce needs_review backlog into a weekly focus list.",
        "",
        "## Gaps",
    ]
    if not items:
        lines.append("_No gaps detected by current checks._")
        return "\n".join(lines).strip() + "\n"

    for item in items:
        lines.extend(
            [
                "",
                f"### {item.id}: {item.title}",
                f"- Priority: **{item.priority}**",
                f"- Category: `{item.category}`",
                f"- Impact: {item.impact}",
                f"- Recommended action: {item.recommended_action}",
                f"- Suggested program: `{item.suggested_program}`",
                f"- Guardrail: {item.guardrail}",
            ]
        )
        if item.evidence:
            lines.append("- Evidence:")
            lines.extend(f"  - {e}" for e in item.evidence)
        if item.source_refs:
            lines.append("- Source refs:")
            lines.extend(f"  - `{ref}`" for ref in item.source_refs)
    return "\n".join(lines).strip() + "\n"


def write_outputs(items: list[GapItem], out_md: Path, out_json: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_markdown(items), encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "mode": "read-only diagnostics",
                "gaps": [asdict(item) for item in items],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-rag-eval", action="store_true", help="Run lightweight Obsidian retrieval checks")
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    args = parser.parse_args()

    items = collect_gaps(run_rag_eval=args.run_rag_eval)
    write_outputs(items, Path(args.out_md), Path(args.out_json))
    print(f"[lease_system_gap_analyzer] gaps={len(items)}")
    print(f"[lease_system_gap_analyzer] wrote: {args.out_md}")
    print(f"[lease_system_gap_analyzer] wrote: {args.out_json}")


if __name__ == "__main__":
    main()
