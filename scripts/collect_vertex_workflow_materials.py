#!/usr/bin/env python3
"""Collect reusable Vertex AI Search workflow materials into Obsidian.

This is intentionally bounded. Vertex is used as a temporary, supplementary
index while credits are available; saved notes remain local Obsidian material
with ``review_status: needs_human_review``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.vertex_distillation import _canonical_query_key, capture_vertex_workflow_result  # noqa: E402
from api.vertex_knowledge_workflows import (  # noqa: E402
    WORKFLOW_DEFINITIONS,
    build_workflow_query,
    run_vertex_knowledge_workflow,
)
from runtime_paths import ICLOUD_OBSIDIAN_DOCS, get_data_dir, resolve_obsidian_vault  # noqa: E402

DEFAULT_TOPICS = [
    "補助金前提の工作機械リース",
    "補助金不採択時の返済余力確認",
    "補助金交付前発注とリース対象可否",
    "ものづくり補助金と設備投資リースの確認論点",
    "再リース案件の耐用年数と残価確認",
    "再リース料の妥当性と物件状態確認",
    "物件別の中古流動性と残価リスク",
    "中古工作機械の転売可能性と査定リスク",
    "半導体製造装置リースの陳腐化リスク",
    "返済余力と資金繰りの異常兆候",
    "営業キャッシュフロー赤字先の設備投資判断",
    "売上急増先の運転資金不足とリース審査",
    "運送業の燃料費・ドライバー不足・2024年問題",
    "物流倉庫自動化設備リースの投資回収リスク",
    "銀行支援・保証・条件変更の実効性",
    "信用保証協会付き融資とリース審査の見方",
    "契約・所有権・検収・詐欺リスク",
    "サプライヤー直送案件の検収確認と架空物件リスク",
    "設備稼働率・保守・更新投資",
    "保守契約なし設備の故障停止リスク",
    "金利・料率・競合条件の組み立て",
    "高金利環境下のリース料率説明と競合対策",
    "延滞・回収・倒産時の物件保全",
    "倒産時に引き揚げ困難な設備の保全リスク",
    "業種別の倒産要因と先行指標",
    "インボイス制度とリース審査への影響",
    "ソフトウェアリースの会計・権利・解約リスク",
    "SaaS導入費用とソフトウェアリースの資産性",
    "医療機器リースの稼働率と診療報酬リスク",
    "歯科医院の医療機器リースと患者数変動リスク",
    "建設機械リースの稼働季節性と転売可能性",
    "飲食業の厨房設備リースと原価高騰リスク",
    "宿泊業の改装設備リースと稼働率リスク",
    "美容業の設備リースと店舗撤退リスク",
    "農業機械リースの季節性と補助金依存リスク",
    "太陽光設備リースの売電単価と制度変更リスク",
    "蓄電池設備リースの補助金と技術陳腐化",
    "EV充電設備リースの需要予測と設置場所リスク",
    "印刷業の設備更新と市場縮小リスク",
    "食品製造業の冷凍冷蔵設備と電気代高騰リスク",
    "介護事業者の設備リースと介護報酬改定リスク",
    "新設法人の設備リースと代表者経験確認",
    "既存取引先の追加設備投資と過剰投資リスク",
    "新リース会計基準がリース審査・物件保全に与える影響の体系整理",
    "業種別中古設備のリセールバリューと陳腐化速度の比較",
    "倒産手続き種別ごとのリース物件回収実務",
    "設備投資系補助金・税制優遇の網羅的比較とリース活用スキーム",
    "EV充電設備リースの法規制・電力供給インフラ・設置場所契約リスク",
    "信用保証協会付き融資の代位弁済・求償債務とリース審査への影響",
    "既存取引先の追加設備投資における過剰投資リスクの定量評価基準",
    "業種特化型の需要変動リスク（稼働率・患者数・店舗撤退）の審査反映方法",
]

DEFAULT_MODES = ["evidence_support", "judgment_candidates", "knowledge_audit"]
LEGACY_NORMAL_VAULT = ICLOUD_OBSIDIAN_DOCS / "Obsidian Vault"
DEFAULT_REPORT = REPO_ROOT / "reports" / "vertex_workflow_materials_latest.json"


def default_vault_path() -> Path:
    if LEGACY_NORMAL_VAULT.is_dir():
        return LEGACY_NORMAL_VAULT
    return resolve_obsidian_vault()


def load_topics(path: Path | None) -> list[str]:
    if not path:
        return list(DEFAULT_TOPICS)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
        if isinstance(data, dict):
            values = data.get("topics") or []
            return [str(item).strip() for item in values if str(item).strip()]
    topics: list[str] = []
    for line in text.splitlines():
        clean = line.strip()
        if clean and not clean.startswith("#"):
            topics.append(clean)
    return topics


def normalize_modes(raw_modes: list[str]) -> list[str]:
    modes: list[str] = []
    for raw in raw_modes:
        for part in str(raw or "").split(","):
            mode = part.strip()
            if mode in WORKFLOW_DEFINITIONS and mode not in modes:
                modes.append(mode)
    return modes or ["evidence_support"]


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def existing_capture_key(topic: str, mode: str) -> str:
    query = build_workflow_query(topic, mode)
    return _canonical_query_key(f"{mode}\n{query}")


def collect_materials(
    *,
    topics: list[str],
    modes: list[str],
    vault_path: Path,
    state_path: Path,
    report_path: Path,
    page_size: int,
    limit: int,
    sleep_seconds: float,
    dry_run: bool,
    skip_existing: bool = True,
) -> dict[str, Any]:
    selected_topics = [topic for topic in topics if topic][:limit]
    existing_state = read_state(state_path)
    report: dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "vault_path": str(vault_path),
        "state_path": str(state_path),
        "topic_count": len(selected_topics),
        "modes": modes,
        "page_size": page_size,
        "entries": [],
        "captured_count": 0,
        "duplicate_count": 0,
        "existing_skipped_count": 0,
        "skipped_count": 0,
        "error_count": 0,
    }

    for topic in selected_topics:
        for mode in modes:
            entry: dict[str, Any] = {"topic": topic, "mode": mode}
            if skip_existing:
                key = existing_capture_key(topic, mode)
                existing = existing_state.get(key)
                if existing:
                    entry["skipped_existing"] = True
                    entry["note_path"] = existing.get("note_path")
                    report["existing_skipped_count"] += 1
                    report["entries"].append(entry)
                    write_report(report_path, report)
                    continue
            if dry_run:
                entry["planned"] = True
                report["entries"].append(entry)
                continue
            try:
                workflow = run_vertex_knowledge_workflow(topic, mode=mode, page_size=page_size)
                capture = capture_vertex_workflow_result(
                    workflow,
                    vault_path=vault_path,
                    state_path=state_path,
                )
                entry.update(
                    {
                        "query": workflow.get("query"),
                        "search_status": (workflow.get("search") or {}).get("status"),
                        "answer_status": (workflow.get("answer") or {}).get("status"),
                        "refs": workflow.get("refs") or [],
                        "capture": capture,
                    }
                )
                if capture.get("captured") and capture.get("duplicate"):
                    report["duplicate_count"] += 1
                elif capture.get("captured"):
                    report["captured_count"] += 1
                    key = existing_capture_key(topic, mode)
                    existing_state[key] = {
                        "note_path": capture.get("note_path"),
                        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "mode": mode,
                        "topic": topic,
                    }
                else:
                    report["skipped_count"] += 1
            except Exception as exc:  # noqa: BLE001 - batch should continue
                entry["error"] = str(exc)[:300]
                report["error_count"] += 1
            report["entries"].append(entry)
            write_report(report_path, report)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    write_report(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics-file", type=Path)
    parser.add_argument("--topic", action="append", default=[], help="Add one topic. Can be repeated.")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--vault", type=Path, default=default_vault_path())
    parser.add_argument(
        "--state",
        type=Path,
        default=get_data_dir() / "vertex_workflow_material_collection_state.json",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    topics = [str(item).strip() for item in args.topic if str(item).strip()]
    if not topics:
        topics = load_topics(args.topics_file)
    modes = normalize_modes(args.modes)
    limit = max(1, min(100, int(args.limit or 1)))
    page_size = max(1, min(10, int(args.page_size or 5)))

    report = collect_materials(
        topics=topics,
        modes=modes,
        vault_path=args.vault,
        state_path=args.state,
        report_path=args.report,
        page_size=page_size,
        limit=limit,
        sleep_seconds=max(0.0, float(args.sleep_seconds or 0.0)),
        dry_run=bool(args.dry_run),
        skip_existing=not bool(args.no_skip_existing),
    )
    print(f"topics={report['topic_count']}")
    print(f"modes={','.join(report['modes'])}")
    print(f"captured={report['captured_count']}")
    print(f"duplicates={report['duplicate_count']}")
    print(f"existing_skipped={report['existing_skipped_count']}")
    print(f"skipped={report['skipped_count']}")
    print(f"errors={report['error_count']}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
