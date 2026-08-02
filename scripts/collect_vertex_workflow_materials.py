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

from api.vertex_distillation import capture_vertex_workflow_result  # noqa: E402
from api.vertex_knowledge_workflows import WORKFLOW_DEFINITIONS, run_vertex_knowledge_workflow  # noqa: E402
from runtime_paths import ICLOUD_OBSIDIAN_DOCS, get_data_dir, resolve_obsidian_vault  # noqa: E402

DEFAULT_TOPICS = [
    "補助金前提の工作機械リース",
    "再リース案件の耐用年数と残価確認",
    "物件別の中古流動性と残価リスク",
    "返済余力と資金繰りの異常兆候",
    "運送業の燃料費・ドライバー不足・2024年問題",
    "銀行支援・保証・条件変更の実効性",
    "契約・所有権・検収・詐欺リスク",
    "設備稼働率・保守・更新投資",
    "金利・料率・競合条件の組み立て",
    "延滞・回収・倒産時の物件保全",
    "業種別の倒産要因と先行指標",
    "インボイス制度とリース審査への影響",
    "ソフトウェアリースの会計・権利・解約リスク",
    "医療機器リースの稼働率と診療報酬リスク",
    "建設機械リースの稼働季節性と転売可能性",
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
) -> dict[str, Any]:
    selected_topics = [topic for topic in topics if topic][:limit]
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
        "skipped_count": 0,
        "error_count": 0,
    }

    for topic in selected_topics:
        for mode in modes:
            entry: dict[str, Any] = {"topic": topic, "mode": mode}
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
    )
    print(f"topics={report['topic_count']}")
    print(f"modes={','.join(report['modes'])}")
    print(f"captured={report['captured_count']}")
    print(f"duplicates={report['duplicate_count']}")
    print(f"skipped={report['skipped_count']}")
    print(f"errors={report['error_count']}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
