"""Build Shion's learned practical knowledge map.

Inputs are local and reviewable:
- data/shion_memory_index.json generated from memory/knowledge sources
- approved human judgment feedback in data/lease_data.db

The output augments the hand-authored practical scene map. It does not call
LLMs and does not write to Obsidian.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.shion_practical_knowledge import (  # noqa: E402
    DEFAULT_MAP_PATH,
    extract_practical_entries_from_memory_records,
)
from judgment_feedback import load_judgment_training_candidates  # noqa: E402

DEFAULT_INDEX = REPO_ROOT / "data" / "shion_memory_index.json"
DEFAULT_DB = REPO_ROOT / "data" / "lease_data.db"


def load_memory_records(path: Path = DEFAULT_INDEX) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    records = data.get("records") if isinstance(data, dict) else []
    return records if isinstance(records, list) else []


def judgment_feedback_records(db_path: Path = DEFAULT_DB) -> list[dict[str, Any]]:
    rows = load_judgment_training_candidates(approved_only=True, db_path=str(db_path))
    records: list[dict[str, Any]] = []
    for row in rows:
        reason = str(row.get("reason") or "").strip()
        if not reason:
            continue
        model = str(row.get("model_decision") or "")
        human = str(row.get("human_decision") or "")
        score = row.get("score")
        score_text = f"スコア{float(score):.1f}。 " if score is not None else ""
        records.append(
            {
                "id": f"judgment_feedback_{row.get('id')}",
                "content": f"{model}から{human}へ修正。{score_text}{reason}",
                "memory_type": "judgment_memory",
                "status": "active",
                "confidence": 0.9,
                "source": "judgment_feedback",
                "source_path": f"judgment_feedback:{row.get('id')}",
            }
        )
    return records


def build_map(*, index_path: Path = DEFAULT_INDEX, db_path: Path = DEFAULT_DB) -> dict[str, Any]:
    records = load_memory_records(index_path)
    records.extend(judgment_feedback_records(db_path))
    learned = extract_practical_entries_from_memory_records(records)
    scenes = learned.get("scenes", [])
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "source": {
            "memory_index": str(index_path),
            "judgment_feedback_db": str(db_path),
            "record_count": len(records),
        },
        "loop": {
            "observe": "Obsidian由来の記憶インデックスとレビュー済み判断差分を読む。",
            "classify": "各記録を実践場面へ割り当て、手順層・意味層・判断層へ分類する。",
            "merge": "固定の実践知マップに学習候補を追加する。",
            "use": "チャット時に場面索引として呼び出し、回答を判断資産へ変換する。",
            "feedback": "人間の反応と判断修正が次回のマップ更新材料になる。",
        },
        "summary": {
            "scene_count": len(scenes),
            "entry_count": sum(
                len(scene.get("procedure_layer", []))
                + len(scene.get("meaning_layer", []))
                + len(scene.get("judgment_layer", []))
                for scene in scenes
                if isinstance(scene, dict)
            ),
        },
        "scenes": scenes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Shion practical knowledge map.")
    parser.add_argument("--memory-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--judgment-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_MAP_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = build_map(index_path=args.memory_index, db_path=args.judgment_db)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.dry_run:
        print(text)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(args.output)
    print(f"wrote={args.output}")
    print(f"scene_count={result['summary']['scene_count']}")
    print(f"entry_count={result['summary']['entry_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
