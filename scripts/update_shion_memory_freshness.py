"""紫苑記憶索引の鮮度更新（docs/shion_memory_architecture.md Next Step 2 の実装）。

使用ログ（data/shion_memory_usage_log.jsonl、`build_recall_prompt_block` が追記）
から各記憶の last_used_at を求め、長期間使われていない記憶を `stale` に落とす。
使用ログが真実の源なので、索引を再生成しても本スクリプトの再実行で状態を再現できる。

ルール:
- last_used_at = 使用ログ上の最新利用日。
- active かつ作成から --stale-days 超、かつ直近 --stale-days 以内の利用が無い → stale。
- stale でも直近利用があれば active に戻す（鮮度確認済みとみなす）。
- value_memory（Mana・良心などの上位規範）は経年で stale に落とさない。
- 削除はしない（アーキテクチャ方針: 古い記憶は revised / deprecated / stale へ）。

使い方:
    python3 scripts/update_shion_memory_freshness.py            # data/ の索引を更新
    python3 scripts/update_shion_memory_freshness.py --dry-run  # 変更内容の確認のみ
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INDEX = REPO_ROOT / "data" / "shion_memory_index.json"
DEFAULT_USAGE_LOG = REPO_ROOT / "data" / "shion_memory_usage_log.jsonl"
# scripts/sync_cloudrun_inputs_from_gcs.py が GCS から取り込むイベントの置き場
DEFAULT_CLOUDRUN_EVENTS_DIR = REPO_ROOT / "data" / "cloudrun_inputs"
DEFAULT_STALE_DAYS = 45
DEFAULT_RAG_FEEDBACK = REPO_ROOT / "data" / "rag_feedback_log.jsonl"

# 上位規範は経年で鮮度切れ扱いにしない
_NEVER_STALE_TYPES = {"value_memory"}

# フィードバック評価の分類（api/main.py の human_response 系と同じ語彙）
_NEGATIVE_RATINGS = {"bad", "wrong", "needs_fix", "thin", "not_shion"}
_POSITIVE_RATINGS = {"good", "useful", "shion_like"}


def load_feedback_signals(path: Path) -> tuple[set[str], set[str]]:
    """rag_feedback_log から (低評価ノート名, 高評価ノート名) の集合を返す。

    記憶レコードとは出典ノートのファイル名で突き合わせる。同じノートに
    高評価と低評価の両方が付いている場合は降格させない（保守的に扱う）。
    """
    negative: set[str] = set()
    positive: set[str] = set()
    if not path.exists():
        return negative, positive
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        rating = str(row.get("rating") or "").strip().lower()
        name = Path(str(row.get("obsidian_ref") or row.get("doc_id") or "")).name
        if not name:
            continue
        if rating in _NEGATIVE_RATINGS:
            negative.add(name)
        elif rating in _POSITIVE_RATINGS:
            positive.add(name)
    return negative - positive, positive


def load_usage_dates(path: Path) -> dict[str, str]:
    """使用ログを読み、記憶ID → 最新利用日(YYYY-MM-DD) を返す。壊れた行は無視。"""
    latest: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        used_on = str(entry.get("ts") or "")[:10]
        if not used_on:
            continue
        for ref in entry.get("refs") or []:
            rid = str(ref)
            if rid and used_on > latest.get(rid, ""):
                latest[rid] = used_on
    return latest


def load_usage_dates_from_cloudrun_events(dir_path: Path = DEFAULT_CLOUDRUN_EVENTS_DIR) -> dict[str, str]:
    """GCSから同期したCloud Runイベントのうち shion_memory_usage を使用日として取り込む。

    Cloud Run 上の想起は `record_cloudrun_input_event` 経由で
    cloudrun-inputs/ へミラーされるため、ローカルの使用ログと合流させる。
    """
    latest: dict[str, str] = {}
    if not dir_path.is_dir():
        return {}
    for path in sorted(dir_path.rglob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict) or entry.get("event_type") != "shion_memory_usage":
                continue
            payload = entry.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            used_on = str(payload.get("ts") or entry.get("ts") or "")[:10]
            if not used_on:
                continue
            for ref in payload.get("refs") or []:
                rid = str(ref)
                if rid and used_on > latest.get(rid, ""):
                    latest[rid] = used_on
    return latest


def merge_usage_dates(*sources: dict[str, str]) -> dict[str, str]:
    """複数の使用日ソースを、記憶IDごとに最新日を採用して統合する。"""
    merged: dict[str, str] = {}
    for source in sources:
        for rid, used_on in source.items():
            if used_on > merged.get(rid, ""):
                merged[rid] = used_on
    return merged


def _parse_date(value: str) -> date | None:
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def apply_freshness(
    index: dict[str, Any],
    usage_dates: dict[str, str],
    *,
    stale_days: int = DEFAULT_STALE_DAYS,
    today: date | None = None,
    negative_files: set[str] | None = None,
    positive_files: set[str] | None = None,
) -> dict[str, int]:
    """索引レコードへ last_used_at 反映と stale 昇降格を行い、変更件数を返す。

    時間ベース（stale_days）に加えてフィードバック連動の忘却を行う:
    - 出典ノートに低評価が付いた記憶は経過日数を待たず stale へ即降格
    - 高評価が付いた記憶は「鮮度確認済み」として経年降格から保護
    """
    today = today or date.today()
    cutoff = today - timedelta(days=stale_days)
    negative_files = negative_files or set()
    positive_files = positive_files or set()
    summary = {
        "last_used_updated": 0,
        "demoted_to_stale": 0,
        "revived_to_active": 0,
        "demoted_by_feedback": 0,
    }

    for record in index.get("records") or []:
        if not isinstance(record, dict):
            continue
        rid = str(record.get("id") or "")
        status = str(record.get("status") or "active")
        memory_type = str(record.get("memory_type") or "")
        source_name = Path(str(record.get("source_path") or "")).name

        used_on = usage_dates.get(rid, "")
        if used_on and used_on != str(record.get("last_used_at") or ""):
            record["last_used_at"] = used_on
            summary["last_used_updated"] += 1

        if status not in {"active", "stale"}:
            continue  # revised / deprecated / private は鮮度で動かさない

        if (
            status == "active"
            and memory_type not in _NEVER_STALE_TYPES
            and source_name
            and source_name in negative_files
        ):
            record["status"] = "stale"
            summary["demoted_by_feedback"] += 1
            continue

        last_used = _parse_date(str(record.get("last_used_at") or ""))
        created = _parse_date(str(record.get("created_at") or ""))
        recently_used = (last_used is not None and last_used >= cutoff) or (
            bool(source_name) and source_name in positive_files
        )

        if status == "stale" and recently_used:
            record["status"] = "active"
            summary["revived_to_active"] += 1
        elif (
            status == "active"
            and memory_type not in _NEVER_STALE_TYPES
            and not recently_used
            and created is not None
            and created < cutoff
        ):
            record["status"] = "stale"
            summary["demoted_to_stale"] += 1

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="紫苑記憶索引の last_used_at 更新と stale 降格")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--usage-log", type=Path, default=DEFAULT_USAGE_LOG)
    parser.add_argument("--cloudrun-events-dir", type=Path, default=DEFAULT_CLOUDRUN_EVENTS_DIR)
    parser.add_argument("--stale-days", type=int, default=DEFAULT_STALE_DAYS)
    parser.add_argument("--rag-feedback", type=Path, default=DEFAULT_RAG_FEEDBACK)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        index = json.loads(args.index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"索引を読めません: {args.index} ({exc})")
        return 1

    usage_dates = merge_usage_dates(
        load_usage_dates(args.usage_log),
        load_usage_dates_from_cloudrun_events(args.cloudrun_events_dir),
    )
    negative_files, positive_files = load_feedback_signals(args.rag_feedback)
    summary = apply_freshness(
        index,
        usage_dates,
        stale_days=args.stale_days,
        negative_files=negative_files,
        positive_files=positive_files,
    )

    print(f"usage_log_refs={len(usage_dates)}")
    print(f"feedback_negative_files={len(negative_files)} feedback_positive_files={len(positive_files)}")
    for key, count in summary.items():
        print(f"{key}={count}")

    if args.dry_run:
        print("dry-run: 索引は書き換えていません")
        return 0

    text = json.dumps(index, ensure_ascii=False, indent=2)
    tmp = args.index.with_suffix(args.index.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(args.index)
    print(f"wrote={args.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
