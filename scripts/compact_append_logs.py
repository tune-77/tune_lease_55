#!/usr/bin/env python3
"""追記オンリーの運用ログをローテーション（圧縮）する。

対象は data/ 配下のランタイムログのみ。台帳（ledger）は監査記録なので触らない。
しきい値超過時に、原本を data/archive/ へ退避してから本体を縮約する:

  - keyed  : 同一キーの最後のエントリだけ残す（読み手の「最後のエントリ有効」規約と同じ）
  - tail   : 直近 N 行だけ残す（時系列ログ）

対話室の文脈生成はこれらのファイルを毎リクエスト全走査するため、
無限成長を放置するとチャット応答が徐々に遅くなる。原本はアーカイブに残るので
監査可能性は失わない（planning/shion_autonomy_guards.md の方針どおり）。

使い方:
  python scripts/compact_append_logs.py --dry-run
  python scripts/compact_append_logs.py --apply
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = REPO_ROOT / "data" / "archive"

# (相対パス, モード, キー or 残す行数, 発動しきい値行数)
TARGETS: list[tuple[str, str, object, int]] = [
    ("data/shion_improvement_triage.jsonl", "keyed", "canonical_key", 5000),
    ("data/pipeline_step_log.jsonl", "tail", 2000, 10000),
    ("data/shion_monitor_report_log.jsonl", "tail", 1000, 5000),
    ("data/pipeline_alert_notify_log.jsonl", "tail", 1000, 5000),
]


def _read_lines(path: Path) -> list[str]:
    try:
        return [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        return []


def compact_keyed(lines: list[str], key_field: str) -> list[str]:
    """同一キーは最後のエントリのみ残す（出現順は維持）。壊れた行は保持する。"""
    latest_index: dict[str, int] = {}
    passthrough: list[int] = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
            key = str(row.get(key_field) or "")
        except json.JSONDecodeError:
            key = ""
        if key:
            latest_index[key] = index
        else:
            passthrough.append(index)
    keep = sorted(set(latest_index.values()) | set(passthrough))
    return [lines[i] for i in keep]


def compact_file(path: Path, mode: str, param: object, threshold: int, apply: bool) -> dict:
    lines = _read_lines(path)
    if len(lines) <= threshold:
        return {"path": str(path), "lines": len(lines), "action": "skip（しきい値未満）"}
    if mode == "keyed":
        kept = compact_keyed(lines, str(param))
    else:
        kept = lines[-int(param):]  # type: ignore[arg-type]
    result = {
        "path": str(path),
        "lines": len(lines),
        "kept": len(kept),
        "action": f"compact（{len(lines)}→{len(kept)}行）",
    }
    if apply:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = ARCHIVE_DIR / f"{path.stem}.{stamp}{path.suffix}"
        shutil.copy2(path, archive_path)  # 原本を先に退避（監査可能性の維持）
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        result["archive"] = str(archive_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    for rel_path, kind, param, threshold in TARGETS:
        path = REPO_ROOT / rel_path
        if not path.exists():
            continue
        result = compact_file(path, kind, param, threshold, apply=args.apply)
        print(f"[compact_logs] {result['path']}: {result['action']}"
              + (f" → {result['archive']}" if "archive" in result else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
