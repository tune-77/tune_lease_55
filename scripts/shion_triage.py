#!/usr/bin/env python3
"""紫苑トリアージ記録の共有リーダー（Phase 2: planning/shion_improvement_loop_plan.md）。

data/shion_improvement_triage.jsonl（追記形式・最後のエントリ有効）を読み、
build_codex_auto_queue.py / auto_approve_safe_recipes.py から共通で使う。

トリアージモード（P2-0 / P2-4）:
  - off:    トリアージを一切参照しない（切り戻し用の安全弁）
  - shadow: 実キューは従来のまま、比較結果だけ出力する（既定）
  - live:   「捨てる」を除外し「今日やる（承認済み）」を優先する
優先順位: CLI引数 > 環境変数 SHION_TRIAGE_QUEUE_MODE > 既定 shadow
"""

from __future__ import annotations

import json
import os
from pathlib import Path

TRIAGE_MODES = ("off", "shadow", "live")
TRIAGE_MODE_ENV = "SHION_TRIAGE_QUEUE_MODE"
TRIAGE_FILE_RELPATH = Path("data") / "shion_improvement_triage.jsonl"


def resolve_triage_mode(cli_value: str | None = None) -> str:
    for candidate in (cli_value, os.environ.get(TRIAGE_MODE_ENV)):
        value = str(candidate or "").strip().lower()
        if value in TRIAGE_MODES:
            return value
    return "shadow"


def load_triage_latest(root: Path) -> dict[str, dict]:
    """canonical_key → 最後のトリアージ記録。破損行・ファイル欠損に耐える。"""
    path = root / TRIAGE_FILE_RELPATH
    if not path.exists():
        return {}
    latest: dict[str, dict] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("canonical_key") or "")
            if key:
                latest[key] = row
    except OSError:
        return {}
    return latest


def triage_record_for_item(latest: dict[str, dict], item: dict) -> dict | None:
    """改善アイテムに対応するトリアージ記録を返す。

    同定は冗長: canonical_key 一致を優先し、無ければ item_id (REV-XXX) で照合する
    （canonical_key は表示ロジック変更でドリフトした実例があるため）。
    """
    if not latest:
        return None
    canonical_key = str(item.get("canonical_key") or "")
    if canonical_key and canonical_key in latest:
        return latest[canonical_key]
    item_id = str(item.get("id") or item.get("rev") or "").strip()
    if not item_id:
        return None
    for record in latest.values():
        if str(record.get("item_id") or "").strip() == item_id:
            return record
    return None


def is_user_discarded(record: dict | None) -> bool:
    """User が確定した「捨てる」か（P2-2: 自動承認の抑制に使う）。"""
    if not record:
        return False
    return (
        str(record.get("decision") or "") == "discard"
        and str(record.get("classified_by") or "") == "user"
    )


def is_approved_today(record: dict | None) -> bool:
    """「今日やる」確定＋実装承認済みか（P2-1/P2-3: キュー優先に使う）。"""
    if not record:
        return False
    return (
        str(record.get("decision") or "") == "today"
        and bool(str(record.get("approved_at") or "").strip())
    )
