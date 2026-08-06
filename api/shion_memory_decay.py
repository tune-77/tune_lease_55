"""
紫苑の記憶減衰エンジン（REV-219）

毎日 04:00 に実行され、shion_memory_index.json の全レコードに
freshness_score（0.0〜1.0）を計算し、at_risk フラグを立てる。

減衰ロジック:
  - 最終参照からの経過日数が主たる減衰要因
  - usage_log（shion_memory_usage_log.jsonl）で最近参照されていれば回復
  - confidence が高い記憶は緩やかに衰える
  - 0.2 以下 → at_risk = True（恐れフェーズで使用）

出力: data/shion_memory_freshness.jsonl に追記（1行1レコード）
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_INDEX_PATH = _REPO_ROOT / "data" / "shion_memory_index.json"
_USAGE_LOG_PATH = _REPO_ROOT / "data" / "shion_memory_usage_log.jsonl"
_FRESHNESS_PATH = _REPO_ROOT / "data" / "shion_memory_freshness.jsonl"

# freshness が 1.0 → 0.5 になるまでの日数（半減期）
_HALF_LIFE_DAYS = 30

# at_risk 判定閾値
_AT_RISK_THRESHOLD = 0.2

# 最近7日以内に参照された記憶はスコアを加算
_RECENT_USAGE_DAYS = 7
_RECENT_USAGE_BOOST = 0.25


def _load_recent_used_ids(within_days: int = _RECENT_USAGE_DAYS) -> set[str]:
    """usage_log から直近 N 日以内に参照された memory id を返す。

    api/shion_memory_recall.py の _append_usage_log が書く実際のスキーマは
    {"ts": ..., "refs": [id, ...]} であり、以前この関数が読んでいた
    "used_at"/"memory_id" というキーは存在しない。そのためブーストが
    本番で一度も適用されないリグレッションになっていた。
    """
    if not _USAGE_LOG_PATH.exists():
        return set()

    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(days=within_days)

    used_ids: set[str] = set()
    for line in _USAGE_LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            ts_str = entry.get("ts", "")
            if not ts_str:
                continue
            used_at = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
            if used_at >= cutoff:
                used_ids.update(str(ref) for ref in entry.get("refs") or [] if ref)
        except Exception:
            pass
    return used_ids


def _calc_freshness(record: dict[str, Any], now: datetime, recent_used_ids: set[str]) -> float:
    """
    記憶 1 件の freshness_score を計算する。

    score = base_decay * confidence_factor + usage_boost (capped at 1.0)

    base_decay: 指数関数的減衰 exp(-λ * days_elapsed)
                λ = ln(2) / HALF_LIFE_DAYS
    """
    # ── 最終参照日の解決 ────────────────────────────────────
    last_used_str = record.get("last_used_at") or record.get("created_at") or ""
    if last_used_str:
        try:
            last_used = datetime.fromisoformat(last_used_str.replace("Z", "")).replace(tzinfo=None)
        except ValueError:
            last_used = now
    else:
        last_used = now

    days_elapsed = max(0.0, (now - last_used).total_seconds() / 86400)

    # ── 指数減衰 ────────────────────────────────────────────
    lam = math.log(2) / _HALF_LIFE_DAYS
    base_decay = math.exp(-lam * days_elapsed)

    # ── confidence が高いほど緩やかに衰える（係数 0.8〜1.2） ─
    confidence = float(record.get("confidence", 0.7))
    confidence_factor = 0.8 + confidence * 0.4  # confidence=1.0 → factor=1.2

    score = base_decay * confidence_factor

    # ── 最近参照ブースト ─────────────────────────────────────
    mem_id = record.get("id", "")
    if mem_id and mem_id in recent_used_ids:
        score = min(1.0, score + _RECENT_USAGE_BOOST)

    return round(min(1.0, max(0.0, score)), 4)


def run_memory_decay_batch() -> dict[str, Any]:
    """
    全記憶の freshness_score を計算し shion_memory_freshness.jsonl に書き出す。
    at_risk（score < 0.2）の件数も返す。
    """
    if not _INDEX_PATH.exists():
        logger.warning("[MemoryDecay] shion_memory_index.json が見つかりません")
        return {"status": "skipped", "reason": "index not found"}

    try:
        index = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"[MemoryDecay] index 読み込みエラー: {e}")
        return {"status": "error", "detail": str(e)}

    records: list[dict] = index.get("records", [])
    if not records:
        return {"status": "ok", "total": 0, "at_risk": 0}

    now = datetime.utcnow()
    recent_used_ids = _load_recent_used_ids()
    logger.info(f"[MemoryDecay] 開始: {len(records)}件, 直近使用ID={len(recent_used_ids)}件")

    at_risk_ids: list[str] = []
    results: list[dict] = []

    for rec in records:
        if rec.get("status") == "deprecated":
            continue

        score = _calc_freshness(rec, now, recent_used_ids)
        at_risk = score < _AT_RISK_THRESHOLD

        result = {
            "id": rec.get("id", ""),
            "content_preview": rec.get("content", "")[:60],
            "memory_type": rec.get("memory_type", ""),
            "freshness_score": score,
            "at_risk": at_risk,
            "calculated_at": now.isoformat(),
        }
        results.append(result)

        if at_risk:
            at_risk_ids.append(rec.get("id", ""))

    # ── 追記（1スナップショット = 1行） ─────────────────────
    snapshot = {
        "snapshot_at": now.isoformat(),
        "total": len(results),
        "at_risk_count": len(at_risk_ids),
        "at_risk_ids": at_risk_ids[:20],  # 上位20件のみ保持
        "records": results,
    }

    with _FRESHNESS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    logger.info(f"[MemoryDecay] 完了: at_risk={len(at_risk_ids)}件")
    return {
        "status": "ok",
        "total": len(results),
        "at_risk": len(at_risk_ids),
        "at_risk_ids": at_risk_ids[:5],
    }


def get_latest_freshness_snapshot() -> dict[str, Any] | None:
    """
    shion_memory_freshness.jsonl の最新スナップショットを返す。
    ファイルがなければ None。
    """
    if not _FRESHNESS_PATH.exists():
        return None
    last_line = ""
    for line in _FRESHNESS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last_line = line.strip()
    if not last_line:
        return None
    try:
        return json.loads(last_line)
    except Exception:
        return None


def get_at_risk_memories(limit: int = 5) -> list[dict[str, Any]]:
    """
    最新スナップショットから at_risk な記憶を freshness_score 昇順で返す。
    恐れプロンプト生成（REV-221）から呼ばれる。
    """
    snapshot = get_latest_freshness_snapshot()
    if not snapshot:
        return []

    at_risk = [r for r in snapshot.get("records", []) if r.get("at_risk")]
    at_risk.sort(key=lambda r: r.get("freshness_score", 1.0))
    return at_risk[:limit]
