"""Explain how recalled memories should influence Shion's answer.

This module is deliberately heuristic and local. It does not decide truth; it
turns selected memory metadata into short, inspectable influence hints.
"""
from __future__ import annotations

from typing import Any


def impact_hint_for_record(record: dict[str, Any]) -> str:
    layer = str(record.get("memory_layer") or "retrieval")
    mtype = str(record.get("memory_type") or "")
    status = str(record.get("status") or "active")
    source = str(record.get("source_path") or record.get("source") or "")

    if status == "revised":
        return "旧記憶として扱い、現在の結論を確認してから使う"
    if status == "stale":
        return "鮮度確認が必要な前提として弱く使う"
    if layer == "persistent":
        return "紫苑の運用原則として回答の境界条件に使う"
    if layer == "long_term":
        return "繰り返し確認された判断軸として結論の理由に使う"
    if layer == "mid_term":
        return "最近の流れとして、現在の依頼との接続に使う"
    if mtype == "judgment_memory":
        return "審査・改善判断の観点として確認項目へ落とす"
    if mtype == "technical_memory":
        return "実装・運用の制約として手順に反映する"
    if source:
        return "検索で見つけた根拠として、必要な範囲だけ参照する"
    return "関連記憶として自然に反映する"


def build_memory_impact_hints(recalled: dict[str, Any], *, limit: int = 5) -> list[dict[str, str]]:
    hints: list[dict[str, str]] = []
    for record in recalled.get("memories") or []:
        if not isinstance(record, dict):
            continue
        rid = str(record.get("id") or "")
        if not rid:
            continue
        hints.append(
            {
                "id": rid,
                "memory_layer": str(record.get("memory_layer") or "retrieval"),
                "memory_type": str(record.get("memory_type") or ""),
                "status": str(record.get("status") or "active"),
                "source_path": str(record.get("source_path") or ""),
                "hint": impact_hint_for_record(record),
            }
        )
        if len(hints) >= limit:
            break
    return hints
