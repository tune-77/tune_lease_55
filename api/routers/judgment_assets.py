"""判断資産ルール参照・効果記録エンドポイント (REV-235)

GET  /api/judgment-assets/rules            — ルール一覧（業種・共通フィルタ）
GET  /api/judgment-assets/rules/match      — 案件業種に合うルールを返す
GET  /api/judgment-assets/rules/{rule_id}  — 1件詳細＋効果ログ
POST /api/judgment-assets/rules/{rule_id}/effectiveness  — 効果記録（効いた/微妙/外した/修正）
GET  /api/judgment-assets/stats            — 全体サマリー
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.db_connection import get_connection

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RULES_PATH = _REPO_ROOT / "data" / "judgment_asset_rules_v2.json"

router = APIRouter(tags=["judgment-assets"])

VALID_RESULTS = {"worked", "marginal", "missed", "revised"}


# ── DB 初期化 ─────────────────────────────────────────────────────────────────

def _ensure_table():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS judgment_asset_effectiveness (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id     TEXT    NOT NULL,
                case_id     TEXT    DEFAULT '',
                result      TEXT    NOT NULL,
                note        TEXT    DEFAULT '',
                recorded_at TEXT    DEFAULT (datetime('now', 'localtime'))
            )
        """)
        conn.commit()


# ── ルール読み込み ─────────────────────────────────────────────────────────────

def _load_rules() -> list[dict]:
    if not _RULES_PATH.exists():
        return []
    with open(_RULES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rules", [])


def _get_rule(rule_id: str) -> dict:
    for r in _load_rules():
        if r["id"] == rule_id:
            return r
    raise HTTPException(status_code=404, detail=f"rule_id '{rule_id}' not found")


def _get_effectiveness_log(rule_id: str) -> list[dict]:
    _ensure_table()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, case_id, result, note, recorded_at "
            "FROM judgment_asset_effectiveness WHERE rule_id = ? ORDER BY recorded_at DESC",
            (rule_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _aggregate_effectiveness(rule_id: str) -> str:
    """最新10件のresultから集計して代表値を返す。"""
    log = _get_effectiveness_log(rule_id)[:10]
    if not log:
        return "none"
    counts: dict[str, int] = {}
    for entry in log:
        counts[entry["result"]] = counts.get(entry["result"], 0) + 1
    return max(counts, key=lambda k: counts[k])


# ── Pydantic ──────────────────────────────────────────────────────────────────

class EffectivenessRequest(BaseModel):
    result: str          # worked | marginal | missed | revised
    case_id: str = ""
    note: str = ""


# ── エンドポイント ─────────────────────────────────────────────────────────────

@router.get("/api/judgment-assets/rules")
def list_rules(
    is_common: Optional[bool] = None,
    industry: Optional[str] = None,
    concept: Optional[str] = None,
):
    """
    ルール一覧を返す。
    - is_common=true  → 共通ルールのみ
    - is_common=false → 業種別ルールのみ
    - industry=製造業 → 製造業タグを持つルール（共通も含む）
    - concept=...     → conceptでフィルタ
    """
    _ensure_table()
    rules = _load_rules()

    if is_common is not None:
        rules = [r for r in rules if r["is_common"] == is_common]

    if industry:
        rules = [
            r for r in rules
            if r["is_common"] or industry in r.get("industry_tags", [])
        ]

    if concept:
        rules = [r for r in rules if concept in r.get("concept", "")]

    # 効果記録の集計を付加（軽量版）
    result = []
    for r in rules:
        entry = {
            "id": r["id"],
            "concept": r["concept"],
            "label": r["label"],
            "is_common": r["is_common"],
            "industry_tags": r.get("industry_tags", []),
            "condition": r["condition"],
            "check_points": r["check_points"],
            "memo_sentence": r["memo_sentence"],
            "exclusion_conditions": r["exclusion_conditions"],
            "canonical_statement": r.get("canonical_statement", ""),
            "effectiveness": _aggregate_effectiveness(r["id"]),
            "confidence": r.get("confidence", 0.0),
        }
        result.append(entry)

    return {"total": len(result), "rules": result}


@router.get("/api/judgment-assets/rules/match")
def match_rules(industry_major: str = "", concept: Optional[str] = None):
    """
    案件の業種（industry_major）に合うルールを返す。
    共通ルール + 業種タグが一致するルールを返す。
    """
    _ensure_table()
    rules = _load_rules()

    matched = [
        r for r in rules
        if r["is_common"] or (industry_major and industry_major in r.get("industry_tags", []))
    ]

    if concept:
        matched = [r for r in matched if concept in r.get("concept", "")]

    result = []
    for r in matched:
        result.append({
            "id": r["id"],
            "concept": r["concept"],
            "label": r["label"],
            "is_common": r["is_common"],
            "industry_tags": r.get("industry_tags", []),
            "condition": r["condition"],
            "check_points": r["check_points"],
            "memo_sentence": r["memo_sentence"],
            "exclusion_conditions": r["exclusion_conditions"],
            "canonical_statement": r.get("canonical_statement", ""),
            "effectiveness": _aggregate_effectiveness(r["id"]),
        })

    return {
        "industry_major": industry_major,
        "total": len(result),
        "common": sum(1 for r in result if r["is_common"]),
        "industry_specific": sum(1 for r in result if not r["is_common"]),
        "rules": result,
    }


@router.get("/api/judgment-assets/rules/{rule_id}")
def get_rule(rule_id: str):
    """1件の詳細＋効果記録ログを返す。"""
    _ensure_table()
    r = _get_rule(rule_id)
    log = _get_effectiveness_log(rule_id)
    return {
        **r,
        "effectiveness_summary": _aggregate_effectiveness(rule_id),
        "effectiveness_log": log,
    }


@router.post("/api/judgment-assets/rules/{rule_id}/effectiveness")
def record_effectiveness(rule_id: str, req: EffectivenessRequest):
    """
    判断型を実際に使った結果を記録する。
    result: worked | marginal | missed | revised
    """
    if req.result not in VALID_RESULTS:
        raise HTTPException(
            status_code=422,
            detail=f"result は {sorted(VALID_RESULTS)} のいずれかで指定してください",
        )
    _get_rule(rule_id)  # 存在確認
    _ensure_table()

    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO judgment_asset_effectiveness (rule_id, case_id, result, note) VALUES (?, ?, ?, ?)",
            (rule_id, req.case_id, req.result, req.note),
        )
        conn.commit()
        new_id = cursor.lastrowid

    return {
        "status": "recorded",
        "id": new_id,
        "rule_id": rule_id,
        "result": req.result,
        "case_id": req.case_id,
        "effectiveness_summary": _aggregate_effectiveness(rule_id),
    }


@router.get("/api/judgment-assets/stats")
def get_stats():
    """全ルールの効果サマリーを返す。"""
    _ensure_table()
    rules = _load_rules()

    stats = []
    for r in rules:
        log = _get_effectiveness_log(r["id"])
        counts: dict[str, int] = {}
        for entry in log:
            counts[entry["result"]] = counts.get(entry["result"], 0) + 1
        stats.append({
            "id": r["id"],
            "label": r["label"],
            "concept": r["concept"],
            "is_common": r["is_common"],
            "industry_tags": r.get("industry_tags", []),
            "total_uses": len(log),
            "worked": counts.get("worked", 0),
            "marginal": counts.get("marginal", 0),
            "missed": counts.get("missed", 0),
            "revised": counts.get("revised", 0),
            "effectiveness_summary": _aggregate_effectiveness(r["id"]),
        })

    # 使用回数降順でソート
    stats.sort(key=lambda x: x["total_uses"], reverse=True)

    return {
        "total_rules": len(rules),
        "total_uses": sum(s["total_uses"] for s in stats),
        "rules": stats,
    }
