from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()

_LEDGER_PATH = Path(__file__).parent.parent / "rule_engine" / "ledger_rules.json"


def _load_rules() -> list[dict[str, Any]]:
    if not _LEDGER_PATH.exists():
        raise HTTPException(status_code=500, detail="ledger_rules.json が見つかりません")
    with open(_LEDGER_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_rules(rules: list[dict[str, Any]]) -> None:
    with open(_LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
        f.write("\n")


@router.get("/rule-engine/rules")
def list_rules() -> dict[str, Any]:
    rules = _load_rules()
    return {"rules": rules, "total": len(rules)}


@router.patch("/rule-engine/rules/{rule_id}/approve")
def approve_rule(rule_id: str) -> dict[str, Any]:
    rules = _load_rules()
    for rule in rules:
        if rule.get("rev_id") == rule_id:
            rule["pending_review"] = False
            _save_rules(rules)
            return {"rev_id": rule_id, "pending_review": False}
    raise HTTPException(status_code=404, detail=f"ルールが見つかりません: {rule_id}")
