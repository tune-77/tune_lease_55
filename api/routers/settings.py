"""設定・マスタ管理ルーター (REV-234 Phase4)"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["settings"])

# ── 設定系 (企業番号設定API)
class CorporateSettings(BaseModel):
    api_key: str = ""
    default_number: str = ""
    auto_fetch: bool = True

_CORPORATE_CONFIG_FILE = "corporate_config.json"

@router.get("/api/settings/corporate_number", response_model=CorporateSettings)
def get_corporate_settings():
    import json
    if os.path.exists(_CORPORATE_CONFIG_FILE):
        try:
            with open(_CORPORATE_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return CorporateSettings(**data)
        except Exception:
            pass
    return CorporateSettings()

@router.post("/api/settings/corporate_number")
def save_corporate_settings(req: CorporateSettings):
    import json
    with open(_CORPORATE_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(req.model_dump(), f, ensure_ascii=False, indent=2)
    return {"status": "success"}

# ── マスタ系 (ビジネスルール設定)
@router.get("/api/settings/rules")
def get_rules():
    from rule_manager import load_business_rules
    return load_business_rules()

@router.post("/api/settings/rules")
def save_rules(rules: dict):
    from rule_manager import save_business_rules
    save_business_rules(rules)
    return {"status": "success"}

# ── 係数変更履歴
@router.get("/api/logs/coefficient")
def get_coefficient_history():
    from category_config import get_recent_coefficient_edits
    return get_recent_coefficient_edits(limit=50)



# =============================================================================
# システム管理・マスタ API (Phase 15.4)
# =============================================================================

# ── 基準金利マスタ
@router.get("/api/settings/interest")
def get_interest_rates():
    from base_rate_master import list_base_rates
    return list_base_rates(limit=60)

class InterestRateUpdate(BaseModel):
    month: str
    rate: float = 0.0
    note: str = ""
    r_2y: float | None = None
    r_3y: float | None = None
    r_4y: float | None = None
    r_5y: float | None = None
    r_6y: float | None = None
    r_7y: float | None = None
    r_8y: float | None = None
    r_9y: float | None = None
    r_over9y: float | None = None

@router.post("/api/settings/interest")
def update_interest_rate(req: InterestRateUpdate):
    from base_rate_master import upsert_base_rate
    upsert_base_rate(
        req.month, req.rate, req.note,
        r_2y=req.r_2y, r_3y=req.r_3y, r_4y=req.r_4y, r_5y=req.r_5y,
        r_6y=req.r_6y, r_7y=req.r_7y, r_8y=req.r_8y, r_9y=req.r_9y,
        r_over9y=req.r_over9y,
    )
    return {"status": "success"}

@router.post("/api/settings/interest/seed")
def seed_interest_rates(overwrite: bool = False):
    from base_rate_master import seed_initial_data
    inserted, skipped = seed_initial_data(overwrite=overwrite)
    return {"inserted": inserted, "skipped": skipped}

@router.get("/api/settings/interest/current")
def get_current_interest():
    from base_rate_master import get_base_rate_by_term, list_base_rates
    import datetime
    today = datetime.date.today()
    current_month = today.strftime("%Y-%m")
    next_month = (today.replace(day=1) + datetime.timedelta(days=32)).strftime("%Y-%m")
    recent = list_base_rates(limit=2)
    return {
        "current_month": current_month,
        "next_month": next_month,
        "current_rate_5y": get_base_rate_by_term(current_month, 60),
        "next_rate_5y": get_base_rate_by_term(next_month, 60),
        "latest": recent[0] if recent else None,
        "prev": recent[1] if len(recent) > 1 else None,
    }

