import os

# APIに追加するコード
code_to_append = """

# =============================================================================
# 高度分析 API (Phase 15.3: 計数分析・ログ・マスタ)
# =============================================================================

# ── 履歴分析・ダッシュボード
@app.get("/api/analysis/contract_drivers")
def api_contract_drivers():
    from analysis_regression import run_contract_driver_analysis
    res = run_contract_driver_analysis()
    if res is None:
        raise HTTPException(status_code=400, detail="Not enough data (minimum 5 closed cases required).")
    
    # JSON直列化のための細かな変換処理
    # closed_cases は大きすぎるので概要だけ返す
    return {
        "closed_count": res.get("closed_count"),
        "avg_financials": res.get("avg_financials"),
        "tag_ranking": res.get("tag_ranking"),
        "top3_drivers": res.get("top3_drivers"),
        "qualitative_summary": {
            "avg_weighted": res.get("qualitative_summary", {}).get("avg_weighted"),
            "n_with_qual": res.get("qualitative_summary", {}).get("n_with_qual"),
            "rank_distribution": res.get("qualitative_summary", {}).get("rank_distribution"),
        } if res.get("qualitative_summary") else None
    }

# ── 定量要因分析
@app.get("/api/analysis/quantitative")
def api_quantitative():
    from analysis_regression import run_quantitative_contract_analysis
    res = run_quantitative_contract_analysis()
    if res is None:
        raise HTTPException(status_code=400, detail="Not enough data (minimum 50 cases required).")
    return res

# ── 定性要因分析
@app.post("/api/analysis/qualitative")
def api_qualitative():
    from analysis_regression import run_qualitative_contract_analysis
    from category_config import QUALITATIVE_SCORING_CORRECTION_ITEMS
    res = run_qualitative_contract_analysis(QUALITATIVE_SCORING_CORRECTION_ITEMS)
    if res is None:
        raise HTTPException(status_code=400, detail="Not enough data.")
    return res

# ── 設定系 (企業番号設定API)
class CorporateSettings(BaseModel):
    api_key: str = ""
    default_number: str = ""
    auto_fetch: bool = True

_CORPORATE_CONFIG_FILE = "corporate_config.json"

@app.get("/api/settings/corporate_number", response_model=CorporateSettings)
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

@app.post("/api/settings/corporate_number")
def save_corporate_settings(req: CorporateSettings):
    import json
    with open(_CORPORATE_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(req.model_dump(), f, ensure_ascii=False, indent=2)
    return {"status": "success"}

# ── マスタ系 (ビジネスルール設定)
@app.get("/api/settings/rules")
def get_rules():
    from rule_manager import load_business_rules
    return load_business_rules()

@app.post("/api/settings/rules")
def save_rules(rules: dict):
    from rule_manager import save_business_rules
    save_business_rules(rules)
    return {"status": "success"}

# ── 係数変更履歴
@app.get("/api/logs/coefficient")
def get_coefficient_history():
    from category_config import get_recent_coefficient_edits
    return get_recent_coefficient_edits(limit=50)

"""

with open('/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/api/main.py', 'a') as f:
    f.write(code_to_append)
print("Appended Admin APIs to main.py")
