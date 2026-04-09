import os

# APIに追加するコード (Phase 15.4)
code_to_append = """

# =============================================================================
# システム管理・マスタ API (Phase 15.4)
# =============================================================================

# ── 基準金利マスタ
@app.get("/api/settings/interest")
def get_interest_rates():
    from base_rate_master import list_base_rates
    return list_base_rates(limit=60)

class InterestRateUpdate(BaseModel):
    month: str
    rate: float
    note: str = ""

@app.post("/api/settings/interest")
def update_interest_rate(req: InterestRateUpdate):
    from base_rate_master import upsert_base_rate
    upsert_base_rate(req.month, req.rate, req.note)
    return {"status": "success"}

# ── 案件結果登録 (成約/失注)
class CaseRegistration(BaseModel):
    case_id: str
    status: str  # "成約" or "失注"
    note: str = ""

@app.post("/api/cases/register")
def register_case_result(req: CaseRegistration):
    from data_cases import load_all_cases, save_case_log
    cases = load_all_cases()
    # Find the case in SQLite (this project usually identifies by company_name + timestamp or similar)
    # for simplicity in this logic, we search in the list
    target_case = None
    for c in cases:
        if c.get("case_id") == req.case_id or (c.get("company_name") == req.case_id):
            target_case = c
            break
            
    if not target_case:
        raise HTTPException(status_code=404, detail="Case not found")
        
    target_case["final_status"] = req.status
    target_case["final_note"] = req.note
    # Logic to update in DB - using save_case_log which handles SQLite update
    save_case_log(target_case.get("inputs", {}), target_case.get("result", {}), status=req.status)
    return {"status": "success", "message": f"Status updated to {req.status}"}

# ── アプリログ
@app.get("/api/logs/app")
def get_app_logs():
    log_path = "streamlit.log"
    if not os.path.exists(log_path):
        return {"logs": []}
    with open(log_path, "r", encoding="utf-8") as f:
        # Return last 100 lines
        lines = f.readlines()
        return {"logs": [line.strip() for line in lines[-100:]]}

"""

with open('/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/api/main.py', 'a') as f:
    f.write(code_to_append)
print("Appended Phase 15.4 APIs to main.py")
