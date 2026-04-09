import os

# APIを更新するコード (Result Registration Full Fields)
code_to_update = """

# ── 案件結果登録 (成約/失注) - 拡張版
class CaseRegistration(BaseModel):
    case_id: str
    status: str  # "成約" or "失注"
    final_rate: float = 0.0
    base_rate_at_time: float = 2.1
    lost_reason: str = ""
    loan_conditions: list[str] = []
    competitor_name: str = ""
    competitor_rate: float = 0.0
    note: str = ""

@app.post("/api/cases/register")
def register_case_result(req: CaseRegistration):
    from data_cases import load_all_cases, update_case
    cases = load_all_cases()
    
    target_case_id = None
    for c in cases:
        if c.get("id") == req.case_id or c.get("company_name") == req.case_id:
            target_case_id = c.get("id")
            break
            
    if not target_case_id:
        raise HTTPException(status_code=404, detail="Case not found")
        
    patches = {
        "final_status": req.status,
        "final_rate": req.final_rate,
        "base_rate_at_time": req.base_rate_at_time,
        "loan_conditions": req.loan_conditions,
        "competitor_name": req.competitor_name,
        "competitor_rate": req.competitor_rate if req.competitor_rate > 0 else None,
        "final_note": req.note,
    }
    if req.status == "成約" and req.final_rate > 0:
        patches["winning_spread"] = req.final_rate - req.base_rate_at_time
    if req.status == "失注":
        patches["lost_reason"] = req.lost_reason

    if update_case(target_case_id, patches):
        # 自動最適化ロジックなどのトリガー（もしあれば）
        return {"status": "success", "message": f"Results updated for {target_case_id}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update DB")

"""

# main.py の当該箇所を置換するか、追記する。
# 今回は簡単のため、前回追記した部分を特定して置換するために、一時的なスクリプトを作成
with open('/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/api/main.py', 'r') as f:
    content = f.read()

import re
# 前回の簡易版 register_case_result を置換
pattern = r'class CaseRegistration\(BaseModel\):.*?return \{"status": "success", "message": f"Status updated to \{req.status\}"\}'
new_content = re.sub(pattern, code_to_update.strip(), content, flags=re.DOTALL)

with open('/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/api/main.py', 'w') as f:
    f.write(new_content)
print("Updated register_case_result API with full fields")
