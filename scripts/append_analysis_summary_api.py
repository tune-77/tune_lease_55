import os

# API拡張コード (Analysis Status Summary)
code_to_add = """

@app.get("/api/analysis/status_summary")
def get_analysis_status_summary():
    from auto_optimizer import get_training_status
    try:
        s = get_training_status()
        # 追加情報の統合（今何件、あと何件）
        res = {
            "auto_opt": {
                "count": s["count"],
                "min": 50,
                "rem": s["next_trigger"] if s["count"] < 50 else max(0, 20 - (s["count"] - s["last_trained_count"]))
            },
            "quant_ml": {
                "count": s["count"],
                "min": 50,
                "rem": max(0, 50 - s["count"])
            },
            "qual_contrib": {
                "count": s["count"],
                "min": 50,
                "rem": max(0, 50 - s["count"])
            }
        }
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""

with open('/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/api/main.py', 'a') as f:
    f.write(code_to_add)

print("Appended /api/analysis/status_summary to main.py")
