import sqlite3
import json
import datetime
import os

db_path = "/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/data/lease_data.db"
os.makedirs(os.path.dirname(db_path), exist_ok=True)

conn = sqlite3.connect(db_path)
conn.execute("CREATE TABLE IF NOT EXISTS past_cases (id TEXT PRIMARY KEY, timestamp TEXT, data JSON)")

dummy_cases = [
    {"id": "case1", "timestamp": "2026-03-01T10:00:00", "final_status": "成約", "score": 92.5, "industry_major": "建設業", "industry_sub": "設備工事業", "competitor_name": "Xリース", "acquisition_cost": 150000, "final_rate": 2.8, "base_rate_at_time": 2.1},
    {"id": "case2", "timestamp": "2026-03-02T11:00:00", "final_status": "失注", "score": 65.0, "industry_major": "運送業", "industry_sub": "一般貨物", "competitor_name": "Yリース", "acquisition_cost": 320000, "final_rate": 4.5, "base_rate_at_time": 2.1},
    {"id": "case3", "timestamp": "2026-03-03T14:00:00", "final_status": "成約", "score": 88.0, "industry_major": "製造業", "industry_sub": "金属加工業", "competitor_name": "Zリース", "acquisition_cost": 80000, "final_rate": 3.1, "base_rate_at_time": 2.1},
    {"id": "case4", "timestamp": "2026-03-04T09:30:00", "final_status": "失注", "score": 45.0, "industry_major": "小売業", "industry_sub": "アパレル", "competitor_name": "Xリース", "acquisition_cost": 45000, "final_rate": 4.8, "base_rate_at_time": 2.1},
    {"id": "case5", "timestamp": "2026-03-05T16:00:00", "final_status": "成約", "score": 81.5, "industry_major": "医療・福祉", "industry_sub": "病院", "competitor_name": "", "acquisition_cost": 550000, "final_rate": 2.5, "base_rate_at_time": 2.1},
    {"id": "case6", "timestamp": "2026-03-06T13:15:00", "final_status": "成約", "score": 75.0, "industry_major": "建設業", "industry_sub": "土木工事業", "competitor_name": "Yリース", "acquisition_cost": 220000, "final_rate": 3.8, "base_rate_at_time": 2.1},
    {"id": "case7", "timestamp": "2026-03-07T10:45:00", "final_status": "失注", "score": 68.0, "industry_major": "サービス業", "industry_sub": "ホテル・旅館", "competitor_name": "Zリース", "acquisition_cost": 120000, "final_rate": 4.2, "base_rate_at_time": 2.1},
    {"id": "case8", "timestamp": "2026-03-08T15:20:00", "final_status": "成約", "score": 95.0, "industry_major": "情報通信業", "industry_sub": "ソフトウェア制作", "competitor_name": "", "acquisition_cost": 60000, "final_rate": 3.5, "base_rate_at_time": 2.1},
    {"id": "case9", "timestamp": "2026-03-09T09:10:00", "final_status": "失注", "score": 52.0, "industry_major": "運送業", "industry_sub": "倉庫業", "competitor_name": "Xリース", "acquisition_cost": 400000, "final_rate": 4.1, "base_rate_at_time": 2.1},
    {"id": "case10", "timestamp": "2026-03-10T11:50:00", "final_status": "未登録", "score": 78.5, "industry_major": "製造業", "industry_sub": "食品製造業", "competitor_name": "Yリース", "acquisition_cost": 180000, "final_rate": 0, "base_rate_at_time": 2.1},
]

for c in dummy_cases:
    # Need to simulate the "inputs" dictionary format
    c["inputs"] = {"acquisition_cost": c["acquisition_cost"]}
    del c["acquisition_cost"]
    
    conn.execute("INSERT OR REPLACE INTO past_cases (id, timestamp, data) VALUES (?, ?, ?)", 
                 (c["id"], c["timestamp"], json.dumps(c)))

conn.commit()
conn.close()

print("Dummy cases generated to db!")
