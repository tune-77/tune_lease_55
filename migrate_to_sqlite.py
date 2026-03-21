import os
import sqlite3
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_REPO_ROOT, "lease_logic_sumaho12", "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

DB_PATH = os.path.join(DATA_DIR, "lease_data.db")
CASES_FILE = os.path.join(_REPO_ROOT, "past_cases.jsonl")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 審査履歴テーブル
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS past_cases (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            industry_sub TEXT,
            score REAL,
            user_eq REAL,
            final_status TEXT,
            data TEXT
        )
    """)
    conn.commit()
    return conn

def migrate_cases(conn):
    if not os.path.exists(CASES_FILE):
        print(f"File not found: {CASES_FILE}")
        return

    cursor = conn.cursor()
    count = 0
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            case_id = data.get("id")
            if not case_id:
                continue

            timestamp = data.get("timestamp", "")
            industry_sub = data.get("industry_sub", "")
            
            # Extract score and user_eq from result object if present
            score = None
            user_eq = None
            res = data.get("result", {})
            if isinstance(res, dict):
                score = res.get("score")
                user_eq = res.get("user_eq")

            # Fallbacks or conversions
            try:
                score_val = float(score) if score is not None else None
            except (ValueError, TypeError):
                score_val = None

            try:
                user_eq_val = float(user_eq) if user_eq is not None else None
            except (ValueError, TypeError):
                user_eq_val = None

            final_status = data.get("final_status", "")
            json_str = json.dumps(data, ensure_ascii=False)

            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO past_cases
                    (id, timestamp, industry_sub, score, user_eq, final_status, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (case_id, timestamp, industry_sub, score_val, user_eq_val, final_status, json_str))
                count += 1
            except Exception as e:
                print(f"Error inserting {case_id}: {e}")

    conn.commit()
    print(f"Migrated {count} records to past_cases table.")

if __name__ == "__main__":
    print("Starting migration to SQLite...")
    conn = init_db()
    migrate_cases(conn)
    
    # Verify
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM past_cases")
    count = cursor.fetchone()[0]
    print(f"Total records in DB: {count}")
    conn.close()
    print("Done!")
